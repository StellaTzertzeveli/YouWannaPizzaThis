
import random
import time
from dataclasses import dataclass
from typing import Optional

from game_objects import (TOPPINGS, CHOP_STRIKES, BANK_CAPACITY, Ingredient, Pizza, Order, Oven, Bank)

# =====================================================================
# SETTINGS - change the "to be decided" numbers here, nowhere else
# =====================================================================
CHEF = 1
SOUS_CHEF = 2

START_TIME = 60.0
DELIVERY_BONUS = 10.0
BURN_PENALTY = 5.0
MAX_TIME = 90.0              # TBD: maximum timer value


ABILITY_1_COOLDOWN = 10.0
ABILITY_2_COOLDOWN = 20.0
KNIFE_FRENZY_DURATION = 4.0  # TBD
HOT_OVEN_DURATION = 4.0      # TBD
HOT_OVEN_SPEEDUP = 2.0       # baking runs 2x faster while active

CLEAR_BANK_ON_BURN = False   # TBD: do items on the bank disappear when a pizza burns?
STUBBORN_CLEARS_BANK = False # TBD: does Stubborn clear the bank?
BANK_CONFLICT_WINDOW = 0.05  # seconds: two bank actions this close = same moment, first wins
MOOD_THRESHOLDS = (40.0, 20.0)  # time left above 40 -> mood 0, above 20 -> mood 1, else mood 2


# =====================================================================
# COMMAND (what the keyboard or the camera sends in)
# =====================================================================
@dataclass
class Command:
    player: int
    action: str
    arg: Optional[str] = None
    t: float = 0.0           # time in seconds (use time.time() for real play)


# =====================================================================
# ABILITIES  (can_activate returns (ok, reason))
# =====================================================================
class Ability:
    name = ""
    role = 0
    cooldown = 10.0

    def can_activate(self, game):
        return True, ""

    def activate(self, game):
        raise NotImplementedError


class KnifeFrenzy(Ability):
    name, role, cooldown = "knife_frenzy", SOUS_CHEF, ABILITY_1_COOLDOWN

    def activate(self, game):
        game.knife_frenzy_left = KNIFE_FRENZY_DURATION


class RushDelivery(Ability):
    name, role, cooldown = "rush_delivery", SOUS_CHEF, ABILITY_2_COOLDOWN

    def can_activate(self, game):
        if game.pizza.stage != "baked":
            return False, "no baked pizza to deliver"
        return True, ""

    def activate(self, game):
        game._deliver()


class HotOven(Ability):
    name, role, cooldown = "hot_oven", CHEF, ABILITY_1_COOLDOWN

    def activate(self, game):
        game.hot_oven_left = HOT_OVEN_DURATION


class Stubborn(Ability):
    name, role, cooldown = "stubborn", CHEF, ABILITY_2_COOLDOWN

    def activate(self, game):
        game._reset_pizza_and_order(clear_bank=STUBBORN_CLEARS_BANK)
        game.events.append("order_replaced")


# =====================================================================
# GAME STATE
# =====================================================================
ROLE_ACTIONS = {
    SOUS_CHEF: {"grab_raw", "chop", "pass_to_bank", "take_pizza_from_bank", "deliver", "ability"},
    CHEF: {"spin_dough", "place_at_station", "add_topping", "oven_in", "oven_out",
           "take_from_bank", "place_on_bank", "ability"},
}
BANK_ACTIONS = {"pass_to_bank", "take_pizza_from_bank", "take_from_bank", "place_on_bank"}


class GameState:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.time_left = START_TIME
        self.score = 0
        self.game_over = False
        self.order = Order.random(self.rng)
        self.pizza = Pizza()
        self.oven = Oven()
        self.bank = Bank()
        self.holding = {CHEF: None, SOUS_CHEF: None}     # one item per player
        self.visible = {CHEF: True, SOUS_CHEF: True}     # False = player lost from frame
        self.abilities = {a.name: a for a in (KnifeFrenzy(), RushDelivery(), HotOven(), Stubborn())}
        self.cooldowns = {name: 0.0 for name in self.abilities}
        self.knife_frenzy_left = 0.0
        self.hot_oven_left = 0.0
        self.events = []
        self.last_message = ""
        self.last_now = None
        self.last_bank_use = None                        # (time, player)

    # ---------- time ----------
    def _advance(self, now):
        if self.last_now is None:
            self.last_now = now
            return
        dt = max(0.0, now - self.last_now)
        self.last_now = now
        if self.game_over or dt == 0:
            return
        self.time_left -= dt
        self.knife_frenzy_left = max(0.0, self.knife_frenzy_left - dt)
        for name in self.cooldowns:
            self.cooldowns[name] = max(0.0, self.cooldowns[name] - dt)
        speed = HOT_OVEN_SPEEDUP if self.hot_oven_left > 0 else 1.0
        self.hot_oven_left = max(0.0, self.hot_oven_left - dt)
        for ev in self.oven.update(now, speed):
            if ev == "baked":
                self.events.append("pizza_baked")
            elif ev == "burnt":
                self._burn()
        if self.time_left <= 0:
            self.time_left = 0.0
            self.game_over = True
            self.events.append("game_over")

    def update(self, now):
        """Call every frame. Returns the list of events since the last call."""
        self._advance(now)
        ev, self.events = self.events, []
        return ev

    @property
    def mood_level(self):
        """0 = calm, 1 = worried, 2 = angry/sad. Send this to Task 1."""
        if self.time_left > MOOD_THRESHOLDS[0]:
            return 0
        if self.time_left > MOOD_THRESHOLDS[1]:
            return 1
        return 2

    # ---------- main entry ----------
    def apply(self, cmd):
        """Do a command if allowed. Returns True if it did something, False if refused."""
        self._advance(cmd.t)
        if self.game_over:
            return self._no("game over")
        if not self.visible[cmd.player]:
            return self._no("that player is lost from the frame (actions paused)")
        if cmd.action not in ROLE_ACTIONS[cmd.player]:
            return self._no(f"{self._role(cmd.player)} cannot {cmd.action}")
        if cmd.action in BANK_ACTIONS and self._bank_busy(cmd):
            return self._no("bank busy: the other player used it first")
        return getattr(self, "_do_" + cmd.action)(cmd)

    # ---------- Sous Chef ----------
    def _do_grab_raw(self, cmd):
        if self.holding[SOUS_CHEF] is not None:
            return self._no("hands full")
        if cmd.arg not in TOPPINGS:
            return self._no(f"unknown ingredient {cmd.arg}")
        self.holding[SOUS_CHEF] = Ingredient(cmd.arg)
        return self._ok(f"Sous Chef grabbed raw {cmd.arg}")

    def _do_chop(self, cmd):
        item = self.holding[SOUS_CHEF]
        if not isinstance(item, Ingredient) or item.state != "raw":
            return self._no("need to hold a raw ingredient to chop")
        item.strikes += 2 if self.knife_frenzy_left > 0 else 1
        if item.strikes >= CHOP_STRIKES:
            item.state = "chopped"
            return self._ok(f"{item.type} is chopped")
        return self._ok(f"chop strike ({item.strikes}/{CHOP_STRIKES})")

    def _do_pass_to_bank(self, cmd):
        item = self.holding[SOUS_CHEF]
        if not isinstance(item, Ingredient) or item.state != "chopped":
            return self._no("need to hold a chopped ingredient")
        if not self.bank.put(item):
            return self._no("bank is full")
        self.holding[SOUS_CHEF] = None
        self._mark_bank(cmd)
        return self._ok(f"{item.type} put on the bank")

    def _do_take_pizza_from_bank(self, cmd):
        if self.holding[SOUS_CHEF] is not None:
            return self._no("hands full")
        if self.pizza not in self.bank.items:
            return self._no("no pizza on the bank")
        self.bank.take(self.pizza)
        self.holding[SOUS_CHEF] = self.pizza
        self._mark_bank(cmd)
        return self._ok("Sous Chef took the pizza from the bank")

    def _do_deliver(self, cmd):
        item = self.holding[SOUS_CHEF]
        if not isinstance(item, Pizza) or item.stage != "baked":
            return self._no("need to hold a baked pizza")
        return self._deliver()

    # ---------- Chef ----------
    def _do_spin_dough(self, cmd):
        if self.pizza.stage != "dough":
            return self._no("dough already spun")
        self.pizza.stage = "spun"
        return self._ok("dough spun")

    def _do_place_at_station(self, cmd):
        if self.pizza.stage != "spun":
            return self._no("need spun dough first")
        self.pizza.stage = "sauced"
        return self._ok("dough placed at station, sauce added")

    def _do_take_from_bank(self, cmd):
        if self.holding[CHEF] is not None:
            return self._no("hands full")
        needed = [i for i in self.bank.items
                  if isinstance(i, Ingredient) and i.type in self.order.toppings
                  and i.type not in self.pizza.toppings]
        if not needed:
            return self._no("no needed ingredient on the bank")
        item = needed[0]
        self.bank.take(item)
        self.holding[CHEF] = item
        self._mark_bank(cmd)
        return self._ok(f"Chef took {item.type} from the bank")

    def _do_add_topping(self, cmd):
        item = self.holding[CHEF]
        if not isinstance(item, Ingredient) or item.state != "chopped":
            return self._no("need to hold a chopped ingredient")
        if self.pizza.stage != "sauced":
            return self._no("pizza must be sauced first (and not yet topped)")
        if item.type not in self.order.toppings or item.type in self.pizza.toppings:
            return self._no(f"order does not need {item.type}")
        self.pizza.toppings.append(item.type)
        self.holding[CHEF] = None
        if self.order.is_satisfied_by(self.pizza):
            self.pizza.stage = "topped"
            return self._ok(f"added {item.type}, pizza is fully topped")
        return self._ok(f"added {item.type}")

    def _do_oven_in(self, cmd):
        if self.pizza.stage != "topped":
            return self._no("pizza must be fully topped")
        if self.oven.pizza is not None:
            return self._no("oven is busy")
        self.pizza.stage = "baking"
        self.oven.put(self.pizza, cmd.t)
        return self._ok("pizza in the oven")

    def _do_oven_out(self, cmd):
        if self.oven.pizza is None:
            return self._no("oven is empty")
        if self.oven.pizza.stage != "baked":
            return self._no("pizza is not baked yet")
        if self.holding[CHEF] is not None:
            return self._no("hands full")
        self.holding[CHEF] = self.oven.take()
        return self._ok("Chef took the baked pizza out")

    def _do_place_on_bank(self, cmd):
        item = self.holding[CHEF]
        if not isinstance(item, Pizza) or item.stage != "baked":
            return self._no("need to hold a baked pizza")
        if not self.bank.put(item):
            return self._no("bank is full")
        self.holding[CHEF] = None
        self._mark_bank(cmd)
        return self._ok("baked pizza put on the bank")

    # ---------- abilities ----------
    def _do_ability(self, cmd):
        ab = self.abilities.get(cmd.arg)
        if ab is None:
            return self._no(f"unknown ability {cmd.arg}")
        if ab.role != cmd.player:
            return self._no(f"{self._role(cmd.player)} cannot use {ab.name}")
        if self.cooldowns[ab.name] > 0:
            return self._no(f"{ab.name} on cooldown ({self.cooldowns[ab.name]:.1f}s)")
        ok, reason = ab.can_activate(self)
        if not ok:
            return self._no(reason)
        ab.activate(self)
        self.cooldowns[ab.name] = ab.cooldown
        self.events.append(f"ability_used:{ab.name}")
        return self._ok(f"{ab.name} activated")

    # ---------- shared helpers ----------
    def _deliver(self):
        if not self.order.is_satisfied_by(self.pizza):
            return self._no("pizza does not match the order")
        self.score += 1
        self.time_left = min(MAX_TIME, self.time_left + DELIVERY_BONUS)
        self.events.append("delivered")
        self._reset_pizza_and_order()
        return self._ok("pizza delivered! +1 score")

    def _burn(self):
        self.events.append("pizza_burnt")
        self.time_left -= BURN_PENALTY
        self._reset_pizza_and_order(clear_bank=CLEAR_BANK_ON_BURN)

    def _reset_pizza_and_order(self, clear_bank=False):
        """New order + new dough. Removes the old pizza from oven, hands and bank."""
        old = self.pizza
        if self.oven.pizza is old:
            self.oven.take()
        for p in self.holding:
            if self.holding[p] is old:
                self.holding[p] = None
        self.bank.take(old)
        if clear_bank:
            self.bank.items.clear()
        self.pizza = Pizza()
        self.order = Order.random(self.rng)
        self.events.append("new_order")

    def _bank_busy(self, cmd):
        if self.last_bank_use is None:
            return False
        t, player = self.last_bank_use
        return player != cmd.player and abs(cmd.t - t) < BANK_CONFLICT_WINDOW

    def _mark_bank(self, cmd):
        self.last_bank_use = (cmd.t, cmd.player)

    def set_visible(self, player, visible):
        """Task 3 calls this when a player leaves/re-enters the frame. What they hold stays."""
        self.visible[player] = visible
        self.events.append(f"player_{player}_{'back' if visible else 'lost'}")

    @staticmethod
    def _role(player):
        return "Chef" if player == CHEF else "Sous Chef"

    def _ok(self, msg):
        self.last_message = "OK: " + msg
        return True

    def _no(self, msg):
        self.last_message = "REFUSED: " + msg
        return False

    def __str__(self):
        def hold(p):
            return str(self.holding[p]) if self.holding[p] else "nothing"
        oven = f"{self.oven.pizza} {self.oven.elapsed:.1f}s" if self.oven.pizza else "empty"
        active = []
        if self.knife_frenzy_left > 0:
            active.append(f"knife_frenzy {self.knife_frenzy_left:.1f}s")
        if self.hot_oven_left > 0:
            active.append(f"hot_oven {self.hot_oven_left:.1f}s")
        cds = {k: round(v, 1) for k, v in self.cooldowns.items() if v > 0}
        return (f"{self.last_message}\n"
                f"  time={self.time_left:.1f} score={self.score} mood={self.mood_level}"
                f"{' GAME OVER' if self.game_over else ''}\n"
                f"  order={self.order.toppings}  {self.pizza}\n"
                f"  oven={oven}  bank={[str(i) for i in self.bank.items]}\n"
                f"  Chef holds: {hold(CHEF)} | Sous Chef holds: {hold(SOUS_CHEF)}\n"
                f"  active={active} cooldowns={cds}")


# =====================================================================
# TEST RUNNER (only runs when you run THIS file directly)
# =====================================================================
KEYMAP = {
    "1": (SOUS_CHEF, "grab_raw", TOPPINGS[0]), "2": (SOUS_CHEF, "grab_raw", TOPPINGS[1]),
    "3": (SOUS_CHEF, "grab_raw", TOPPINGS[2]), "4": (SOUS_CHEF, "grab_raw", TOPPINGS[3]),
    "f": (SOUS_CHEF, "chop", None), "g": (SOUS_CHEF, "pass_to_bank", None),
    "h": (SOUS_CHEF, "take_pizza_from_bank", None), "j": (SOUS_CHEF, "deliver", None),
    "z": (SOUS_CHEF, "ability", "knife_frenzy"), "x": (SOUS_CHEF, "ability", "rush_delivery"),
    "u": (CHEF, "spin_dough", None), "i": (CHEF, "place_at_station", None),
    "o": (CHEF, "add_topping", None), "k": (CHEF, "oven_in", None),
    "l": (CHEF, "oven_out", None), "p": (CHEF, "take_from_bank", None),
    ";": (CHEF, "place_on_bank", None),
    "n": (CHEF, "ability", "hot_oven"), "m": (CHEF, "ability", "stubborn"),
}


def selftest():
    """Full flow: 3 pizzas delivered, 1 burnt, every ability used once."""
    g = GameState(seed=1)
    t = [0.0]
    log = []

    def do(player, action, arg=None):
        t[0] += 0.2
        ok = g.apply(Command(player, action, arg, t[0]))
        assert ok, f"{action} refused: {g.last_message}"

    def wait(sec):
        t[0] += sec
        log.extend(g.update(t[0]))

    def build_pizza():
        for top in list(g.order.toppings):
            do(SOUS_CHEF, "grab_raw", top)
            for _ in range(CHOP_STRIKES):
                if g.holding[SOUS_CHEF].state == "chopped":
                    break
                do(SOUS_CHEF, "chop")
            do(SOUS_CHEF, "pass_to_bank")
        do(CHEF, "spin_dough")
        do(CHEF, "place_at_station")
        for _ in list(g.order.toppings):
            do(CHEF, "take_from_bank")
            do(CHEF, "add_topping")
        assert g.pizza.stage == "topped"

    g.update(0.0)
    # Pizza 1: Knife Frenzy, normal bake, normal delivery
    do(SOUS_CHEF, "ability", "knife_frenzy")
    build_pizza(); do(CHEF, "oven_in"); wait(6)
    do(CHEF, "oven_out"); do(CHEF, "place_on_bank"); do(SOUS_CHEF, "take_pizza_from_bank"); do(SOUS_CHEF, "deliver")
    # Pizza 2: Hot Oven, then Rush Delivery straight from the oven
    build_pizza(); do(CHEF, "ability", "hot_oven"); do(CHEF, "oven_in"); wait(3)
    do(SOUS_CHEF, "ability", "rush_delivery")
    # Pizza 3: left in the oven, burns
    build_pizza(); do(CHEF, "oven_in"); wait(11)
    # Pizza 4: Stubborn replaces the order, then normal delivery
    do(CHEF, "ability", "stubborn")
    build_pizza(); do(CHEF, "oven_in"); wait(6)
    do(CHEF, "oven_out"); do(CHEF, "place_on_bank"); do(SOUS_CHEF, "take_pizza_from_bank"); do(SOUS_CHEF, "deliver")

    log.extend(g.update(t[0]))
    print(g)
    assert g.score == 3, g.score
    assert g.events.count("pizza_burnt") + log.count("pizza_burnt") >= 1
    print("\nSELFTEST PASSED: 3 delivered, 1 burnt, all 4 abilities used.")


def draw_window(state, events_log, offset_note=""):
    """Draws the key legend (left) and the live game state (right) into an image."""
    import textwrap
    import numpy as np
    import cv2
    img = np.full((600, 1040, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, x, y, color=(220, 220, 220), size=0.45):
        cv2.putText(img, text, (x, y), font, size, color, 1, cv2.LINE_AA)

    # left: keys
    y = 24
    put("KEYS  (click this window first)", 10, y, (255, 255, 255), 0.5)
    for k, (p, a, arg) in KEYMAP.items():
        y += 19
        color = (255, 180, 120) if p == CHEF else (120, 220, 120)
        put(f"{k}  {'Chef' if p == CHEF else 'Sous'}  {a} {arg or ''}", 10, y, color)
    y += 26
    put("w = skip 5 s    9 = Sous lost/back", 10, y, (180, 180, 180))
    y += 19
    put("0 = Chef lost/back   t = selftest (terminal)", 10, y, (180, 180, 180))
    y += 19
    put("ESC = quit", 10, y, (180, 180, 180))

    # right: state
    x, y = 400, 24
    put("GAME STATE", x, y, (255, 255, 255), 0.5)
    lines = str(state).split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            color = (120, 220, 120) if line.startswith("OK") else (90, 90, 255)
        else:
            color = (220, 220, 220)
        for part in textwrap.wrap(line, 68) or [""]:
            y += 20
            put(part, x, y, color, 0.5 if i == 0 else 0.45)
    y += 34
    put("RECENT EVENTS", x, y, (255, 255, 255), 0.5)
    for ev in events_log[-8:]:
        y += 19
        put(ev, x, y, (120, 200, 255))
    return img


def main():
    """OpenCV window: reads keys, turns them into commands, prints the state after every command."""
    import cv2
    state = GameState()
    offset = 0.0
    now = lambda: time.time() + offset
    events_log = []
    state.update(now())
    print("Window opened. Click it, then press keys. ESC quits. State prints here after every command.")
    print(state)
    while True:
        cv2.imshow("Pizza Kitchen - keyboard test", draw_window(state, events_log))
        k = cv2.waitKey(30) & 0xFF
        t = now()

        for ev in state.update(t):              # timer events: baked, burnt, game_over...
            events_log.append(ev)
            print("EVENT:", ev)

        if k == 27:                             # ESC
            break
        if k == 255:                            # no key pressed
            continue
        key = chr(k).lower()

        if key == "w":
            offset += 5
            print("\n(skipped 5 seconds)")
        elif key == "9":
            state.set_visible(SOUS_CHEF, not state.visible[SOUS_CHEF])
        elif key == "0":
            state.set_visible(CHEF, not state.visible[CHEF])
        elif key == "t":
            selftest()
            continue
        elif key in KEYMAP:
            player, action, arg = KEYMAP[key]
            cmd = Command(player, action, arg, t)
            print(f"\nCOMMAND: player={player} {action} {arg or ''}")
            state.apply(cmd)
        else:
            continue

        print(state)                            # state after EVERY command
        for ev in state.update(t):
            events_log.append(ev)
            print("EVENT:", ev)
    cv2.destroyAllWindows()


def main_typing():
    """Fallback without OpenCV: type keys and press Enter."""
    state = GameState()
    offset = 0.0
    now = lambda: time.time() + offset
    state.update(now())
    print("Type keys then Enter. 'wait 5' skips 5 s, 'test' runs selftest, 'q' quits.")
    print(state)
    while True:
        text = input("> ").strip().lower()
        if text == "q":
            break
        if text == "test":
            selftest()
            continue
        if text.startswith("wait"):
            offset += float(text.split()[1])
        t = now()
        for key in ("" if text.startswith("wait") else text):
            if key == "9":
                state.set_visible(SOUS_CHEF, not state.visible[SOUS_CHEF])
            elif key == "0":
                state.set_visible(CHEF, not state.visible[CHEF])
            elif key in KEYMAP:
                p, a, arg = KEYMAP[key]
                state.apply(Command(p, a, arg, t))
        events = state.update(t)
        print(state)
        if events:
            print("EVENTS:", events)


if __name__ == "__main__":
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("OpenCV not installed (pip install opencv-python numpy). Using typing mode instead.")
        main_typing()
    else:
        main()