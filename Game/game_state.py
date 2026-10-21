"""
Pizza Kitchen - game rules (Track B). RULES ONLY: no OpenCV, no keyboard, no printing.
Objects and the two player classes live in game_objects.py.
The keyboard / OpenCV test window lives in keyboard_test.py.

HOW THE FINAL GAME USES THIS
    from game_objects import Chef, SousChef
    from game_state import GameState, Command

    state = GameState(chef=my_chef, sous_chef=my_sous_chef)   # or GameState() for defaults
    state.apply(Command(player_id, "chop", None, time.time())) # from keyboard or camera
    events = state.update(time.time())                         # call every frame
    state.set_visible(player_id, False)                        # player left the frame
    state.mood_level, state.score, state.time_left, state.game_over   # read for display

Your own player classes only need:  .id  .role_key ("chef"/"sous_chef")  .holding  .visible  .can(action)
(easiest: subclass Chef / SousChef from game_objects.py).
"""
import random
from dataclasses import dataclass
from typing import Optional

from game_objects import (TOPPINGS, CHOP_STRIKES, Ingredient, Pizza, Order, Oven, Bank,
                          Chef, SousChef, CHEF, SOUS_CHEF)

# =====================================================================
# SETTINGS - change the "to be decided" numbers here, nowhere else
# =====================================================================
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
    player: int              # the player's id
    action: str
    arg: Optional[str] = None
    t: float = 0.0           # time in seconds (use time.time() for real play)


# =====================================================================
# ABILITIES  (can_activate returns (ok, reason))
# role is "chef" or "sous_chef" and is matched against player.role_key
# =====================================================================
class Ability:
    name = ""
    role = ""
    cooldown = 10.0

    def can_activate(self, game):
        return True, ""

    def activate(self, game):
        raise NotImplementedError


class KnifeFrenzy(Ability):
    name, role, cooldown = "knife_frenzy", "sous_chef", ABILITY_1_COOLDOWN

    def activate(self, game):
        game.knife_frenzy_left = KNIFE_FRENZY_DURATION


class RushDelivery(Ability):
    name, role, cooldown = "rush_delivery", "sous_chef", ABILITY_2_COOLDOWN

    def can_activate(self, game):
        if game.pizza.stage != "baked":
            return False, "no baked pizza to deliver"
        return True, ""

    def activate(self, game):
        game._deliver()


class HotOven(Ability):
    name, role, cooldown = "hot_oven", "chef", ABILITY_1_COOLDOWN

    def activate(self, game):
        game.hot_oven_left = HOT_OVEN_DURATION


class Stubborn(Ability):
    name, role, cooldown = "stubborn", "chef", ABILITY_2_COOLDOWN

    def activate(self, game):
        game._reset_pizza_and_order(clear_bank=STUBBORN_CLEARS_BANK)
        game.events.append("order_replaced")


# =====================================================================
# GAME STATE
# =====================================================================
BANK_ACTIONS = {"pass_to_bank", "take_pizza_from_bank", "take_from_bank", "place_on_bank"}


class GameState:
    def __init__(self, chef=None, sous_chef=None, seed=None):
        # The two players. Pass your own objects in, or leave empty for the defaults.
        self.chef = chef if chef is not None else Chef(CHEF)
        self.sous = sous_chef if sous_chef is not None else SousChef(SOUS_CHEF)
        self.players = {self.chef.id: self.chef, self.sous.id: self.sous}

        self.rng = random.Random(seed)
        self.time_left = START_TIME
        self.score = 0
        self.game_over = False
        self.order = Order.random(self.rng)
        self.pizza = Pizza()
        self.oven = Oven()
        self.bank = Bank()
        self.abilities = {a.name: a for a in (KnifeFrenzy(), RushDelivery(), HotOven(), Stubborn())}
        self.cooldowns = {name: 0.0 for name in self.abilities}
        self.knife_frenzy_left = 0.0
        self.hot_oven_left = 0.0
        self.events = []
        self.last_message = ""
        self.last_now = None
        self.last_bank_use = None                        # (time, player id)

    # ---------- read-only views for other tasks (display, camera, ...) ----------
    @property
    def holding(self):
        """{player_id: item or None}"""
        return {pid: p.holding for pid, p in self.players.items()}

    @property
    def visible(self):
        """{player_id: bool}"""
        return {pid: p.visible for pid, p in self.players.items()}

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
        player = self.players.get(cmd.player)
        if player is None:
            return self._no(f"unknown player {cmd.player}")
        if self.game_over:
            return self._no("game over")
        if not player.visible:
            return self._no("that player is lost from the frame (actions paused)")
        if not player.can(cmd.action):
            return self._no(f"{player.role_name} cannot {cmd.action}")
        if cmd.action in BANK_ACTIONS and self._bank_busy(cmd):
            return self._no("bank busy: the other player used it first")
        handler = getattr(self, "_do_" + cmd.action, None)
        if handler is None:
            return self._no(f"unknown action {cmd.action}")
        return handler(cmd)

    # ---------- Sous Chef ----------
    def _do_grab_raw(self, cmd):
        if self.sous.holding is not None:
            return self._no("hands full")
        if cmd.arg not in TOPPINGS:
            return self._no(f"unknown ingredient {cmd.arg}")
        self.sous.holding = Ingredient(cmd.arg)
        return self._ok(f"Sous Chef grabbed raw {cmd.arg}")

    def _do_chop(self, cmd):
        item = self.sous.holding
        if not isinstance(item, Ingredient) or item.state != "raw":
            return self._no("need to hold a raw ingredient to chop")

        # 2 strikes if Knife Frenzy is active, otherwise 1
        item.strikes += 2 if self.knife_frenzy_left > 0 else 1

        if item.strikes >= CHOP_STRIKES:
            item.state = "chopped"
            return self._ok(f"{item.type} is chopped")
        return self._ok(f"chop strike ({item.strikes}/{CHOP_STRIKES})")

    def _do_pass_to_bank(self, cmd):
        item = self.sous.holding
        if not isinstance(item, Ingredient) or item.state != "chopped":
            return self._no("need to hold a chopped ingredient")
        if not self.bank.put(item):
            return self._no("bank is full")
        self.sous.holding = None
        self._mark_bank(cmd)
        return self._ok(f"{item.type} put on the bank")

    def _do_take_pizza_from_bank(self, cmd):
        if self.sous.holding is not None:
            return self._no("hands full")
        if self.pizza not in self.bank.items:
            return self._no("no pizza on the bank")
        self.bank.take(self.pizza)
        self.sous.holding = self.pizza
        self._mark_bank(cmd)
        return self._ok("Sous Chef took the pizza from the bank")

    def _do_deliver(self, cmd):
        item = self.sous.holding
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
        if self.chef.holding is not None:
            return self._no("hands full")
        needed = [i for i in self.bank.items
                  if isinstance(i, Ingredient) and i.type in self.order.toppings
                  and i.type not in self.pizza.toppings]
        if not needed:
            return self._no("no needed ingredient on the bank")
        item = needed[0]
        self.bank.take(item)
        self.chef.holding = item
        self._mark_bank(cmd)
        return self._ok(f"Chef took {item.type} from the bank")

    def _do_add_topping(self, cmd):
        item = self.chef.holding
        if not isinstance(item, Ingredient) or item.state != "chopped":
            return self._no("need to hold a chopped ingredient")
        if self.pizza.stage != "sauced":
            return self._no("pizza must be sauced first (and not yet topped)")
        if item.type not in self.order.toppings or item.type in self.pizza.toppings:
            return self._no(f"order does not need {item.type}")
        self.pizza.toppings.append(item.type)
        self.chef.holding = None
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
        if self.chef.holding is not None:
            return self._no("hands full")
        self.chef.holding = self.oven.take()
        return self._ok("Chef took the baked pizza out")

    def _do_place_on_bank(self, cmd):
        item = self.chef.holding
        if not isinstance(item, Pizza) or item.stage != "baked":
            return self._no("need to hold a baked pizza")
        if not self.bank.put(item):
            return self._no("bank is full")
        self.chef.holding = None
        self._mark_bank(cmd)
        return self._ok("baked pizza put on the bank")

    # ---------- abilities ----------
    def _do_ability(self, cmd):
        player = self.players[cmd.player]
        ab = self.abilities.get(cmd.arg)
        if ab is None:
            return self._no(f"unknown ability {cmd.arg}")
        if ab.role != player.role_key:
            return self._no(f"{player.role_name} cannot use {ab.name}")
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
        """Removes old pizza from oven, hands, and bank, then generates a new order and dough."""
        old = self.pizza
        if self.oven.pizza is old:
            self.oven.take()
        for p in self.players.values():
            if p.holding is old:
                p.holding = None
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

    def set_visible(self, player_id, visible):
        """Task 3 calls this when a player leaves/re-enters the frame. What they hold stays."""
        self.players[player_id].visible = visible
        self.events.append(f"player_{player_id}_{'back' if visible else 'lost'}")

    def _ok(self, msg):
        self.last_message = "OK: " + msg
        return True

    def _no(self, msg):
        self.last_message = "REFUSED: " + msg
        return False

    def __str__(self):
        def hold(p):
            return str(p.holding) if p.holding else "nothing"
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
                f"  Chef holds: {hold(self.chef)} | Sous Chef holds: {hold(self.sous)}\n"
                f"  active={active} cooldowns={cds}")