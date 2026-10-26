"""Pizza Kitchen - Task 4 (Player Interaction and Game Control), all in one file.

HOW TO RUN
    python main.py              webcam game  (q = quit, r = restart)
    python main.py --selftest   check the game rules, no camera needed
    python main.py --camera 1   use another camera (if you get a black picture)

WHAT HAPPENS EVERY FRAME (the pipeline)
    webcam frame
      -> prepare_frame()        mirror + resize to 1280 x 720
      -> FakeInput.read()       keypoints for 2 players        (later: Tasks 2 + 3)
      -> zone_at + DwellTimer   "a hand was held in a zone for 1 s"
      -> zone_to_command()      turn that into a Command       (Track A)
      -> to_rule_command()      translate it for the rules     (the bridge)
      -> GameState.apply()      allowed right now? update game (Track B)
      -> draw_...()             visual feedback on the frame

NOT DONE YET (keyboard stand-ins, see KEYBOARD_ACTIONS in section 9)
    Chopping, dough spinning, sprinkling and the two ability poses need
    motion/pose detectors, which do not exist yet. Until then you press a key.
    The key creates the SAME Command a detector will create later, so when the
    detectors are ready, only section 9 changes.

SECTIONS (search for the ===== lines)
    1  General settings
    2  Commands          (was commands.py)
    3  Zones and timers  (was zones.py)
    4  Fake input        (was fake_input.py)
    5  Game objects      (was game_objects.py)
    6  Game rules        (was game_state.py)
    7  Bridge: camera commands -> rule commands
    8  Zones -> commands (what a held hand means, per role)
    9  Keyboard stand-ins for actions without a detector yet
    10 Drawing / visual feedback (sprite code from KeyboardTest.py)
    11 Self test
    12 Main loop
"""
import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# =====================================================================
# 1. GENERAL SETTINGS
# =====================================================================
CAMERA_INDEX = 0            # 0 = first camera. Black picture? Try 1.
LOST_AFTER_SECONDS = 0.5    # a player missing this long counts as "left the frame"
WINDOW_NAME = "Pizza Kitchen"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Player IDs: Player 1 = Chef (right side), Player 2 = Sous Chef (left side)
CHEF_ID = 1
SOUS_CHEF_ID = 2


# =====================================================================
# 2. COMMANDS (was commands.py)
#
# A command is the "message" the seeing part (zones, detectors, players)
# sends to the rules part (GameState):
#     "player 2 held a hand in the cheese box for 1 s"
#     -> Command(player_id=2, role=SOUS_CHEF, action=GRAB, item=CHEESE)
# The names in this section are the agreed interface: change them only
# if both tracks agree.
# =====================================================================

# Roles: which job a player has. Left side = Sous Chef, right side = Chef.
CHEF = "CHEF"
SOUS_CHEF = "SOUS_CHEF"
ALL_ROLES = (CHEF, SOUS_CHEF)

# Action names: every game action a player can trigger.
GRAB = "GRAB"                       # take a raw ingredient from a box (1 s hold)
PLACE = "PLACE"                     # put the dough on the topping station
CHOP_STRIKE = "CHOP_STRIKE"         # ONE downward strike at the chopping counter
SPIN_DONE = "SPIN_DONE"             # dough spinning finished
SPRINKLE_DONE = "SPRINKLE_DONE"     # 2 s of sprinkling finished (one topping)
OVEN_IN = "OVEN_IN"                 # put the pizza in the oven (1 s hold)
OVEN_OUT = "OVEN_OUT"               # take the pizza out of the oven (1 s hold)
PASS_TO_BANK = "PASS_TO_BANK"       # put an item on the bank for the other player
TAKE_FROM_BANK = "TAKE_FROM_BANK"   # take an item from the bank
DELIVER = "DELIVER"                 # place the baked pizza on the delivery stand
ABILITY_1 = "ABILITY_1"             # arms crossed in an X, held 1 s
ABILITY_2 = "ABILITY_2"             # double thumbs up, held 1 s

ALL_ACTIONS = (
    GRAB, PLACE, CHOP_STRIKE, SPIN_DONE, SPRINKLE_DONE,
    OVEN_IN, OVEN_OUT, PASS_TO_BANK, TAKE_FROM_BANK, DELIVER,
    ABILITY_1, ABILITY_2,
)

# Items: the optional "what" of a command (e.g. GRAB *cheese*).
CHEESE = "cheese"
PEPPERONI = "pepperoni"
MUSHROOMS = "mushrooms"
PEPPERS = "peppers"
DOUGH = "dough"
PIZZA = "pizza"

TOPPINGS = (CHEESE, PEPPERONI, MUSHROOMS, PEPPERS)
ALL_ITEMS = TOPPINGS + (DOUGH, PIZZA)


@dataclass(frozen=True)
class Command:
    """One game action by one player (the camera side's format).

    player_id:  the stable ID from Task 3 (1 or 2)
    role:       CHEF or SOUS_CHEF
    action:     one of ALL_ACTIONS
    item:       one of ALL_ITEMS, or None if the action has no object
    timestamp:  when the action was recognised (seconds, time.monotonic())
    """
    player_id: int
    role: str
    action: str
    item: str | None = None
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # Runs right after a Command is created: catches typos like "GRABB" at once.
        if self.role not in ALL_ROLES:
            raise ValueError(f"Unknown role: {self.role!r}")
        if self.action not in ALL_ACTIONS:
            raise ValueError(f"Unknown action: {self.action!r}")
        if self.item is not None and self.item not in ALL_ITEMS:
            raise ValueError(f"Unknown item: {self.item!r}")


# =====================================================================
# 3. ZONES AND TIMERS (was zones.py)
#
# A zone is a rectangle on the screen where a player can do something,
# e.g. hold a hand in "box_cheese" for 1 s to grab cheese.
# Zone NAMES are agreed with the team; POSITIONS can be tuned here.
# =====================================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# The webcam image is flipped left-right like a mirror, so the left of the
# screen is the left of the players. The Sous Chef stands left.
MIRROR_WEBCAM = True


def prepare_frame(frame):
    """Mirror and resize a webcam frame so it matches the zones.

    Call this FIRST, right after cap.read(). Then keypoints, zones and
    drawings all use the same picture.
    """
    if MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)   # 1 = flip around the vertical axis (left <-> right)
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))


# Zone names. Sous Chef side (left half):
BOX_CHEESE = "box_cheese"
BOX_PEPPERONI = "box_pepperoni"
BOX_MUSHROOMS = "box_mushrooms"
BOX_PEPPERS = "box_peppers"
CHOPPING_COUNTER = "chopping_counter"
DELIVERY_STAND = "delivery_stand"
# Middle, shared by both players:
BANK = "bank"
# Chef side (right half):
BOX_DOUGH = "box_dough"
TOPPING_STATION = "topping_station"
OVEN = "oven"

ALL_ZONES = (
    BOX_CHEESE, BOX_PEPPERONI, BOX_MUSHROOMS, BOX_PEPPERS,
    CHOPPING_COUNTER, DELIVERY_STAND, BANK, BOX_DOUGH, TOPPING_STATION, OVEN,
)

# Each zone is (x1, y1, x2, y2): top-left and bottom-right corner in pixels.
# x grows to the RIGHT, y grows DOWNWARD (y = 0 is the top of the screen).
#
#   +------------------------------+------+------------------------------+
#   | cheese  pepperoni mush peppr |      |   TIMER  (display only)      |
#   |                              |      |   RECIPE (display only)      |
#   |DELIV|                        | BANK |                       |OVEN  |
#   |STAND|       SOUS CHEF        |      |        CHEF           |      |
#   |ABIL1|  CHOPPING COUNTER      |      |DOUGH| TOPPING STATION |ABIL1 |
#   |ABIL2|                        |      | BOX |                 |ABIL2 |
#   +------------------------------+------+------------------------------+
#   0                             600    680                          1280
ZONE_RECTS = {
    BOX_CHEESE:       (140,   0,  245, 130),
    BOX_PEPPERONI:    (255,   0,  360, 130),
    BOX_MUSHROOMS:    (370,   0,  475, 130),
    BOX_PEPPERS:      (485,   0,  590, 130),
    DELIVERY_STAND:   (0,   200,  130, 520),
    CHOPPING_COUNTER: (140, 580,  590, 720),
    BANK:             (600, 200,  680, 520),
    BOX_DOUGH:        (690, 580,  790, 720),   # left of the topping station
    TOPPING_STATION:  (800, 580, 1140, 720),
    OVEN:             (1150, 200, 1280, 520),
}

# Display areas: the game SHOWS information here. A hand here does nothing,
# so zone_at() never returns them.
TIMER_DISPLAY = "timer_display"
RECIPE_DISPLAY = "recipe_display"
SOUS_ABILITY_1_DISPLAY = "sous_ability_1_display"   # Knife Frenzy
SOUS_ABILITY_2_DISPLAY = "sous_ability_2_display"   # Rush Delivery
CHEF_ABILITY_1_DISPLAY = "chef_ability_1_display"   # Hot Oven
CHEF_ABILITY_2_DISPLAY = "chef_ability_2_display"   # Stubborn

DISPLAY_RECTS = {
    TIMER_DISPLAY:          (690,  0, 1140,  60),
    RECIPE_DISPLAY:         (690, 65, 1140, 175),
    SOUS_ABILITY_1_DISPLAY: (0, 580, 130, 645),
    SOUS_ABILITY_2_DISPLAY: (0, 655, 130, 720),
    CHEF_ABILITY_1_DISPLAY: (1150, 580, 1280, 645),
    CHEF_ABILITY_2_DISPLAY: (1150, 655, 1280, 720),
}

# Which ingredient each box gives
BOX_INGREDIENT = {
    BOX_CHEESE: CHEESE,
    BOX_PEPPERONI: PEPPERONI,
    BOX_MUSHROOMS: MUSHROOMS,
    BOX_PEPPERS: PEPPERS,
}


class Zone:
    """A named rectangle a hand can be inside of.

    Example:
        oven = Zone("oven", (1150, 200, 1280, 520))
        oven.contains((1200, 300))  -> True
    """

    def __init__(self, name, rect):
        self.name = name
        self.x1, self.y1, self.x2, self.y2 = rect

    def contains(self, point):
        x, y = point
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


ZONES = [Zone(name, rect) for name, rect in ZONE_RECTS.items()]


def zone_at(x, y):
    """Name of the zone that contains point (x, y), or None.

    Example: zone_at(200, 50) -> "box_cheese"
    """
    for zone in ZONES:
        if zone.contains((x, y)):
            return zone.name
    return None


HOLD_SECONDS = 1.0


class DwellTimer:
    """Fires ONCE when a hand has stayed in the same zone for HOLD_SECONDS.

    Use one DwellTimer per hand per player. Every frame:
        fired = timer.update("box_cheese", now)
        if fired:              # "box_cheese", exactly once per visit
            ...make a command...
    After firing it waits until the hand LEAVES the zone, so keeping your
    hand in the box does not grab again and again.
    """

    def __init__(self, hold_seconds=HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.zone = None          # the zone the hand is in right now
        self.start_time = None    # when the hand entered that zone
        self.fired = False        # already fired for this visit?

    def update(self, zone, now):
        """zone: zone name or None. Returns the zone name on the one frame
        the hold completes, otherwise None."""
        if zone != self.zone:              # entered a new zone, or left one
            self.zone = zone
            self.start_time = now
            self.fired = False
            return None
        if zone is None or self.fired:
            return None
        if now - self.start_time >= self.hold_seconds:
            self.fired = True
            return zone
        return None

    def progress(self, now):
        """0.0 (just entered) to 1.0 (done). Used to draw the hold bar."""
        if self.zone is None:
            return 0.0
        if self.fired:
            return 1.0
        return min((now - self.start_time) / self.hold_seconds, 1.0)


class Cooldown:
    """Blocks an action from firing again too soon. Not used yet: meant for
    the motion detectors (e.g. at most one chop strike every 0.3 s).

        chop_cooldown = Cooldown(0.3)
        if chop_cooldown.ready(now):
            chop_cooldown.trigger(now)
            ...do the chop...
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.last_time = None     # None = never triggered, so ready at once

    def ready(self, now):
        return self.last_time is None or now - self.last_time >= self.seconds

    def trigger(self, now):
        self.last_time = now

    def remaining(self, now):
        if self.ready(now):
            return 0.0
        return self.seconds - (now - self.last_time)


# =====================================================================
# 4. FAKE INPUT (was fake_input.py) - TEMPORARY stand-in for Tasks 2 + 3
#
# Cuts the picture in half (left = Sous Chef, right = Chef) and runs one
# MediaPipe Pose per half. When Tasks 2 and 3 deliver, replace FakeInput
# with their code; as long as read() returns the same format, nothing
# else has to change.
#
# read() returns a list with one dictionary per player:
#   {"player_id": 1, "role": "CHEF", "visible": True,
#    "keypoints": {"nose": (x, y, confidence), "left_wrist": (...), ...}}
# x, y are pixels in the prepared 1280 x 720 frame. If "visible" is False,
# "keypoints" is empty.
# =====================================================================
KEYPOINT_INDEX = {            # MediaPipe Pose numbers its 33 body points
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
WRISTS = ("left_wrist", "right_wrist")
CONFIDENCE_THRESHOLD = 0.5    # below this, a keypoint is not trusted
HALF = FRAME_WIDTH // 2       # 1280 // 2 = 640


def is_reliable(keypoint):
    """True if a keypoint exists and MediaPipe is confident enough about it."""
    return keypoint is not None and keypoint[2] >= CONFIDENCE_THRESHOLD


class FakeInput:
    """Finds two players in a frame and returns their keypoints."""

    def __init__(self):
        # Imported here (not at the top) so --selftest works without mediapipe.
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            # MediaPipe 0.10.30 and newer removed mp.solutions (the old Pose API).
            sys.exit(f"Your MediaPipe ({mp.__version__}) no longer has mp.solutions.pose.\n"
                     "Use the same Python + MediaPipe install that runs fake_input.py,\n"
                     "or ask Task 2 which pose library the team uses.")
        # One Pose detector per half: each remembers its own person between frames.
        self.pose_left = mp.solutions.pose.Pose(model_complexity=1)
        self.pose_right = mp.solutions.pose.Pose(model_complexity=1)

    def read(self, frame):
        """frame: a PREPARED frame. Returns [chef, sous_chef]."""
        left_half = frame[:, :HALF]      # all rows, columns 0 .. 639
        right_half = frame[:, HALF:]     # all rows, columns 640 .. 1279
        sous_chef = self._detect(self.pose_left, left_half, 0, SOUS_CHEF_ID, SOUS_CHEF)
        chef = self._detect(self.pose_right, right_half, HALF, CHEF_ID, CHEF)
        return [chef, sous_chef]

    def _detect(self, pose, image, x_offset, player_id, role):
        player = {"player_id": player_id, "role": role, "visible": False, "keypoints": {}}

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # OpenCV = BGR, MediaPipe = RGB
        result = pose.process(rgb)
        if result.pose_landmarks is None:              # nobody in this half
            return player

        height, width = image.shape[:2]
        for name, index in KEYPOINT_INDEX.items():
            landmark = result.pose_landmarks.landmark[index]
            # MediaPipe gives fractions (0.0 to 1.0); turn them into pixels
            # and shift the right half back into place.
            x = int(landmark.x * width) + x_offset
            y = int(landmark.y * height)
            player["keypoints"][name] = (x, y, landmark.visibility)

        player["visible"] = True
        return player

    def close(self):
        self.pose_left.close()
        self.pose_right.close()


# =====================================================================
# 5. GAME OBJECTS (was game_objects.py)
# Ingredient, Pizza, Order, Oven, Bank and the two player classes.
# No game rules here (who may do what): those are in section 6.
# =====================================================================
# Strikes to chop one ingredient. 6 matches the art in assets/ingredients:
#   0 strikes -> _raw, 1..5 strikes -> _cut_1 .. _cut_5, 6 strikes -> _sliced
CHOP_STRIKES = 6
BAKE_TIME = 5.0              # seconds in the oven until baked
BURN_TIME = 10.0             # seconds in the oven until burnt
BANK_CAPACITY = 4            # TBD: max items on the bank

PIZZA_STAGES = ["dough", "spun", "sauced", "topped", "baking", "baked", "burnt"]


class Player:
    """GameState needs: .id .role_key .holding .visible .can(action)"""
    role_key = ""            # "chef" or "sous_chef"
    role_name = ""           # for messages
    allowed_actions = frozenset()

    def __init__(self, player_id):
        self.id = player_id
        self.holding = None      # one item at a time (Ingredient or Pizza)
        self.visible = True      # False = lost from the camera (actions paused)

    def can(self, action):
        return action in self.allowed_actions

    def __repr__(self):
        return f"{self.role_name}(id={self.id}, holding={self.holding})"


class Chef(Player):
    role_key = "chef"
    role_name = "Chef"
    allowed_actions = frozenset({
        "grab_dough", "spin_dough", "place_at_station", "add_topping", "oven_in", "oven_out",
        "take_from_bank", "place_on_bank", "ability"})


class SousChef(Player):
    role_key = "sous_chef"
    role_name = "Sous Chef"
    allowed_actions = frozenset({
        "grab_raw", "chop", "pass_to_bank", "take_pizza_from_bank", "deliver", "ability"})


@dataclass
class Ingredient:
    type: str
    state: str = "raw"       # "raw" or "chopped"
    strikes: int = 0         # how many chop strikes it has had

    def __post_init__(self):
        if self.type not in TOPPINGS:
            raise ValueError(f"unknown ingredient: {self.type}")

    @property
    def image_name(self):
        """PNG name for this ingredient right now (without .png):
        cheese_raw -> cheese_cut_1 ... -> cheese_sliced"""
        if self.state == "chopped":
            return f"{self.type}_sliced"
        if self.strikes == 0:
            return f"{self.type}_raw"
        return f"{self.type}_cut_{min(self.strikes, CHOP_STRIKES - 1)}"

    def __str__(self):
        if self.state == "raw":
            return f"raw {self.type} ({self.strikes}/{CHOP_STRIKES})"
        return f"chopped {self.type}"


@dataclass
class Pizza:
    stage: str = "dough"
    toppings: list = field(default_factory=list)

    def __post_init__(self):
        if self.stage not in PIZZA_STAGES:
            raise ValueError(f"unknown stage: {self.stage}")

    def __str__(self):
        return f"pizza[{self.stage}, toppings={self.toppings}]"


@dataclass
class Order:
    toppings: list           # 1 to 3 different toppings

    def __post_init__(self):
        if not 1 <= len(self.toppings) <= 3:
            raise ValueError("an order has 1 to 3 toppings")
        if len(set(self.toppings)) != len(self.toppings):
            raise ValueError("no duplicate toppings in an order")

    @staticmethod
    def random(rng=None):
        rng = rng or random
        n = rng.randint(1, 3)
        return Order(rng.sample(TOPPINGS, n))      # sample = no duplicates

    def is_satisfied_by(self, pizza):
        """True if the pizza has exactly the toppings the order asks for."""
        return sorted(pizza.toppings) == sorted(self.toppings)


class Oven:
    """Holds one pizza. update(now) makes it baked at 5 s and burnt at 10 s."""

    def __init__(self):
        self.pizza = None
        self.elapsed = 0.0       # oven seconds so far
        self.last_now = None

    def put(self, pizza, now):
        if self.pizza is not None:
            return False
        self.pizza = pizza
        self.elapsed = 0.0
        self.last_now = now
        return True

    def take(self):
        pizza = self.pizza
        self.pizza, self.elapsed, self.last_now = None, 0.0, None
        return pizza

    def update(self, now, speed=1.0):
        """Call every frame (speed > 1 = Hot Oven).
        Returns what just happened: [], ['baked'], ['burnt'] or ['baked', 'burnt']."""
        happened = []
        if self.pizza is None:
            return happened
        dt = max(0.0, now - self.last_now)
        self.last_now = now
        self.elapsed += dt * speed
        if self.pizza.stage == "baking" and self.elapsed >= BAKE_TIME:
            self.pizza.stage = "baked"
            happened.append("baked")
        if self.elapsed >= BURN_TIME:
            self.pizza.stage = "burnt"
            happened.append("burnt")
        return happened


class Bank:
    """Shared shelf between the two players."""

    def __init__(self, max_items=BANK_CAPACITY):
        self.items = []
        self.max_items = max_items

    def put(self, item):
        """Returns False if the bank is full."""
        if len(self.items) >= self.max_items:
            return False
        self.items.append(item)
        return True

    def take(self, item):
        """Removes that item. Returns False if it is not on the bank."""
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def is_full(self):
        return len(self.items) >= self.max_items


# =====================================================================
# 6. GAME RULES (was game_state.py)
# Rules only: no OpenCV, no keyboard, no printing.
#
#   state = GameState()
#   state.apply(RuleCommand(player_id, "chop", None, now))
#   events = state.update(now)          # call every frame
#   state.set_visible(player_id, False) # player left the frame
#   state.mood_level, state.score, state.time_left, state.game_over
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
MOOD_THRESHOLDS = (40.0, 20.0)  # time left > 40 -> mood 0, > 20 -> mood 1, else mood 2


@dataclass
class RuleCommand:
    """The rules' own command format (was game_state.Command).
    Made from a camera Command by to_rule_command() in section 7."""
    player: int              # the player's id
    action: str              # e.g. "grab_raw", "chop", "ability"
    arg: str | None = None   # e.g. "cheese", or an ability name
    t: float = 0.0           # time in seconds


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


BANK_ACTIONS = {"pass_to_bank", "take_pizza_from_bank", "take_from_bank", "place_on_bank"}


class GameState:
    def __init__(self, chef=None, sous_chef=None, seed=None):
        self.chef = chef if chef is not None else Chef(CHEF_ID)
        self.sous = sous_chef if sous_chef is not None else SousChef(SOUS_CHEF_ID)
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

    # ---------- read-only views for other tasks ----------
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
        item.strikes += 2 if self.knife_frenzy_left > 0 else 1   # Knife Frenzy = double
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
    def dough_in_box(self):
        """True while the dough for the current order is still in the dough box
        (not taken yet). The dough IS the current pizza, at stage "dough"."""
        return self.pizza.stage == "dough" and self.chef.holding is not self.pizza

    def _do_grab_dough(self, cmd):
        if self.chef.holding is not None:
            return self._no("hands full")
        if not self.dough_in_box():
            return self._no("the dough for this order is already taken")
        self.chef.holding = self.pizza
        return self._ok("Chef grabbed the dough")

    def _do_spin_dough(self, cmd):
        if self.chef.holding is not self.pizza or self.pizza.stage != "dough":
            return self._no("grab dough from the dough box first")
        self.pizza.stage = "spun"
        return self._ok("dough spun")

    def _do_place_at_station(self, cmd):
        if self.chef.holding is not self.pizza or self.pizza.stage != "spun":
            return self._no("need to hold spun dough")
        self.chef.holding = None              # hands free again for toppings
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
        """Removes the old pizza from oven, hands and bank, then makes a new order and dough."""
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
        """Call when a player leaves / re-enters the frame. What they hold stays."""
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


# =====================================================================
# 7. BRIDGE: camera Command -> RuleCommand
#
# The camera side (section 2) and the rules (section 6) were written
# separately and use different names, e.g. GRAB vs "grab_raw".
# Some camera actions mean different rule actions depending on the role:
# PASS_TO_BANK by the Chef = "place_on_bank" (the baked pizza),
# PASS_TO_BANK by the Sous Chef = "pass_to_bank" (a chopped ingredient).
# =====================================================================
RULE_ACTIONS = {
    (SOUS_CHEF, GRAB): "grab_raw",
    (SOUS_CHEF, CHOP_STRIKE): "chop",
    (SOUS_CHEF, PASS_TO_BANK): "pass_to_bank",
    (SOUS_CHEF, TAKE_FROM_BANK): "take_pizza_from_bank",
    (SOUS_CHEF, DELIVER): "deliver",
    (CHEF, GRAB): "grab_dough",
    (CHEF, SPIN_DONE): "spin_dough",
    (CHEF, PLACE): "place_at_station",
    (CHEF, SPRINKLE_DONE): "add_topping",
    (CHEF, OVEN_IN): "oven_in",
    (CHEF, OVEN_OUT): "oven_out",
    (CHEF, TAKE_FROM_BANK): "take_from_bank",
    (CHEF, PASS_TO_BANK): "place_on_bank",
}

ABILITY_NAMES = {
    (SOUS_CHEF, ABILITY_1): "knife_frenzy",
    (SOUS_CHEF, ABILITY_2): "rush_delivery",
    (CHEF, ABILITY_1): "hot_oven",
    (CHEF, ABILITY_2): "stubborn",
}


def to_rule_command(command):
    """Translate a camera Command into a RuleCommand, or None if there is no rule for it.

    Example:
        Command(player_id=2, role=SOUS_CHEF, action=GRAB, item=CHEESE)
        -> RuleCommand(player=2, action="grab_raw", arg="cheese", t=...)
    """
    key = (command.role, command.action)
    if key in ABILITY_NAMES:
        return RuleCommand(command.player_id, "ability", ABILITY_NAMES[key], command.timestamp)
    if key in RULE_ACTIONS:
        return RuleCommand(command.player_id, RULE_ACTIONS[key], command.item, command.timestamp)
    return None


# =====================================================================
# 8. ZONES -> COMMANDS (what a 1 s hold in a zone means)
# Some zones mean different things depending on what is going on:
# a hand in the bank PUTS something down if you hold something, and
# TAKES something if your hands are empty.
# =====================================================================
def zone_to_command(player, zone, state, now):
    """Return the Command for a completed hold in a zone, or None if that
    zone means nothing for this player's role."""
    player_id, role = player["player_id"], player["role"]

    def make(action, item=None):
        return Command(player_id=player_id, role=role, action=action,
                       item=item, timestamp=now)

    if role == SOUS_CHEF:
        if zone in BOX_INGREDIENT:
            return make(GRAB, BOX_INGREDIENT[zone])
        if zone == BANK:
            holding = state.sous.holding
            if isinstance(holding, Ingredient):
                return make(PASS_TO_BANK, holding.type)
            return make(TAKE_FROM_BANK, PIZZA)
        if zone == DELIVERY_STAND:
            return make(DELIVER, PIZZA)

    if role == CHEF:
        if zone == BOX_DOUGH:
            return make(GRAB, DOUGH)
        if zone == TOPPING_STATION:
            return make(PLACE, DOUGH)
        if zone == OVEN:
            if state.oven.pizza is None:
                return make(OVEN_IN, PIZZA)
            return make(OVEN_OUT, PIZZA)
        if zone == BANK:
            if isinstance(state.chef.holding, Pizza):
                return make(PASS_TO_BANK, PIZZA)
            return make(TAKE_FROM_BANK)

    return None


# =====================================================================
# 9. KEYBOARD STAND-INS (until the motion / pose detectors exist)
# key: (player id, role, action). Each key makes the same Command the
# detector will make later.
# =====================================================================
KEYBOARD_ACTIONS = {
    "f": (SOUS_CHEF_ID, SOUS_CHEF, CHOP_STRIKE),   # later: chop motion detector
    "z": (SOUS_CHEF_ID, SOUS_CHEF, ABILITY_1),     # later: arms-X pose
    "x": (SOUS_CHEF_ID, SOUS_CHEF, ABILITY_2),     # later: double thumbs up
    "u": (CHEF_ID, CHEF, SPIN_DONE),               # later: circle-above-head detector
    "o": (CHEF_ID, CHEF, SPRINKLE_DONE),           # later: sprinkle detector
    "n": (CHEF_ID, CHEF, ABILITY_1),               # later: arms-X pose
    "m": (CHEF_ID, CHEF, ABILITY_2),               # later: double thumbs up
}


def key_to_command(key, now):
    """Command for a pressed key, or None if the key is not a game key."""
    if key not in KEYBOARD_ACTIONS:
        return None
    player_id, role, action = KEYBOARD_ACTIONS[key]
    return Command(player_id=player_id, role=role, action=action, timestamp=now)


# =====================================================================
# 10. DRAWING / VISUAL FEEDBACK
# Ingredient pictures: assets/ingredients/<name>.png next to this file,
# e.g. cheese_raw.png, cheese_cut_1.png ... cheese_sliced.png.
# Missing files are created as simple placeholders on the first run.
# =====================================================================
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ingredients")
SPRITE_SIZE = 128             # size of the generated placeholder PNGs
CROP_TO_CONTENT = True        # trim empty transparent space around your art
CROP_PADDING = 0.06           # room kept around the ingredient (fraction of its size)
INGREDIENT_COLORS = {         # BGR, only for the placeholders
    CHEESE: (60, 200, 250), PEPPERONI: (60, 60, 210),
    MUSHROOMS: (150, 175, 200), PEPPERS: (60, 180, 60),
}
PIZZA_COLOURS = {             # BGR fill colour of the pizza per stage
    "dough": (170, 220, 245), "spun": (170, 220, 245), "sauced": (60, 60, 200),
    "topped": (60, 90, 210), "baking": (60, 120, 220), "baked": (40, 120, 200),
    "burnt": (30, 30, 30),
}
MOOD_COLOURS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255)}  # green, orange, red
WHITE = (255, 255, 255)
ZONE_COLOUR = (0, 200, 255)
ZONE_HINTS = {                # shown inside the zones while the detectors are missing
    CHOPPING_COUNTER: "F = chop",
    TOPPING_STATION: "U = spin dough   O = sprinkle",
}
ABILITY_DISPLAYS = {          # display area: (ability name, label)
    SOUS_ABILITY_1_DISPLAY: ("knife_frenzy", "Z Knife Frenzy"),
    SOUS_ABILITY_2_DISPLAY: ("rush_delivery", "X Rush Deliv."),
    CHEF_ABILITY_1_DISPLAY: ("hot_oven", "N Hot Oven"),
    CHEF_ABILITY_2_DISPLAY: ("stubborn", "M Stubborn"),
}

_sprite_cache = {}
_crop_cache = {}


def all_ingredient_image_names():
    names = []
    for t in TOPPINGS:
        names.append(f"{t}_raw")
        names += [f"{t}_cut_{i}" for i in range(1, CHOP_STRIKES)]
        names.append(f"{t}_sliced")
    return names


def _make_placeholder(name):
    ing_type, state = name.split("_", 1)
    color = INGREDIENT_COLORS.get(ing_type, (200, 200, 200))
    img = np.full((SPRITE_SIZE, SPRITE_SIZE, 3), 45, dtype=np.uint8)
    c = SPRITE_SIZE // 2
    if state == "sliced":
        for dx, dy in [(-30, -20), (0, -30), (30, -20), (-15, 15), (20, 20)]:
            cv2.ellipse(img, (c + dx, c + dy), (16, 10), 20, 0, 360, color, -1)
    else:
        cv2.circle(img, (c, c), 46, color, -1)
        if state.startswith("cut_"):
            n = int(state.split("_")[1])
            for i in range(n):                       # one cut line per strike
                x = c - 40 + i * (80 // max(n, 1)) + 8
                cv2.line(img, (x, c - 44), (x, c + 44), (30, 30, 30), 2)
    cv2.putText(img, name, (4, SPRITE_SIZE - 6), FONT, 0.32, WHITE, 1, cv2.LINE_AA)
    return img


def ensure_placeholder_assets():
    """Creates any missing ingredient PNGs. Returns how many were created."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    made = 0
    for name in all_ingredient_image_names():
        path = os.path.join(ASSET_DIR, name + ".png")
        if not os.path.exists(path):
            cv2.imwrite(path, _make_placeholder(name))
            made += 1
    return made


def report_assets():
    """Prints which ingredient pictures are your own art and which are still
    generated placeholders, so you can see what is missing."""
    mine, placeholders, missing = [], [], []
    for name in all_ingredient_image_names():
        path = os.path.join(ASSET_DIR, name + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is None:
            missing.append(name)
        # A placeholder is exactly SPRITE_SIZE x SPRITE_SIZE with no transparency
        elif img.shape[:2] == (SPRITE_SIZE, SPRITE_SIZE) and _alpha_box(img) is None:
            placeholders.append(name)
        else:
            mine.append(name)
    print(f"Ingredient pictures in {ASSET_DIR}")
    print(f"  your own art: {len(mine)}   placeholders: {len(placeholders)}   missing: {len(missing)}")
    if placeholders:
        print("  still placeholders:", ", ".join(placeholders))
    if missing:
        print("  missing:", ", ".join(missing))


def _alpha_box(img):
    """Where the visible (non-transparent) pixels are, as fractions (x0, y0, x1, y1).
    None if the image has no transparency."""
    if img.ndim < 3 or img.shape[2] != 4:
        return None
    alpha = img[:, :, 3]
    if alpha.min() > 250:
        return None
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    h, w = alpha.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _ingredient_crop(ing_type):
    """One crop box shared by ALL pictures of an ingredient, so it keeps the
    same size from picture to picture instead of jumping."""
    if ing_type not in _crop_cache:
        boxes = []
        for name in all_ingredient_image_names():
            if name.startswith(ing_type + "_"):
                img = cv2.imread(os.path.join(ASSET_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
                box = _alpha_box(img) if img is not None else None
                if box:
                    boxes.append(box)
        if boxes:
            x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
            pad_x, pad_y = (x1 - x0) * CROP_PADDING, (y1 - y0) * CROP_PADDING
            _crop_cache[ing_type] = (max(0.0, x0 - pad_x), max(0.0, y0 - pad_y),
                                     min(1.0, x1 + pad_x), min(1.0, y1 + pad_y))
        else:
            _crop_cache[ing_type] = None
    return _crop_cache[ing_type]


def load_sprite(name, size):
    """Loads assets/ingredients/<name>.png fitted inside a size x size box
    (no stretching, extra space transparent). None if the file is missing."""
    key = (name, size)
    if key not in _sprite_cache:
        img = cv2.imread(os.path.join(ASSET_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            _sprite_cache[key] = None
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            h, w = img.shape[:2]
            crop = _ingredient_crop(name.split("_")[0]) if CROP_TO_CONTENT else None
            if crop and _alpha_box(img):
                img = img[int(crop[1] * h):int(crop[3] * h) + 1, int(crop[0] * w):int(crop[2] * w) + 1]
                h, w = img.shape[:2]
            scale = size / max(h, w)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, (new_w, new_h), interpolation=method)
            box = np.zeros((size, size, 4), dtype=np.uint8)
            ox, oy = (size - new_w) // 2, (size - new_h) // 2
            box[oy:oy + new_h, ox:ox + new_w] = img
            _sprite_cache[key] = box
    return _sprite_cache[key]


def blit(img, sprite, x, y):
    """Paste a BGRA sprite onto img at (x, y), respecting transparency.
    Skips sprites that would stick out of the image."""
    h, w = sprite.shape[:2]
    if x < 0 or y < 0:
        return
    roi = img[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):
        return
    a = sprite[:, :, 3:4] / 255.0
    roi[:] = (sprite[:, :, :3] * a + roi * (1 - a)).astype("uint8")


def put_text(frame, text, origin, scale=0.6, colour=WHITE, thickness=1):
    """Text with a black shadow, so it stays readable on any background."""
    x, y = origin
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_item(frame, item, x, y, size):
    """Draw an Ingredient (as its picture) or a Pizza (as a coloured circle)."""
    x, y = max(0, int(x)), max(0, int(y))
    if isinstance(item, Ingredient):
        sprite = load_sprite(item.image_name, size)
        if sprite is not None:
            blit(frame, sprite, x, y)
    elif isinstance(item, Pizza):
        centre = (x + size // 2, y + size // 2)
        radius = size // 2 - 4
        cv2.circle(frame, centre, radius, PIZZA_COLOURS.get(item.stage, (170, 220, 245)), -1)
        cv2.circle(frame, centre, radius, (40, 40, 40), 2)
        put_text(frame, item.stage, (x + 4, y + size // 2 + 5), 0.45)


def draw_zones(frame):
    for name, (x1, y1, x2, y2) in ZONE_RECTS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLOUR, 2)
        put_text(frame, name, (x1 + 5, y1 + 18), 0.45)
        if name in ZONE_HINTS:
            put_text(frame, ZONE_HINTS[name], (x1 + 5, y1 + 42), 0.5, (200, 255, 200))


def draw_timer(frame, state):
    x1, y1, x2, y2 = DISPLAY_RECTS[TIMER_DISPLAY]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    put_text(frame, f"TIME {state.time_left:4.1f}s", (x1 + 10, y1 + 42), 1.0,
             MOOD_COLOURS[state.mood_level], 2)
    put_text(frame, f"SCORE {state.score}", (x2 - 160, y1 + 42), 1.0, WHITE, 2)


def draw_order(frame, state):
    """The current recipe: one picture per topping, green border = already on the pizza."""
    x1, y1, x2, y2 = DISPLAY_RECTS[RECIPE_DISPLAY]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    put_text(frame, "ORDER", (x1 + 8, y1 + 22), 0.55)
    size = 80
    for i, topping in enumerate(state.order.toppings):
        x, y = x1 + 90 + i * (size + 15), y1 + 20
        sprite = load_sprite(f"{topping}_sliced", size)
        if sprite is not None:
            blit(frame, sprite, x, y)
        done = topping in state.pizza.toppings
        cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0) if done else (200, 200, 200), 2)


def draw_pizza_status(frame, state):
    x1, y1, _, _ = ZONE_RECTS[TOPPING_STATION]
    stage = "dough in box" if state.dough_in_box() else state.pizza.stage
    toppings = ", ".join(state.pizza.toppings) or "none"
    put_text(frame, f"PIZZA: {stage}", (x1 + 5, y1 + 68), 0.5)
    put_text(frame, f"toppings: {toppings}", (x1 + 5, y1 + 92), 0.5)

    # Dough box: a ball of dough while the dough for this order is still there
    bx1, by1, bx2, by2 = ZONE_RECTS[BOX_DOUGH]
    if state.dough_in_box():
        centre = ((bx1 + bx2) // 2, (by1 + by2) // 2 + 10)
        cv2.circle(frame, centre, 32, PIZZA_COLOURS["dough"], -1)
        cv2.circle(frame, centre, 32, (40, 40, 40), 2)


def draw_oven(frame, state):
    """Bar that fills up while baking: orange = baking, green = take it out NOW."""
    x1, y1, x2, y2 = ZONE_RECTS[OVEN]
    if state.oven.pizza is None:
        return
    elapsed = state.oven.elapsed
    fraction = min(elapsed / BURN_TIME, 1.0)
    bar_top = int(y2 - (y2 - y1) * fraction)       # the bar grows upward
    colour = (0, 165, 255) if elapsed < BAKE_TIME else (0, 220, 0)
    cv2.rectangle(frame, (x1 + 10, bar_top), (x1 + 40, y2), colour, -1)
    bake_line = int(y2 - (y2 - y1) * BAKE_TIME / BURN_TIME)
    cv2.line(frame, (x1 + 5, bake_line), (x1 + 45, bake_line), WHITE, 2)
    put_text(frame, "BAKING" if elapsed < BAKE_TIME else "TAKE OUT!", (x1 + 2, y1 - 8), 0.55, colour, 2)


def draw_bank(frame, state):
    x1, y1, _, _ = ZONE_RECTS[BANK]
    for i, item in enumerate(state.bank.items):
        draw_item(frame, item, x1 + 5, y1 + 25 + i * 72, 70)


def draw_abilities(frame, state):
    """Green box = ready, grey box = cooling down (seconds left shown)."""
    for display, (name, label) in ABILITY_DISPLAYS.items():
        x1, y1, x2, y2 = DISPLAY_RECTS[display]
        cooldown = state.cooldowns[name]
        ready = cooldown <= 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0) if ready else (70, 70, 70), -1)
        put_text(frame, label, (x1 + 4, y1 + 22), 0.42)
        put_text(frame, "READY" if ready else f"{cooldown:.1f}s", (x1 + 4, y1 + 50), 0.6)


def draw_player(frame, player, state):
    """Keypoints (green = reliable, red = unsure) and what the player holds, above the head."""
    if not player["visible"]:
        return
    for keypoint in player["keypoints"].values():
        x, y, _ = keypoint
        cv2.circle(frame, (x, y), 6, (0, 255, 0) if is_reliable(keypoint) else (0, 0, 255), -1)

    holding = state.players[player["player_id"]].holding
    nose = player["keypoints"].get("nose")
    if holding is not None and is_reliable(nose):
        x, y, _ = nose
        draw_item(frame, holding, x - 35, y - 150, 70)


def draw_hold_bar(frame, x, y, progress):
    """Grey bar above a wrist that fills green during a 1 s hold."""
    bar_x, bar_y = x - 40, y - 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 80, bar_y + 10), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(80 * progress), bar_y + 10), (0, 255, 0), -1)


def draw_active_effects(frame, state):
    if state.knife_frenzy_left > 0:
        put_text(frame, f"KNIFE FRENZY {state.knife_frenzy_left:.1f}s", (140, 165), 0.7, (0, 255, 255), 2)
    if state.hot_oven_left > 0:
        put_text(frame, f"HOT OVEN {state.hot_oven_left:.1f}s", (900, 195), 0.7, (0, 140, 255), 2)


def draw_message(frame, state):
    """The last rule decision: green = OK, red = REFUSED (with the reason)."""
    if state.last_message:
        colour = (0, 255, 0) if state.last_message.startswith("OK") else (80, 80, 255)
        put_text(frame, state.last_message, (140, 560), 0.6, colour, 2)


def draw_game_over(frame, state):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)   # darken the picture
    put_text(frame, "GAME OVER", (430, 320), 2.0, (0, 0, 255), 4)
    put_text(frame, f"Pizzas served: {state.score}", (480, 390), 1.0, WHITE, 2)
    put_text(frame, "press R to play again, Q to quit", (420, 440), 0.8)


def draw_everything(frame, players, state, hold_bars):
    draw_zones(frame)
    draw_timer(frame, state)
    draw_order(frame, state)
    draw_pizza_status(frame, state)
    draw_oven(frame, state)
    draw_bank(frame, state)
    draw_abilities(frame, state)
    for player in players:
        draw_player(frame, player, state)
    for x, y, progress in hold_bars:
        draw_hold_bar(frame, x, y, progress)
    draw_active_effects(frame, state)
    draw_message(frame, state)
    if state.game_over:
        draw_game_over(frame, state)


# =====================================================================
# 11. SELF TEST   (python main.py --selftest, no camera needed)
# Plays 4 pizzas through the SAME path the camera uses:
# Command -> to_rule_command -> GameState.apply
# =====================================================================
def selftest():
    # Zones and dwell timer
    assert zone_at(200, 50) == BOX_CHEESE
    assert zone_at(640, 300) == BANK
    assert zone_at(640, 50) is None
    timer = DwellTimer()
    assert timer.update(BOX_CHEESE, 0.0) is None
    assert timer.update(BOX_CHEESE, 0.5) is None
    assert timer.update(BOX_CHEESE, 1.0) == BOX_CHEESE
    assert timer.update(BOX_CHEESE, 2.0) is None        # still there: no second grab
    timer.update(None, 2.1)                             # hand leaves...
    timer.update(BOX_CHEESE, 2.2)                       # ...and comes back
    assert timer.update(BOX_CHEESE, 3.2) == BOX_CHEESE
    print("Zones + DwellTimer OK")

    # The bridge
    rule = to_rule_command(Command(player_id=SOUS_CHEF_ID, role=SOUS_CHEF, action=GRAB, item=CHEESE))
    assert rule.action == "grab_raw" and rule.arg == CHEESE
    rule = to_rule_command(Command(player_id=CHEF_ID, role=CHEF, action=PASS_TO_BANK, item=PIZZA))
    assert rule.action == "place_on_bank"
    rule = to_rule_command(Command(player_id=CHEF_ID, role=CHEF, action=ABILITY_2))
    assert rule.action == "ability" and rule.arg == "stubborn"
    print("Bridge (Command -> RuleCommand) OK")

    # Zone -> command depends on the situation
    state = GameState(seed=1)
    sous_player = {"player_id": SOUS_CHEF_ID, "role": SOUS_CHEF}
    chef_player = {"player_id": CHEF_ID, "role": CHEF}
    assert zone_to_command(sous_player, BANK, state, 0.0).action == TAKE_FROM_BANK
    state.sous.holding = Ingredient(CHEESE, state="chopped")
    assert zone_to_command(sous_player, BANK, state, 0.0).action == PASS_TO_BANK
    assert zone_to_command(chef_player, BOX_CHEESE, state, 0.0) is None   # Chef can't use boxes
    assert zone_to_command(chef_player, OVEN, state, 0.0).action == OVEN_IN
    assert zone_to_command(chef_player, BOX_DOUGH, state, 0.0).item == DOUGH
    assert zone_to_command(sous_player, BOX_DOUGH, state, 0.0) is None    # dough is Chef-only
    print("Zone -> command OK")

    # Dough box: the Chef must grab the dough before spinning or placing it
    state = GameState(seed=1)

    def chef_try(action, item=None):
        return state.apply(to_rule_command(
            Command(player_id=CHEF_ID, role=CHEF, action=action, item=item, timestamp=0.0)))

    assert state.dough_in_box()
    assert not chef_try(SPIN_DONE)                      # no dough in hands yet
    assert not chef_try(PLACE, DOUGH)
    assert chef_try(GRAB, DOUGH) and not state.dough_in_box()
    assert not chef_try(GRAB, DOUGH)                    # only one dough per order
    assert chef_try(SPIN_DONE) and chef_try(PLACE, DOUGH)
    assert state.chef.holding is None and state.pizza.stage == "sauced"
    print("Dough box OK")

    # Full game: 3 pizzas delivered, 1 burnt, every ability used once
    state = GameState(seed=1)
    clock = [0.0]          # a list so the inner functions can change it
    log = []

    def do(player_id, role, action, item=None):
        clock[0] += 0.2
        command = Command(player_id=player_id, role=role, action=action,
                          item=item, timestamp=clock[0])
        ok = state.apply(to_rule_command(command))
        assert ok, f"{action} refused: {state.last_message}"

    def sous(action, item=None):
        do(SOUS_CHEF_ID, SOUS_CHEF, action, item)

    def chef(action, item=None):
        do(CHEF_ID, CHEF, action, item)

    def wait(seconds):
        clock[0] += seconds
        log.extend(state.update(clock[0]))

    def build_pizza():
        for topping in list(state.order.toppings):
            sous(GRAB, topping)
            while state.sous.holding.state != "chopped":
                sous(CHOP_STRIKE)
            sous(PASS_TO_BANK, topping)
        chef(GRAB, DOUGH)
        chef(SPIN_DONE)
        chef(PLACE, DOUGH)
        for _ in list(state.order.toppings):
            chef(TAKE_FROM_BANK)
            chef(SPRINKLE_DONE)
        assert state.pizza.stage == "topped"

    def bake_and_deliver():
        chef(OVEN_IN); wait(6)
        chef(OVEN_OUT); chef(PASS_TO_BANK, PIZZA)
        sous(TAKE_FROM_BANK, PIZZA); sous(DELIVER, PIZZA)

    state.update(0.0)
    sous(ABILITY_1)                                     # Pizza 1: Knife Frenzy
    build_pizza(); bake_and_deliver()
    build_pizza(); chef(ABILITY_1)                      # Pizza 2: Hot Oven...
    chef(OVEN_IN); wait(3); sous(ABILITY_2)             # ...then Rush Delivery
    build_pizza(); chef(OVEN_IN); wait(11)              # Pizza 3: burns
    chef(ABILITY_2)                                     # Pizza 4: Stubborn, new order
    build_pizza(); bake_and_deliver()

    log.extend(state.update(clock[0]))
    print(state)
    assert state.score == 3, state.score
    assert "pizza_burnt" in log
    print("\nSELFTEST PASSED: 3 delivered, 1 burnt, all 4 abilities used.")


# =====================================================================
# 12. MAIN LOOP
# =====================================================================
def apply_command(state, command):
    """Send one camera Command through the bridge to the rules and print the result."""
    rule = to_rule_command(command)
    if rule is None:
        print(f"No rule for {command.role} {command.action}")
        return
    state.apply(rule)
    print(f"P{command.player_id} {command.action} {command.item or ''} -> {state.last_message}")


def run_game(camera_index):
    # DirectShow often fixes a black picture on Windows; elsewhere use the default.
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, api)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_index}. Try: python main.py --camera 1")

    made = ensure_placeholder_assets()
    if made:
        print(f"Created {made} placeholder pictures in {ASSET_DIR}")
    report_assets()

    tracker = FakeInput()

    def new_game(now):
        """Fresh game state, dwell timers and 'last seen' times."""
        state = GameState()
        state.update(now)
        # One DwellTimer per hand per player, keys like (2, "left_wrist")
        timers = {(pid, wrist): DwellTimer() for pid in (CHEF_ID, SOUS_CHEF_ID) for wrist in WRISTS}
        last_seen = {CHEF_ID: now, SOUS_CHEF_ID: now}
        return state, timers, last_seen

    state, timers, last_seen = new_game(time.monotonic())
    print("Game started. Hold a hand in a zone for 1 s to use it.")
    print("Keys: F chop | U spin | O sprinkle | Z/X Sous abilities | N/M Chef abilities | R restart | Q quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No picture from the camera. Is another app (Teams, Zoom...) using it?")
            break

        now = time.monotonic()
        frame = prepare_frame(frame)          # mirror + resize, always first
        players = tracker.read(frame)

        # --- 1. Turn hands into commands -------------------------------------
        commands = []
        hold_bars = []
        for player in players:
            pid = player["player_id"]

            # A player only counts as "lost" after LOST_AFTER_SECONDS without
            # being seen, so a single missed frame does not pause their actions.
            if player["visible"]:
                last_seen[pid] = now
            present = now - last_seen[pid] < LOST_AFTER_SECONDS
            if state.players[pid].visible != present:
                state.set_visible(pid, present)

            for wrist in WRISTS:
                keypoint = player["keypoints"].get(wrist)
                zone = None
                if player["visible"] and is_reliable(keypoint):
                    x, y, _ = keypoint
                    zone = zone_at(x, y)

                timer = timers[(pid, wrist)]
                fired = timer.update(zone, now)
                if zone is not None:
                    hold_bars.append((x, y, timer.progress(now)))
                if fired:
                    command = zone_to_command(player, fired, state, now)
                    if command is not None:
                        commands.append(command)

        # --- 2. Apply them to the game rules ---------------------------------
        # Both players at the bank in the same frame: the rules let the first one win.
        for command in commands:
            apply_command(state, command)
        for event in state.update(now):       # baked, burnt, game_over, ...
            print("EVENT:", event)

        # --- 3. Visual feedback ----------------------------------------------
        draw_everything(frame, players, state, hold_bars)
        cv2.imshow(WINDOW_NAME, frame)

        # --- 4. Keyboard -----------------------------------------------------
        key_code = cv2.waitKey(1) & 0xFF
        if key_code == 255:                   # no key pressed
            continue
        key = chr(key_code).lower()
        if key == "q":
            break
        if key == "r":
            state, timers, last_seen = new_game(time.monotonic())
            print("New game!")
            continue
        command = key_to_command(key, time.monotonic())
        if command is not None:
            apply_command(state, command)

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pizza Kitchen (Task 4)")
    parser.add_argument("--selftest", action="store_true",
                        help="check the game rules without a camera")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="camera number (0 = first camera)")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    else:
        run_game(args.camera)
