"""Game objects: the players, Ingredient, Pizza, Order, Oven and Bank.

Only the THINGS in the kitchen live here. Who may do what (the rules)
is in game_state.py.
"""
import random        # random orders
from dataclasses import dataclass, field   # short way to write classes that mostly hold data

from commands import TOPPINGS


# =====================================================================
# GAME OBJECTS 
# Ingredient, Pizza, Order, Oven, Bank and the two player classes.
# No game rules here (who may do what): those are in game_state.py.
# =====================================================================
# Strikes to chop one ingredient. 6 matches the art in assets/ingredients:
#   0 strikes -> _raw, 1..5 strikes -> _cut_1 .. _cut_5, 6 strikes -> _sliced
CHOP_STRIKES = 6
BAKE_TIME = 5.0              # seconds in the oven until baked
BURN_TIME = 10.0             # seconds in the oven until burnt
BANK_CAPACITY = 4            # TBD: max items on the bank

PIZZA_STAGES = ["dough", "spun", "sauced", "topped", "baking", "baked", "burnt"]


# --- Players -------------------------------------------------------------
# Player is the shared "parent" class. Chef and SousChef below are "child"
# classes (inheritance): they get everything from Player and only fill in
# their own role and the list of actions they are allowed to do.
class Player:
    """GameState needs: .id .role_key .holding .visible .can(action)"""
    # Class attributes: the same for every object of the class (children override them).
    role_key = ""            # "chef" or "sous_chef"
    role_name = ""           # for messages
    allowed_actions = frozenset()

    def __init__(self, player_id):
        self.id = player_id
        self.holding = None      # one item at a time (Ingredient or Pizza)
        self.visible = True      # False = lost from the camera (actions paused)

    def can(self, action):
        # Role check: is this action on this player's list? (e.g. a Chef cannot "chop")
        return action in self.allowed_actions

    def __repr__(self):
        # How the object looks when printed, e.g. "Chef(id=1, holding=None)"
        return f"{self.role_name}(id={self.id}, holding={self.holding})"


class Chef(Player):            # "(Player)" = Chef inherits from Player
    role_key = "chef"
    role_name = "Chef"
    # frozenset = a set that cannot change: fast "is X in it?" checks, no duplicates.
    allowed_actions = frozenset({
        "grab_dough", "spin_dough", "place_at_station", "add_topping", "oven_in", "oven_out",
        "take_from_bank", "place_on_bank", "ability"})


class SousChef(Player):
    role_key = "sous_chef"
    role_name = "Sous Chef"
    allowed_actions = frozenset({
        "grab_raw", "chop", "pass_to_bank", "take_pizza_from_bank", "deliver", "ability"})


# --- Things in the kitchen -----------------------------------------------
# These are @dataclass classes too, but NOT frozen: an ingredient changes
# (more strikes, raw -> chopped) and a pizza changes (dough -> ... -> baked).
@dataclass
class Ingredient:
    type: str
    state: str = "raw"       # "raw" or "chopped"
    strikes: int = 0         # how many chop strikes it has had

    def __post_init__(self):
        if self.type not in TOPPINGS:
            raise ValueError(f"unknown ingredient: {self.type}")

    # @property: used like a variable (ingredient.image_name), but computed fresh
    # every time from the current state, so it is always up to date.
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
    # default_factory=list gives EVERY pizza its own new empty list.
    # (Writing "= []" would make all pizzas share one list: a classic Python trap.)
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
        # Safety checks right after creation.
        if not 1 <= len(self.toppings) <= 3:
            raise ValueError("an order has 1 to 3 toppings")
        # set() removes duplicates: if the length shrinks, there was a duplicate.
        if len(set(self.toppings)) != len(self.toppings):
            raise ValueError("no duplicate toppings in an order")

    # @staticmethod: belongs to the class, no "self" needed. Called as Order.random().
    @staticmethod
    def random(rng=None):
        # rng = a random number generator. "rng or random": use the one given, or
        # Python's normal one. Passing our own (with a seed) makes tests repeatable.
        rng = rng or random
        n = rng.randint(1, 3)                      # how many toppings: 1, 2 or 3
        return Order(rng.sample(TOPPINGS, n))      # sample = no duplicates

    def is_satisfied_by(self, pizza):
        """True if the pizza has exactly the toppings the order asks for."""
        # Sort both lists so the ORDER of adding toppings does not matter:
        # ["peppers", "cheese"] and ["cheese", "peppers"] count as the same pizza.
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
        if self.pizza is None:                 # empty oven: nothing to bake
            return happened
        # dt ("delta time") = seconds since the last update. Frames do not come at a
        # perfectly steady rate, so we measure the real time instead of counting frames.
        dt = max(0.0, now - self.last_now)
        self.last_now = now
        # Add the time, faster when Hot Oven is active (speed 2.0 = twice as fast).
        self.elapsed += dt * speed
        # Check the two thresholds. "baking" check: report "baked" only once.
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
        self.items = []              # what lies on the bank right now, oldest first
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
