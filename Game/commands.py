"""Commands: the agreed interface between "seeing" and "rules".

A command is the message the seeing part (zones, detectors) sends to the
rules part (GameState):
    "player 2 held a hand in the cheese box for 1 s"
    -> Command(player_id=2, role=SOUS_CHEF, action=GRAB, item=CHEESE)

Shared with the whole team: change names here only if everyone agrees.
Try it:  python commands.py
"""
import time          # time.monotonic(): when a command was made
from dataclasses import dataclass, field   # short way to write classes that mostly hold data


# Player IDs: Player 1 = Chef (right side), Player 2 = Sous Chef (left side)
CHEF_ID = 1
SOUS_CHEF_ID = 2


# =====================================================================
# COMMANDS 
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

# All valid action names together, so a Command can check it got a real one.
# Round brackets = a tuple: a list that cannot be changed later.
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
ALL_ITEMS = TOPPINGS + (DOUGH, PIZZA)     # + on tuples glues them together


# @dataclass writes the boring parts of the class for us: __init__ (so you can
# write Command(player_id=2, role=..., ...)), printing, and comparing.
# frozen=True: once made, a Command cannot be changed. A message that was sent
# should not be edited afterwards.
@dataclass(frozen=True)
class Command:
    """One game action by one player (the camera side's format).

    player_id:  the stable ID from Task 3 (1 or 2)
    role:       CHEF or SOUS_CHEF
    action:     one of ALL_ACTIONS
    item:       one of ALL_ITEMS, or None if the action has no object
    timestamp:  when the action was recognised (seconds, time.monotonic())
    """
    # The fields (name: type). Fields with "= ..." are optional when creating one.
    player_id: int
    role: str
    action: str
    item: str | None = None          # "str | None" = a text, or nothing
    # default_factory: if no timestamp is given, call time.monotonic() at the
    # moment the Command is CREATED (so each Command gets its own time).
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # Runs right after a Command is created: catches typos like "GRABB" at once.
        if self.role not in ALL_ROLES:
            raise ValueError(f"Unknown role: {self.role!r}")
        if self.action not in ALL_ACTIONS:
            raise ValueError(f"Unknown action: {self.action!r}")
        if self.item is not None and self.item not in ALL_ITEMS:
            raise ValueError(f"Unknown item: {self.item!r}")


# Small demo: runs only when you start this file directly (python commands.py)
if __name__ == "__main__":
    print(Command(player_id=SOUS_CHEF_ID, role=SOUS_CHEF, action=GRAB, item=CHEESE))
    try:
        Command(player_id=CHEF_ID, role=CHEF, action="GRABB")     # typo on purpose
    except ValueError as error:
        print("Caught a typo:", error)
