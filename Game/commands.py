"""Command format for Task 4 (Player Interaction and Game Control).

A command is the "message" that the seeing part (Track A: zones, detectors,
players) sends to the rules part (Track B: GameState).

    Track A: "player 2 held a hand in the cheese box for 1 s"
        -> Command(player_id=2, role=SOUS_CHEF, action=GRAB, item=CHEESE)
    Track B: GameState.apply(command) decides if that is allowed right now.

This file is shared and FROZEN after Phase 0: change it only if both
tracks agree, because both sides depend on these exact names.
"""
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Roles: which job a player has. Left side = Sous Chef, right side = Chef.
# ---------------------------------------------------------------------------
CHEF = "CHEF"
SOUS_CHEF = "SOUS_CHEF"

ALL_ROLES = (CHEF, SOUS_CHEF)


# ---------------------------------------------------------------------------
# Action names: every game action a player can trigger.
# ---------------------------------------------------------------------------
GRAB = "GRAB"                       # take a raw ingredient from a box (1 s hold)
PLACE = "PLACE"                     # put something down, e.g. dough on the topping station
CHOP_STRIKE = "CHOP_STRIKE"         # ONE downward strike at the chopping counter
SPIN_DONE = "SPIN_DONE"             # one full circle of dough spinning finished
SPRINKLE_DONE = "SPRINKLE_DONE"     # 2 s of sprinkling over the pizza finished (one topping)
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


# ---------------------------------------------------------------------------
# Items: the optional "what" of a command (e.g. GRAB *cheese*).
# ---------------------------------------------------------------------------
CHEESE = "cheese"
PEPPERONI = "pepperoni"
MUSHROOMS = "mushrooms"
PEPPERS = "peppers"
DOUGH = "dough"
PIZZA = "pizza"

TOPPINGS = (CHEESE, PEPPERONI, MUSHROOMS, PEPPERS)
ALL_ITEMS = TOPPINGS + (DOUGH, PIZZA)


# ---------------------------------------------------------------------------
# The command itself
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Command:
    """One game action by one player.

    player_id:  the stable ID from Task 3 (e.g. 1 or 2)
    role:       CHEF or SOUS_CHEF
    action:     one of ALL_ACTIONS
    item:       one of ALL_ITEMS, or None if the action has no object
    timestamp:  when the action was recognised (seconds, filled in automatically)
    """
    player_id: int
    role: str
    action: str
    item: str | None = None
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # Runs automatically right after a Command is created.
        # Catches typos like "GRABB" immediately instead of hours later.
        if self.role not in ALL_ROLES:
            raise ValueError(f"Unknown role: {self.role!r}")
        if self.action not in ALL_ACTIONS:
            raise ValueError(f"Unknown action: {self.action!r}")
        if self.item is not None and self.item not in ALL_ITEMS:
            raise ValueError(f"Unknown item: {self.item!r}")


# ---------------------------------------------------------------------------
# Small demo: runs only when you start this file directly
#   python commands.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grab = Command(player_id=2, role=SOUS_CHEF, action=GRAB, item=CHEESE)
    oven = Command(player_id=1, role=CHEF, action=OVEN_IN)

    print(grab)
    print(oven)
    print("Who grabbed?", grab.player_id, "| What?", grab.item)

    try:
        Command(player_id=1, role=CHEF, action="GRABB")
    except ValueError as error:
        print("Caught a typo:", error)