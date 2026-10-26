"""Player interaction: turns what players DO into game commands.

    hand held in a zone  ->  zone_to_command()  ->  Command (camera format)
    Command              ->  to_rule_command()  ->  RuleCommand (what GameState understands)

Also the helpers for the agreed keypoint format (is_reliable, WRISTS).
"""
from commands import (CHEF, SOUS_CHEF, Command, GRAB, PLACE, CHOP_STRIKE, SPIN_DONE,
                      SPRINKLE_DONE, OVEN_IN, OVEN_OUT, PASS_TO_BANK, TAKE_FROM_BANK,
                      DELIVER, ABILITY_1, ABILITY_2, DOUGH, PIZZA)
from zones import (BOX_INGREDIENT, BANK, DELIVERY_STAND, BOX_DOUGH, TOPPING_STATION,
                   OVEN, HOLD_SECONDS)
from game_objects import Ingredient, Pizza
from game_state import RuleCommand


# ---------------------------------------------------------------------
# KEYPOINT HELPERS (for the agreed keypoint format, see fake_input.py)
# ---------------------------------------------------------------------
WRISTS = ("left_wrist", "right_wrist")
CONFIDENCE_THRESHOLD = 0.5    # below this, a keypoint is not trusted


def is_reliable(keypoint):
    """True if a keypoint exists and MediaPipe is confident enough about it."""
    # A keypoint is a tuple (x, y, confidence), so keypoint[2] is the confidence.
    # Checking "is not None" FIRST matters: Python stops at the first False,
    # so keypoint[2] is never read from a missing keypoint (that would crash).
    return keypoint is not None and keypoint[2] >= CONFIDENCE_THRESHOLD


# =====================================================================
# BRIDGE: camera Command -> RuleCommand
#
# The camera side (commands.py) and the rules (game_state.py) were written
# separately and use different names, e.g. GRAB vs "grab_raw".
# Some camera actions mean different rule actions depending on the role:
# PASS_TO_BANK by the Chef = "place_on_bank" (the baked pizza),
# PASS_TO_BANK by the Sous Chef = "pass_to_bank" (a chopped ingredient).
# =====================================================================
# Translation table. The KEY is a pair (role, camera action), because the same
# camera action can mean a different rule action per role (see GRAB and PASS_TO_BANK).
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
    # Build the lookup key, then check the two tables.
    key = (command.role, command.action)
    # Abilities: all become the rule action "ability", with the ability NAME as argument.
    if key in ABILITY_NAMES:
        return RuleCommand(command.player_id, "ability", ABILITY_NAMES[key], command.timestamp)
    # Normal actions: swap the name, keep player, item and time the same.
    if key in RULE_ACTIONS:
        return RuleCommand(command.player_id, RULE_ACTIONS[key], command.item, command.timestamp)
    return None                     # no rule for this role + action (e.g. a Chef CHOP_STRIKE)


# =====================================================================
# ZONES -> COMMANDS (what a 1 s hold in a zone means)
# Some zones mean different things depending on what is going on:
# a hand in the bank PUTS something down if you hold something, and
# TAKES something if your hands are empty.
# =====================================================================
def zone_to_command(player, zone, state, now):
    """Return the Command for a completed hold in a zone, or None if that
    zone means nothing for this player's role."""
    player_id, role = player["player_id"], player["role"]

    # A small helper function INSIDE this function. It can use player_id, role
    # and now from the outer function, so each case below only says what differs:
    # make(GRAB, CHEESE) instead of the full Command(player_id=..., role=..., ...).
    def make(action, item=None):
        return Command(player_id=player_id, role=role, action=action,
                       item=item, timestamp=now)

    # Sous Chef zones (left side). Each zone checks only what this role can do there.
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

    # Chef zones (right side). Several zones look at what the Chef is holding,
    # because the same zone means different things at different moments.
    if role == CHEF:
        if zone == BOX_DOUGH:
            return make(GRAB, DOUGH)
        if zone == TOPPING_STATION:
            holding = state.chef.holding
            if holding is state.pizza:                 # holding the dough: put it down
                return make(PLACE, DOUGH)
            if isinstance(holding, Ingredient):        # holding a topping: put it on
                # Stand-in for the sprinkle detector: a 2 s hover (see hold_seconds_for)
                return make(SPRINKLE_DONE, holding.type)
            return None
        if zone == OVEN:
            if state.oven.pizza is None:
                return make(OVEN_IN, PIZZA)
            return make(OVEN_OUT, PIZZA)
        if zone == BANK:
            holding = state.chef.holding
            if isinstance(holding, Pizza):             # baked pizza for the Sous Chef
                return make(PASS_TO_BANK, PIZZA)
            if isinstance(holding, Ingredient):        # put a topping back
                return make(PASS_TO_BANK, holding.type)
            return make(TAKE_FROM_BANK)                # empty hands: take a topping

    # Any other zone/role combination does nothing (e.g. the Chef in a Sous Chef box).
    return None


SPRINKLE_SECONDS = 2.0      # hover time to add a topping (concept: "sprinkle for 2 s")


def hold_seconds_for(player, zone, state):
    """How long a hand must stay in this zone before it counts.
    Normally 1 s; 2 s when the Chef hovers a topping over the pizza."""
    if (player["role"] == CHEF and zone == TOPPING_STATION
            and isinstance(state.chef.holding, Ingredient)):
        return SPRINKLE_SECONDS
    return HOLD_SECONDS
