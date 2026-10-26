"""Detectors: motions and poses that become commands.

Not done yet: chopping, dough spinning, sprinkling and the two ability poses.
Until those detectors exist, a key press makes the SAME Command a detector
will make later. When a detector is written, it goes in this file.
"""
from commands import (CHEF, SOUS_CHEF, CHEF_ID, SOUS_CHEF_ID, CHOP_STRIKE, SPIN_DONE,
                      SPRINKLE_DONE, ABILITY_1, ABILITY_2, Command)


# =====================================================================
# KEYBOARD STAND-INS (until the motion / pose detectors exist)
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
    # Look up who does what for this key, then build the same Command a detector would.
    player_id, role, action = KEYBOARD_ACTIONS[key]
    return Command(player_id=player_id, role=role, action=action, timestamp=now)
