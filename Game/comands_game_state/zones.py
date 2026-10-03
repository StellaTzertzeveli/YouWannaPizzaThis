"""Zone names and positions for Task 4 (Player Interaction and Game Control).

A zone is a rectangle on the screen where a player can do something,
e.g. hold a hand in "box_cheese" for 1 s to grab cheese.

Phase 0: the NAMES are frozen (both tracks and Task 5 use them).
The POSITIONS are a first draft and can be tuned later, but always here,
in this one dictionary, so everyone draws and checks the same rectangles.
"""

# ---------------------------------------------------------------------------
# Screen size the positions are made for.
# If the webcam gives another size, resize each frame to this first:
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
# Positions are in the MIRRORED image (what the players see, like a mirror).
# ---------------------------------------------------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Phase 0 decision: the webcam image is flipped left-right, like a mirror,
# so the left of the screen is the left of the players. Sous Chef stands left.
MIRROR_WEBCAM = True


def prepare_frame(frame):
    """Flip (mirror) and resize a webcam frame so it matches the zones.

    Call this FIRST, right after cap.read(), before MediaPipe and before
    drawing. Then keypoints, zones and drawings all use the same picture.
    """
    import cv2
    if MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)   # 1 = flip around the vertical axis (left <-> right)
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))


# ---------------------------------------------------------------------------
# Zone names (frozen)
# ---------------------------------------------------------------------------
# Sous Chef side (left half)
BOX_CHEESE = "box_cheese"
BOX_PEPPERONI = "box_pepperoni"
BOX_MUSHROOMS = "box_mushrooms"
BOX_PEPPERS = "box_peppers"
CHOPPING_COUNTER = "chopping_counter"
DELIVERY_STAND = "delivery_stand"

# Middle, shared by both players
BANK = "bank"

# Chef side (right half)
TOPPING_STATION = "topping_station"
OVEN = "oven"

ALL_ZONES = (
    BOX_CHEESE, BOX_PEPPERONI, BOX_MUSHROOMS, BOX_PEPPERS,
    CHOPPING_COUNTER, DELIVERY_STAND, BANK, TOPPING_STATION, OVEN,
)


# ---------------------------------------------------------------------------
# Zone positions (first draft, in pixels for 1280 x 720)
#
# Each zone is (x1, y1, x2, y2):
#   (x1, y1) = top-left corner, (x2, y2) = bottom-right corner.
#   x grows to the RIGHT, y grows DOWNWARD (y = 0 is the top of the screen).
#
#   +------------------------------+------+------------------------------+
#   | cheese  pepperoni mush peppr |      |   TIMER  (display only)      |
#   |                              |      |   RECIPE (display only)      |
#   |DELIV|                        | BANK |                       |OVEN  |
#   |STAND|       SOUS CHEF        |      |        CHEF           |      |
#   |     |                        |      |                       |      |
#   |ABIL1|  CHOPPING COUNTER      |      |   TOPPING STATION     |ABIL1 |
#   |ABIL2|                        |      |                       |ABIL2 |
#   +------------------------------+------+------------------------------+
#   0                             600    680                          1280
# ---------------------------------------------------------------------------
ZONE_RECTS = {
    # Ingredient boxes: a row along the top of the left half
    BOX_CHEESE:       (140,   0,  245, 130),
    BOX_PEPPERONI:    (255,   0,  360, 130),
    BOX_MUSHROOMS:    (370,   0,  475, 130),
    BOX_PEPPERS:      (485,   0,  590, 130),

    # Left edge of the screen, middle height
    DELIVERY_STAND:   (0,   200,  130, 520),

    # Bottom of the left half
    CHOPPING_COUNTER: (140, 580,  590, 720),

    # The dividing strip in the middle
    BANK:             (600, 200,  680, 520),

    # Bottom of the right half
    TOPPING_STATION:  (690, 580, 1140, 720),

    # Right edge of the screen, middle height
    OVEN:             (1150, 200, 1280, 520),
}

# ---------------------------------------------------------------------------
# Display areas: places where the game SHOWS information.
# Kept separate from ZONE_RECTS on purpose: a hand in these areas does nothing,
# so zone_at() never returns them.
# ---------------------------------------------------------------------------
TIMER_DISPLAY = "timer_display"
RECIPE_DISPLAY = "recipe_display"

# Ability icons with their cooldown. Abilities are triggered by POSES
# (arms X, double thumbs up), not by touching these, so they are display only.
SOUS_ABILITY_1_DISPLAY = "sous_ability_1_display"   # Knife Frenzy
SOUS_ABILITY_2_DISPLAY = "sous_ability_2_display"   # Rush Delivery
CHEF_ABILITY_1_DISPLAY = "chef_ability_1_display"   # Hot Oven
CHEF_ABILITY_2_DISPLAY = "chef_ability_2_display"   # Stubborn

DISPLAY_RECTS = {
    # Top of the right half: the timer, with the current recipe right under it
    TIMER_DISPLAY:  (690,  0, 1140,  60),
    RECIPE_DISPLAY: (690, 65, 1140, 175),

    # Bottom-left corner (left of the chopping counter): Sous Chef abilities
    SOUS_ABILITY_1_DISPLAY: (0, 580, 130, 645),
    SOUS_ABILITY_2_DISPLAY: (0, 655, 130, 720),

    # Bottom-right corner (right of the topping station): Chef abilities
    CHEF_ABILITY_1_DISPLAY: (1150, 580, 1280, 645),
    CHEF_ABILITY_2_DISPLAY: (1150, 655, 1280, 720),
}

# Which ingredient each box gives (used when a GRAB command is made)
BOX_INGREDIENT = {
    BOX_CHEESE: "cheese",
    BOX_PEPPERONI: "pepperoni",
    BOX_MUSHROOMS: "mushrooms",
    BOX_PEPPERS: "peppers",
}


def zone_at(x, y):
    """Return the name of the zone that contains point (x, y), or None.

    Example: zone_at(200, 50) -> "box_cheese"
    """
    for name, (x1, y1, x2, y2) in ZONE_RECTS.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return name
    return None


# ---------------------------------------------------------------------------
# Demo: draws all zones on an empty frame and saves it as zones_preview.png
#   python zones.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    import numpy as np

    # A black image: height x width x 3 colour channels, all zeros
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    for name, (x1, y1, x2, y2) in ZONE_RECTS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, name, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Display areas in a different colour, so you can tell them apart
    for name, (x1, y1, x2, y2) in DISPLAY_RECTS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        label = name.replace("_display", "")   # shorter, so it fits the small boxes
        cv2.putText(frame, label, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite("zones_preview.png", frame)
    print("Saved zones_preview.png")

    # Try the helper with a few points
    print(zone_at(200, 50))     # box_cheese
    print(zone_at(640, 300))    # bank
    print(zone_at(640, 50))     # None (no zone there)
