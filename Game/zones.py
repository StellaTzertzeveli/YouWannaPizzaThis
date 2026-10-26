"""Zones: where on the screen a player can do something, and the timers
that decide when "a hand in a zone" counts as an action.

Also: prepare_frame() (mirror + resize every webcam frame first).
Try it:  python zones.py
"""
import cv2           # OpenCV: flipping and resizing frames

from commands import CHEESE, PEPPERONI, MUSHROOMS, PEPPERS


# =====================================================================
# ZONES AND TIMERS
#
# A zone is a rectangle on the screen where a player can do something,
# e.g. hold a hand in "box_cheese" for 1 s to grab cheese.
# =====================================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# The webcam image is mirrored, so the left of the
# screen is the left of the players. The Sous Chef stands left.
MIRROR_WEBCAM = True


def prepare_frame(frame):
    """Mirror and resize a webcam frame so it matches the zones.

    Call this FIRST, right after cap.read(). Then keypoints, zones and
    drawings all use the same picture.
    """
    # Step 1: mirror, so players see themselves like in a mirror.
    if MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)   # 1 = flip around the vertical axis (left <-> right)
    # Step 2: resize to 1280 x 720, the size all zone positions are made for.
    # Webcams differ (640x480, 1920x1080...), the zones should not.
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

ZONE_RECTS = {
    BOX_CHEESE:       (140,   0,  245, 130),
    BOX_PEPPERONI:    (255,   0,  360, 130),
    BOX_MUSHROOMS:    (370,   0,  475, 130),
    BOX_PEPPERS:      (485,   0,  590, 130),
    DELIVERY_STAND:   (0,   200,  130, 520),
    CHOPPING_COUNTER: (140, 580,  590, 720),
    BANK:             (600, 200,  680, 520),
    BOX_DOUGH:        (690, 580,  790, 720), 
    TOPPING_STATION:  (800, 580, 1140, 720),
    OVEN:             (1150, 200, 1280, 520),
}

# Display areas: the game shows information. A hand here does nothing,
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
        # "Unpacking": the 4 numbers of the rect go into 4 variables at once.
        self.x1, self.y1, self.x2, self.y2 = rect

    def contains(self, point):
        x, y = point
        # Inside = between the left and right edge AND between the top and bottom edge.
        # "<" (not "<=") on the right/bottom edge, so a point on the border
        # between two touching zones belongs to only one of them.
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


# Build one Zone object per entry in ZONE_RECTS. This is a "list comprehension":
# a short for-loop that collects the results in a list. Same as:
#     ZONES = []
#     for name, rect in ZONE_RECTS.items():
#         ZONES.append(Zone(name, rect))
ZONES = [Zone(name, rect) for name, rect in ZONE_RECTS.items()]


def zone_at(x, y):
    """Name of the zone that contains point (x, y), or None.

    Example: zone_at(200, 50) -> "box_cheese"
    """
    # Check the zones one by one; the first one that contains the point wins.
    # (Zones do not overlap, so there is at most one.)
    for zone in ZONES:
        if zone.contains((x, y)):
            return zone.name
    return None                     # the point is not in any zone


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
        self.hold_seconds = hold_seconds   # the default hold time
        self.current_hold = hold_seconds   # the hold time used right now
        self.zone = None          # the zone the hand is in right now
        self.start_time = None    # when the hand entered that zone
        self.fired = False        # already fired for this visit?

    def update(self, zone, now, hold_seconds=None):
        """zone: zone name or None. hold_seconds: optional, to need a longer
        (or shorter) hold this time, e.g. 2 s for sprinkling.
        Returns the zone name on the one frame the hold completes, otherwise None."""
        # Which hold time counts this frame: the one passed in, or the default.
        # ("A if condition else B" is a one-line if/else.)
        self.current_hold = hold_seconds if hold_seconds is not None else self.hold_seconds

        # Case 1: the hand is in a DIFFERENT zone than last frame (or just left one).
        # Start counting again from now.
        if zone != self.zone:
            self.zone = zone
            self.start_time = now
            self.fired = False
            return None

        # Case 2: the hand is in no zone, or this visit already fired once.
        # Nothing to do (this is what stops "grab, grab, grab" every frame).
        if zone is None or self.fired:
            return None

        # Case 3: same zone as before. Has the hand been here long enough?
        if now - self.start_time >= self.current_hold:
            self.fired = True                # remember: do not fire again this visit
            return zone
        return None                          # not long enough yet

    def progress(self, now):
        """0.0 (just entered) to 1.0 (done). Used to draw the hold bar."""
        if self.zone is None:
            return 0.0
        if self.fired:
            return 1.0
        # Time spent so far divided by the time needed, e.g. 0.5 s / 1.0 s = 0.5.
        # min(..., 1.0) makes sure it never goes above "full".
        return min((now - self.start_time) / self.current_hold, 1.0)


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
        # Ready if it never happened yet, or enough seconds passed since last time.
        return self.last_time is None or now - self.last_time >= self.seconds

    def trigger(self, now):
        # Call this when the action actually happens: starts the waiting time.
        self.last_time = now

    def remaining(self, now):
        # How many seconds until ready again (for drawing a countdown).
        if self.ready(now):
            return 0.0
        return self.seconds - (now - self.last_time)


# Small demo: runs only when you start this file directly (python zones.py)
if __name__ == "__main__":
    print(zone_at(200, 50))     # box_cheese
    print(zone_at(640, 300))    # bank
    print(zone_at(640, 50))     # None: no zone there
