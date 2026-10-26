"""Visual feedback: everything the game draws on the frame.

Zones, timer, order, oven bar, bank, ability cooldowns, the items players
hold, hold bars and messages. Ingredient pictures come from
assets/ingredients/ next to this file.
"""
import os            # file paths: where the ingredient pictures are
import cv2           # OpenCV: drawing and loading images
import numpy as np   # images are numpy arrays

from commands import TOPPINGS, CHEESE, PEPPERONI, MUSHROOMS, PEPPERS
from zones import (FRAME_WIDTH, FRAME_HEIGHT, ZONE_RECTS, DISPLAY_RECTS,
                   CHOPPING_COUNTER, TOPPING_STATION, BOX_DOUGH, OVEN, BANK,
                   TIMER_DISPLAY, RECIPE_DISPLAY, SOUS_ABILITY_1_DISPLAY,
                   SOUS_ABILITY_2_DISPLAY, CHEF_ABILITY_1_DISPLAY, CHEF_ABILITY_2_DISPLAY)
from game_objects import CHOP_STRIKES, BAKE_TIME, BURN_TIME, Ingredient, Pizza
from players import is_reliable


FONT = cv2.FONT_HERSHEY_SIMPLEX

# =====================================================================
# DRAWING / VISUAL FEEDBACK
# Ingredient pictures: assets/ingredients/<name>.png next to this file,
# e.g. cheese_raw.png, cheese_cut_1.png ... cheese_sliced.png.
# Missing files are created as simple placeholders on the first run.
# =====================================================================
# The picture folder, built from THIS file's location (__file__), not from the folder
# you started Python in. So it works no matter where you run "python main.py" from.
# os.path.join puts the right separator in (\ on Windows, / on Mac/Linux).
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
    TOPPING_STATION: "U = spin   hover topping 2 s (or O)",
}
ABILITY_DISPLAYS = {          # display area: (ability name, label)
    SOUS_ABILITY_1_DISPLAY: ("knife_frenzy", "Z Knife Frenzy"),
    SOUS_ABILITY_2_DISPLAY: ("rush_delivery", "X Rush Deliv."),
    CHEF_ABILITY_1_DISPLAY: ("hot_oven", "N Hot Oven"),
    CHEF_ABILITY_2_DISPLAY: ("stubborn", "M Stubborn"),
}

# Caches: remember pictures that were already loaded and resized. Reading a PNG
# from disk every frame (30 times a second) would make the game slow.
# The "_" in front of the name means "only used inside this file".
_sprite_cache = {}
_crop_cache = {}


def all_ingredient_image_names():
    # Every picture name the game can ask for, e.g. cheese_raw, cheese_cut_1 ... cheese_sliced.
    names = []
    for t in TOPPINGS:
        names.append(f"{t}_raw")
        names += [f"{t}_cut_{i}" for i in range(1, CHOP_STRIKES)]
        names.append(f"{t}_sliced")
    return names


def _make_placeholder(name):
    # "cheese_cut_2".split("_", 1) -> ["cheese", "cut_2"]  (split only at the first "_")
    ing_type, state = name.split("_", 1)
    color = INGREDIENT_COLORS.get(ing_type, (200, 200, 200))
    # A new dark-grey square image: height x width x 3 colour channels, every value 45.
    # uint8 = whole numbers 0..255, the normal type for image pixels.
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
    os.makedirs(ASSET_DIR, exist_ok=True)     # make the folder; no error if it exists
    made = 0
    for name in all_ingredient_image_names():
        path = os.path.join(ASSET_DIR, name + ".png")
        # Only create MISSING pictures, so your own art is never overwritten.
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
    # Transparency lives in a 4th colour channel called "alpha" (B, G, R, A):
    # 0 = fully see-through, 255 = fully solid. No 4th channel = no transparency.
    if img.ndim < 3 or img.shape[2] != 4:
        return None
    alpha = img[:, :, 3]                 # just the alpha channel: a 2D array
    if alpha.min() > 250:                # everything solid: nothing to trim
        return None
    # np.where gives the row (y) and column (x) of every pixel that is visible.
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    # The smallest/largest x and y form the box around the drawing.
    # Divided by the size -> fractions (0..1), so the box works for any picture size.
    h, w = alpha.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _ingredient_crop(ing_type):
    """One crop box shared by ALL pictures of an ingredient, so it keeps the
    same size from picture to picture instead of jumping."""
    if ing_type not in _crop_cache:           # compute once per ingredient, then reuse
        # Collect the visible-area box of every picture of this ingredient...
        boxes = []
        for name in all_ingredient_image_names():
            if name.startswith(ing_type + "_"):
                img = cv2.imread(os.path.join(ASSET_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
                box = _alpha_box(img) if img is not None else None
                if box:
                    boxes.append(box)
        # ...and take one box that fits around ALL of them, plus a little padding.
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
    if key not in _sprite_cache:              # not loaded at this size yet
        # Step 1: read the PNG. IMREAD_UNCHANGED keeps the alpha (transparency) channel.
        img = cv2.imread(os.path.join(ASSET_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
        if img is None:                       # file missing or unreadable
            _sprite_cache[key] = None
        else:
            # Step 2: make sure it has 4 channels (B, G, R, alpha), whatever the file was.
            if img.ndim == 2:                                   # greyscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:                             # colour, no transparency
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            h, w = img.shape[:2]

            # Step 3: cut away the empty transparent border (same box for all
            # pictures of this ingredient), so the drawing fills the space.
            crop = _ingredient_crop(name.split("_")[0]) if CROP_TO_CONTENT else None
            if crop and _alpha_box(img):
                img = img[int(crop[1] * h):int(crop[3] * h) + 1, int(crop[0] * w):int(crop[2] * w) + 1]
                h, w = img.shape[:2]

            # Step 4: scale so the LONGEST side fits the box, keeping the proportions.
            # INTER_AREA looks best when shrinking, INTER_CUBIC when enlarging.
            scale = size / max(h, w)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, (new_w, new_h), interpolation=method)

            # Step 5: paste it in the middle of a transparent size x size square,
            # so every sprite has the same size no matter its shape.
            box = np.zeros((size, size, 4), dtype=np.uint8)     # all zeros = fully transparent
            ox, oy = (size - new_w) // 2, (size - new_h) // 2   # offsets to centre it
            box[oy:oy + new_h, ox:ox + new_w] = img
            _sprite_cache[key] = box
    return _sprite_cache[key]


def blit(img, sprite, x, y):
    """Paste a BGRA sprite onto img at (x, y), respecting transparency.
    Skips sprites that would stick out of the image."""
    h, w = sprite.shape[:2]
    if x < 0 or y < 0:
        return
    # roi ("region of interest") = the part of the frame the sprite will cover.
    # It is a VIEW into the frame: changing roi changes the frame itself.
    roi = img[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):             # sprite would stick out of the frame
        return
    # Alpha blending, per pixel:  result = sprite * a + background * (1 - a)
    # a = 1.0 -> only the sprite; a = 0.0 -> only the background; 0.5 -> half/half.
    a = sprite[:, :, 3:4] / 255.0           # alpha as 0.0..1.0 ("3:4" keeps it 3D for numpy)
    roi[:] = (sprite[:, :, :3] * a + roi * (1 - a)).astype("uint8")


def put_text(frame, text, origin, scale=0.6, colour=WHITE, thickness=1):
    """Text with a black shadow, so it stays readable on any background."""
    x, y = origin
    # Draw the text twice: first in black 2 pixels lower-right (the shadow), then in
    # colour on top. LINE_AA = anti-aliased, i.e. smooth edges instead of jagged ones.
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_item(frame, item, x, y, size):
    """Draw an Ingredient (as its picture) or a Pizza (as a coloured circle).
    (x, y) is the top-left corner; it is pushed back inside the frame if needed."""
    height, width = frame.shape[:2]
    x = min(max(0, int(x)), width - size)       # keep the whole item on screen
    y = min(max(0, int(y)), height - size)
    # Ingredients have pictures; the pizza is drawn as a circle coloured by its stage.
    if isinstance(item, Ingredient):
        sprite = load_sprite(item.image_name, size)
        if sprite is not None:
            blit(frame, sprite, x, y)
    elif isinstance(item, Pizza):
        centre = (x + size // 2, y + size // 2)
        # Filled circle first (thickness -1 = fill), then a dark outline on top.
        radius = size // 2 - 4
        cv2.circle(frame, centre, radius, PIZZA_COLOURS.get(item.stage, (170, 220, 245)), -1)
        cv2.circle(frame, centre, radius, (40, 40, 40), 2)
        put_text(frame, item.stage, (x + 4, y + size // 2 + 5), 0.45)


def draw_zones(frame):
    # An outline + name for every hand zone, so players see where to reach.
    for name, (x1, y1, x2, y2) in ZONE_RECTS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLOUR, 2)
        put_text(frame, name, (x1 + 5, y1 + 18), 0.45)
        if name in ZONE_HINTS:
            put_text(frame, ZONE_HINTS[name], (x1 + 5, y1 + 42), 0.5, (200, 255, 200))


def draw_timer(frame, state):
    x1, y1, x2, y2 = DISPLAY_RECTS[TIMER_DISPLAY]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)     # dark background
    # The timer colour follows the mood: green (calm) -> orange -> red (time running out).
    # {...:4.1f} = show the number with 1 decimal, e.g. "47.9".
    put_text(frame, f"TIME {state.time_left:4.1f}s", (x1 + 10, y1 + 42), 1.0,
             MOOD_COLOURS[state.mood_level], 2)
    put_text(frame, f"SCORE {state.score}", (x2 - 160, y1 + 42), 1.0, WHITE, 2)


def draw_order(frame, state):
    """The current recipe: one picture per topping, green border = already on the pizza."""
    x1, y1, x2, y2 = DISPLAY_RECTS[RECIPE_DISPLAY]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    put_text(frame, "ORDER", (x1 + 8, y1 + 22), 0.55)
    size = 80
    # enumerate gives a counter i (0, 1, 2) with each topping, used to place the
    # pictures next to each other: every next one 95 pixels further right.
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
    # How full the bar is: 0.0 = just put in, 1.0 = burnt (10 s).
    fraction = min(elapsed / BURN_TIME, 1.0)
    # y grows DOWNWARD, so a bar that grows upward starts at the bottom (y2)
    # and its top moves up by (zone height x fraction).
    bar_top = int(y2 - (y2 - y1) * fraction)       # the bar grows upward
    colour = (0, 165, 255) if elapsed < BAKE_TIME else (0, 220, 0)
    cv2.rectangle(frame, (x1 + 10, bar_top), (x1 + 40, y2), colour, -1)
    # A white line at the "baked" mark (5 of 10 s = halfway): above it, take it out!
    bake_line = int(y2 - (y2 - y1) * BAKE_TIME / BURN_TIME)
    cv2.line(frame, (x1 + 5, bake_line), (x1 + 45, bake_line), WHITE, 2)
    put_text(frame, "BAKING" if elapsed < BAKE_TIME else "TAKE OUT!", (x1 + 2, y1 - 8), 0.55, colour, 2)


def draw_bank(frame, state):
    x1, y1, _, _ = ZONE_RECTS[BANK]
    # Items stacked from top to bottom, 72 pixels apart. "_" = a value we do not need.
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


HELD_ITEM_SIZE = 70          # size of the item drawn in a player's hand (pixels)
PALM_OFFSET = 0.25           # how far past the wrist the palm is (fraction of the forearm)


def hand_position(player, wrist):
    """Estimated position of the palm for "left_wrist" or "right_wrist", or None.

    MediaPipe's wrist point sits at the wrist, not in the hand. The hand
    continues in the same direction as the forearm (elbow -> wrist), so we
    step a bit further along that line.

    Example: elbow (400, 400), wrist (400, 300)  ->  palm (400, 275)
    """
    keypoints = player["keypoints"]
    wrist_point = keypoints.get(wrist)
    if not is_reliable(wrist_point):        # no trustworthy wrist: we cannot say
        return None
    x, y, _ = wrist_point
    elbow = keypoints.get(wrist.replace("wrist", "elbow"))   # "left_wrist" -> "left_elbow"
    # If the elbow is known, step further along the forearm direction.
    # (x - ex, y - ey) is the vector from elbow to wrist; 25% of it more = the palm.
    # Without a reliable elbow we simply use the wrist itself.
    if is_reliable(elbow):
        ex, ey, _ = elbow
        x += int(PALM_OFFSET * (x - ex))
        y += int(PALM_OFFSET * (y - ey))
    return x, y


def draw_player(frame, player, state, holding_hand=None):
    """Keypoints (green = reliable, red = unsure) and what the player holds, in their hand.

    holding_hand: "left_wrist" or "right_wrist", the hand that grabbed the item.
    If that hand is not visible, the other hand is used, and as a last resort
    the item is drawn above the head.
    """
    if not player["visible"]:
        return
    # Step 1: a dot on every keypoint. Colours are BGR: (0,255,0) = green, (0,0,255) = red.
    for keypoint in player["keypoints"].values():
        x, y, _ = keypoint
        cv2.circle(frame, (x, y), 6, (0, 255, 0) if is_reliable(keypoint) else (0, 0, 255), -1)

    # Step 2: the item this player holds (the game state knows what, we know where).
    holding = state.players[player["player_id"]].holding
    if holding is None:
        return

    # Try the hand that grabbed it first, then the other hand.
    # "holding_hand or 'right_wrist'": use the right hand if we do not know yet.
    first = holding_hand or "right_wrist"
    other = "left_wrist" if first == "right_wrist" else "right_wrist"
    half = HELD_ITEM_SIZE // 2              # to centre the item: move it half its size up-left
    for wrist in (first, other):
        palm = hand_position(player, wrist)
        if palm is not None:
            x, y = palm
            draw_item(frame, holding, x - half, y - half, HELD_ITEM_SIZE)   # centred on the palm
            return

    nose = player["keypoints"].get("nose")       # no hand visible: show it above the head
    if is_reliable(nose):
        x, y, _ = nose
        draw_item(frame, holding, x - half, y - 150, HELD_ITEM_SIZE)


def draw_hold_bar(frame, x, y, progress):
    """Grey bar above a wrist that fills green during a 1 s hold."""
    # An 80-pixel-wide bar, centred above the point: grey background first,
    # then a green part on top whose width = 80 x progress (0.5 -> 40 pixels).
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
    # Darken the whole picture: make an all-black copy, then mix it with the frame,
    # 60% black + 40% original (the same blending idea as in blit, for the whole image).
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)   # darken the picture
    put_text(frame, "GAME OVER", (430, 320), 2.0, (0, 0, 255), 4)
    put_text(frame, f"Pizzas served: {state.score}", (480, 390), 1.0, WHITE, 2)
    put_text(frame, "press R to play again, Q to quit", (420, 440), 0.8)


def draw_everything(frame, players, state, hold_bars, holding_hands=None):
    """holding_hands: {player_id: "left_wrist" / "right_wrist"}, which hand holds the item."""
    holding_hands = holding_hands or {}
    # ORDER MATTERS: things drawn later appear on top. So: zones and panels
    # first, then players and what they hold, then hold bars and messages,
    # and the game-over screen last (on top of everything).
    draw_zones(frame)
    draw_timer(frame, state)
    draw_order(frame, state)
    draw_pizza_status(frame, state)
    draw_oven(frame, state)
    draw_bank(frame, state)
    draw_abilities(frame, state)
    for player in players:
        draw_player(frame, player, state, holding_hands.get(player["player_id"]))
    for x, y, progress in hold_bars:
        draw_hold_bar(frame, x, y, progress)
    draw_active_effects(frame, state)
    draw_message(frame, state)
    if state.game_over:
        draw_game_over(frame, state)
