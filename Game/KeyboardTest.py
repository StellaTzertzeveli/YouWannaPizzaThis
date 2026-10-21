"""
Pizza Kitchen - keyboard test window (NOT part of the final game).
Needs game_objects.py and game_state.py in the same folder.

Player 1 = Chef (right side), Player 2 = Sous Chef (left side).
"""
import os
import time

from game_objects import TOPPINGS, CHOP_STRIKES, Ingredient, Pizza, CHEF, SOUS_CHEF
from game_state import GameState, Command

# =====================================================================
# KEYBOARD
# =====================================================================
KEYMAP = {
    "1": (SOUS_CHEF, "grab_raw", TOPPINGS[0]), "2": (SOUS_CHEF, "grab_raw", TOPPINGS[1]),
    "3": (SOUS_CHEF, "grab_raw", TOPPINGS[2]), "4": (SOUS_CHEF, "grab_raw", TOPPINGS[3]),
    "f": (SOUS_CHEF, "chop", None), "g": (SOUS_CHEF, "pass_to_bank", None),
    "h": (SOUS_CHEF, "take_pizza_from_bank", None), "j": (SOUS_CHEF, "deliver", None),
    "z": (SOUS_CHEF, "ability", "knife_frenzy"), "x": (SOUS_CHEF, "ability", "rush_delivery"),
    "u": (CHEF, "spin_dough", None), "i": (CHEF, "place_at_station", None),
    "o": (CHEF, "add_topping", None), "k": (CHEF, "oven_in", None),
    "l": (CHEF, "oven_out", None), "p": (CHEF, "take_from_bank", None),
    ";": (CHEF, "place_on_bank", None),
    "n": (CHEF, "ability", "hot_oven"), "m": (CHEF, "ability", "stubborn"),
}


# =====================================================================
# SELF TEST (also runs without OpenCV)
# =====================================================================
def selftest():
    """Full flow: 3 pizzas delivered, 1 burnt, every ability used once."""
    g = GameState(seed=1)
    t = [0.0]
    log = []

    def do(player, action, arg=None):
        t[0] += 0.2
        ok = g.apply(Command(player, action, arg, t[0]))
        assert ok, f"{action} refused: {g.last_message}"

    def wait(sec):
        t[0] += sec
        log.extend(g.update(t[0]))

    def build_pizza():
        for top in list(g.order.toppings):
            do(SOUS_CHEF, "grab_raw", top)
            for _ in range(CHOP_STRIKES):
                if g.sous.holding.state == "chopped":
                    break
                do(SOUS_CHEF, "chop")
            do(SOUS_CHEF, "pass_to_bank")
        do(CHEF, "spin_dough")
        do(CHEF, "place_at_station")
        for _ in list(g.order.toppings):
            do(CHEF, "take_from_bank")
            do(CHEF, "add_topping")
        assert g.pizza.stage == "topped"

    g.update(0.0)
    # Pizza 1: Knife Frenzy, normal bake, normal delivery
    do(SOUS_CHEF, "ability", "knife_frenzy")
    build_pizza(); do(CHEF, "oven_in"); wait(6)
    do(CHEF, "oven_out"); do(CHEF, "place_on_bank"); do(SOUS_CHEF, "take_pizza_from_bank"); do(SOUS_CHEF, "deliver")
    # Pizza 2: Hot Oven, then Rush Delivery straight from the oven
    build_pizza(); do(CHEF, "ability", "hot_oven"); do(CHEF, "oven_in"); wait(3)
    do(SOUS_CHEF, "ability", "rush_delivery")
    # Pizza 3: left in the oven, burns
    build_pizza(); do(CHEF, "oven_in"); wait(11)
    # Pizza 4: Stubborn replaces the order, then normal delivery
    do(CHEF, "ability", "stubborn")
    build_pizza(); do(CHEF, "oven_in"); wait(6)
    do(CHEF, "oven_out"); do(CHEF, "place_on_bank"); do(SOUS_CHEF, "take_pizza_from_bank"); do(SOUS_CHEF, "deliver")

    log.extend(g.update(t[0]))
    print(g)
    assert g.score == 3, g.score
    assert log.count("pizza_burnt") >= 1
    print("\nSELFTEST PASSED: 3 delivered, 1 burnt, all 4 abilities used.")


# =====================================================================
# INGREDIENT PICTURES
# One PNG per ingredient state, in the folder  assets/ingredients/  next to this file:
#   cheese_raw.png, cheese_cut_1.png ... cheese_cut_3.png, cheese_sliced.png
# (same for pepperoni, mushrooms, peppers).  Missing files are created as simple
# placeholders the first time you run this file - replace them with your own art,
# keeping the same file names.
# =====================================================================
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ingredients")
SPRITE_SIZE = 128             # size of the generated placeholder PNGs
CROP_TO_CONTENT = True        # trim empty transparent space around your art so the ingredient fills the box
CROP_PADDING = 0.06           # breathing room kept around the ingredient (fraction of its size)
CELL_SIZE = 140               # size of each picture box in the window (make bigger/smaller here)
WINDOW_W, WINDOW_H = 1440, 680
INGREDIENT_COLORS = {            # BGR, only used for the placeholders
    "cheese": (60, 200, 250), "pepperoni": (60, 60, 210),
    "mushrooms": (150, 175, 200), "peppers": (60, 180, 60),
}
_sprite_cache = {}


def all_ingredient_image_names():
    names = []
    for t in TOPPINGS:
        names.append(f"{t}_raw")
        names += [f"{t}_cut_{i}" for i in range(1, CHOP_STRIKES)]
        names.append(f"{t}_sliced")
    return names


def _make_placeholder(name):
    import numpy as np
    import cv2
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
    cv2.putText(img, name, (4, SPRITE_SIZE - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ensure_placeholder_assets():
    """Creates any missing ingredient PNGs. Returns how many were created."""
    import cv2
    os.makedirs(ASSET_DIR, exist_ok=True)
    made = 0
    for name in all_ingredient_image_names():
        path = os.path.join(ASSET_DIR, name + ".png")
        if not os.path.exists(path):
            cv2.imwrite(path, _make_placeholder(name))
            made += 1
    return made


def report_assets():
    """Prints which pictures are being used and their sizes, so you can see if your own PNGs are found."""
    import cv2
    print(f"Reading ingredient pictures from: {ASSET_DIR}")
    mine, placeholders, missing = [], [], []
    for name in all_ingredient_image_names():
        img = cv2.imread(os.path.join(ASSET_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            missing.append(name)
        elif img.shape[:2] == (SPRITE_SIZE, SPRITE_SIZE):
            placeholders.append(name)           # exactly the generated placeholder size
        else:
            mine.append(f"{name} ({img.shape[1]}x{img.shape[0]})")
    print(f"  your own pictures: {len(mine)}   still placeholders ({SPRITE_SIZE}x{SPRITE_SIZE}): {len(placeholders)}   missing: {len(missing)}")
    for m in mine:
        print("   ", m)
    if placeholders:
        print("  placeholders:", ", ".join(placeholders))


_crop_cache = {}


def _alpha_box(img):
    """Where the visible (non-transparent) pixels are, as fractions (x0, y0, x1, y1). None if the image has no transparency."""
    import numpy as np
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
    """One crop box shared by ALL pictures of an ingredient (raw, cut_1..5, sliced),
    so the ingredient keeps the same size from picture to picture instead of jumping."""
    import cv2
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
    """Loads assets/ingredients/<name>.png, fitted INSIDE a size x size box without stretching
    (keeps the original proportions, extra space is transparent). None if the file is missing."""
    import cv2
    import numpy as np
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
            if crop and _alpha_box(img):                  # only crop pictures that have transparency
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
    """Paste a BGRA sprite onto img at (x, y), respecting transparency."""
    h, w = sprite.shape[:2]
    roi = img[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):
        return
    a = sprite[:, :, 3:4] / 255.0
    roi[:] = (sprite[:, :, :3] * a + roi * (1 - a)).astype("uint8")


def draw_cell(img, x, y, size, sprite_name=None, text=None, caption=""):
    import cv2
    cv2.rectangle(img, (x, y), (x + size, y + size), (90, 90, 90), 1)
    sprite = load_sprite(sprite_name, size - 2) if sprite_name else None
    if sprite is not None:
        blit(img, sprite, x + 1, y + 1)
    elif text:
        cv2.putText(img, text, (x + 4, y + size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    if caption:
        cv2.putText(img, caption, (x, y + size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def _item_cell(img, x, y, size, item, caption):
    if isinstance(item, Ingredient):
        draw_cell(img, x, y, size, item.image_name, item.image_name, caption)
    elif isinstance(item, Pizza):
        draw_cell(img, x, y, size, None, f"pizza:{item.stage}", caption)
    else:
        draw_cell(img, x, y, size, None, None, caption)


def draw_kitchen_panel(img, state, y0=470, size=None):
    """Pictures: what each player holds, the order, and the bank slots."""
    import cv2
    size = size or CELL_SIZE
    gap = 10
    title = lambda t, x: cv2.putText(img, t, (x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    top = y0 + 12
    x_sous = 10
    x_chef = x_sous + size + gap
    x_order = x_chef + size + 40
    x_bank = x_order + 3 * (size + gap) + 30
    title("SOUS HOLDS", x_sous)
    _item_cell(img, x_sous, top, size, state.sous.holding, "")
    title("CHEF HOLDS", x_chef)
    _item_cell(img, x_chef, top, size, state.chef.holding, "")
    title("ORDER", x_order)
    for i, t in enumerate(state.order.toppings):
        done = t in state.pizza.toppings
        draw_cell(img, x_order + i * (size + gap), top, size, f"{t}_sliced", t, "on pizza" if done else "needed")
    title("BANK", x_bank)
    for i in range(state.bank.max_items):
        item = state.bank.items[i] if i < len(state.bank.items) else None
        _item_cell(img, x_bank + i * (size + gap), top, size, item, "")


def draw_window(state, events_log):
    """Draws the key legend (left) and the live game state (right) into an image."""
    import textwrap
    import numpy as np
    import cv2
    img = np.full((WINDOW_H, WINDOW_W, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, x, y, color=(220, 220, 220), size=0.45):
        cv2.putText(img, text, (x, y), font, size, color, 1, cv2.LINE_AA)

    # left: keys
    y = 24
    put("KEYS  (click this window first)", 10, y, (255, 255, 255), 0.5)
    for k, (p, a, arg) in KEYMAP.items():
        y += 19
        color = (255, 180, 120) if p == CHEF else (120, 220, 120)
        put(f"{k}  {'Chef' if p == CHEF else 'Sous'}  {a} {arg or ''}", 10, y, color)
    y += 26
    put("w = skip 5 s    9 = Sous lost/back", 10, y, (180, 180, 180))
    y += 19
    put("0 = Chef lost/back   t = selftest (terminal)", 10, y, (180, 180, 180))
    y += 19
    put("ESC = quit", 10, y, (180, 180, 180))

    # right: state
    x, y = 400, 24
    put("GAME STATE", x, y, (255, 255, 255), 0.5)
    lines = str(state).split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            color = (120, 220, 120) if line.startswith("OK") else (90, 90, 255)
        else:
            color = (220, 220, 220)
        for part in textwrap.wrap(line, 68) or [""]:
            y += 20
            put(part, x, y, color, 0.5 if i == 0 else 0.45)
    y += 34
    put("RECENT EVENTS", x, y, (255, 255, 255), 0.5)
    for ev in events_log[-8:]:
        y += 19
        put(ev, x, y, (120, 200, 255))
    draw_kitchen_panel(img, state)
    put(f"picture box = {CELL_SIZE}px   window = {WINDOW_W}x{WINDOW_H}   (change CELL_SIZE / WINDOW_W / WINDOW_H in keyboard_test.py)",
        10, WINDOW_H - 10, (150, 150, 150), 0.45)
    return img


# =====================================================================
# MAIN LOOPS
# =====================================================================
def main():
    """OpenCV window: reads keys, turns them into commands, prints the state after every command."""
    import cv2
    state = GameState()
    offset = 0.0
    now = lambda: time.time() + offset
    events_log = []
    made = ensure_placeholder_assets()
    if made:
        print(f"Created {made} placeholder PNGs in {ASSET_DIR} - replace them with your own art (same file names).")
    report_assets()
    state.update(now())
    print("Window opened. Click it, then press keys. ESC quits. State prints here after every command.")
    print(state)
    while True:
        cv2.imshow(f"Pizza Kitchen - keyboard test (picture box {CELL_SIZE}px)", draw_window(state, events_log))
        k = cv2.waitKey(30) & 0xFF
        t = now()

        for ev in state.update(t):              # timer events: baked, burnt, game_over...
            events_log.append(ev)
            print("EVENT:", ev)

        if k == 27:                             # ESC
            break
        if k == 255:                            # no key pressed
            continue
        key = chr(k).lower()

        if key == "w":
            offset += 5
            print("\n(skipped 5 seconds)")
        elif key == "9":
            state.set_visible(SOUS_CHEF, not state.sous.visible)
        elif key == "0":
            state.set_visible(CHEF, not state.chef.visible)
        elif key == "t":
            selftest()
            continue
        elif key in KEYMAP:
            player, action, arg = KEYMAP[key]
            print(f"\nCOMMAND: player={player} {action} {arg or ''}")
            state.apply(Command(player, action, arg, t))
        else:
            continue

        print(state)                            # state after EVERY command
        for ev in state.update(t):
            events_log.append(ev)
            print("EVENT:", ev)
    cv2.destroyAllWindows()


def main_typing():
    """Fallback without OpenCV: type keys and press Enter."""
    state = GameState()
    offset = 0.0
    now = lambda: time.time() + offset
    state.update(now())
    print("Type keys then Enter. 'wait 5' skips 5 s, 'test' runs selftest, 'q' quits.")
    print(state)
    while True:
        text = input("> ").strip().lower()
        if text == "q":
            break
        if text == "test":
            selftest()
            continue
        if text.startswith("wait"):
            offset += float(text.split()[1])
        t = now()
        for key in ("" if text.startswith("wait") else text):
            if key == "9":
                state.set_visible(SOUS_CHEF, not state.sous.visible)
            elif key == "0":
                state.set_visible(CHEF, not state.chef.visible)
            elif key in KEYMAP:
                p, a, arg = KEYMAP[key]
                state.apply(Command(p, a, arg, t))
        events = state.update(t)
        print(state)
        if events:
            print("EVENTS:", events)


if __name__ == "__main__":
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("OpenCV not installed (pip install opencv-python numpy). Using typing mode instead.")
        main_typing()
    else:
        main()