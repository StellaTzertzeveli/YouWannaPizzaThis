import os
import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FOLDERS = [os.path.join(ASSETS, "ingredients"), os.path.join(ASSETS, "zone_sprites")]  # where pictures are searched, in this order
cache = {}  # name -> loaded image (or None if the file doesn't exist), so each file is read only once


def load(name):
    if name not in cache:
        img = None
        paths = [os.path.join(folder, name + ext) for folder in FOLDERS for ext in (".png", ".jpg")]
        for path in paths:
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                break
        if img is not None:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # no transparency in the file: make it fully solid
            ys, xs = np.where(img[:, :, 3] > 20)  # rows and columns of every clearly visible pixel
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # crop away the empty border
        cache[name] = img
    return cache[name]


def draw_image(frame, name, x, y, size):
    # Draws the image to fit in a size x size square with its top-left corner at (x, y). False if there is no such image.
    img = load(name)
    if img is None or size <= 0:
        return False
    h, w = img.shape[:2]
    scale = size / max(h, w)  # shrink the longest side to 'size', keep the shape
    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    x += (size - img.shape[1]) // 2  # centre a wide or tall picture inside the square
    y += (size - img.shape[0]) // 2
    if x < 0:
        img, x = img[:, -x:], 0  # sticks out on the left: cut that part off
    if y < 0:
        img, y = img[-y:], 0     # sticks out at the top: cut that part off
    h, w = img.shape[:2]
    h, w = min(h, frame.shape[0] - y), min(w, frame.shape[1] - x)  # don't draw past the right or bottom edge
    if h <= 0 or w <= 0:
        return True  # completely outside the frame: nothing to draw
    img = img[:h, :w]
    alpha = img[:, :, 3:] / 255.0  # 0 = see-through, 1 = solid
    region = frame[y:y + h, x:x + w]
    region[:] = (img[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    return True


def draw_fill(frame, name, x1, y1, x2, y2):
    # Stretches the image to exactly fill the rectangle (x1, y1)-(x2, y2). False if there is no such image.
    img = load(name)
    if img is None or x2 <= x1 or y2 <= y1:
        return False
    key = (name, x2 - x1, y2 - y1)
    if key not in cache:
        cache[key] = cv2.resize(img, (x2 - x1, y2 - y1))  # resize once per size, not every frame
    img = cache[key]
    alpha = img[:, :, 3:] / 255.0
    region = frame[y1:y2, x1:x2]
    region[:] = (img[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    return True

def blend_person(frame, camera, mask):
    # Puts the camera image back wherever the mask says "person" (1), keeps the frame wherever it says "background" (0)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    m = mask.reshape(mask.shape[0], mask.shape[1], 1)  # always (height, width, 1), so it multiplies all 3 colour channels
    frame[:] = (camera * m + frame * (1 - m)).astype(np.uint8)

if __name__ == "__main__":
    # A fake 4x4 image: left half solid red, right half see-through
    test = np.zeros((4, 4, 4), dtype=np.uint8)
    test[:, :2] = (0, 0, 255, 255)
    cache["test"] = test
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert draw_image(frame, "test", 0, 0, 4)
    assert frame[0, 0].tolist() == [0, 0, 255]    # solid part: red
    assert frame[0, 3].tolist() == [100, 100, 100]  # see-through part: background unchanged
    assert draw_image(frame, "test", 8, 8, 4)     # partly off the bottom-right: no crash
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert draw_image(frame, "test", -1, -2, 4)   # partly off the top-left: no crash
    assert frame[0, 0].tolist() == [0, 0, 255]    # the visible part of the red half landed in the corner
    assert draw_image(frame, "test", 50, 50, 4)   # completely outside: no crash
    assert draw_image(frame, "does_not_exist", 0, 0, 4) is False

    cache["big"] = np.full((100, 100, 4), 255, dtype=np.uint8)  # a solid white 100x100 picture
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_image(frame, "big", 0, 0, 10)
    assert frame[20, 20].tolist() == [0, 0, 0]  # shrunk to 10x10, so (20, 20) stays black

    # draw_fill stretches to the exact rectangle
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert draw_fill(frame, "big", 10, 20, 40, 30)
    assert frame[25, 35].tolist() == [255, 255, 255]  # inside the rectangle: filled
    assert frame[35, 35].tolist() == [0, 0, 0]        # below it: untouched
    assert draw_fill(frame, "does_not_exist", 0, 0, 10, 10) is False
    print("All image tests passed!")