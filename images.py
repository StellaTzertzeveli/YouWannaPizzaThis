import os
import cv2
import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ingredients")
cache = {}  # name -> loaded image (or None if the file doesn't exist), so each file is read only once


def load(name):
    if name not in cache:
        path = os.path.join(FOLDER, name + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is not None:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # no transparency in the file: make it fully solid
            ys, xs = np.where(img[:, :, 3] > 0)  # rows and columns of every visible pixel
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