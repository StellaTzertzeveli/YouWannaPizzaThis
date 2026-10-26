"""Webcam test for A2: hold a hand in a box for 1 s -> exactly ONE command.

    python test_dwell.py      (press q to quit)

Each wrist shows a hold bar that fills up in 1 second. When it is full,
the terminal prints the command once. Keep your hand there: nothing more
happens. Move it out and back in: it can fire again.
"""
import time

import cv2

from commands import Command, GRAB
from fake_input import FakeInput, is_reliable
from zones import (prepare_frame, zone_at, ZONE_RECTS, BOX_INGREDIENT,
                   DwellTimer)

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
tracker = FakeInput()

# One DwellTimer per hand per player.
# The keys are pairs like (2, "left_wrist"): player 2's left hand.
timers = {}
for player_id in (1, 2):
    for wrist in ("left_wrist", "right_wrist"):
        timers[(player_id, wrist)] = DwellTimer()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.monotonic()
    frame = prepare_frame(frame)
    players = tracker.read(frame)

    for (x1, y1, x2, y2) in ZONE_RECTS.values():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for player in players:
        for wrist in ("left_wrist", "right_wrist"):
            timer = timers[(player["player_id"], wrist)]

            # Which zone is this hand in? None if the hand is lost or unsure.
            keypoint = player["keypoints"].get(wrist)
            if player["visible"] and is_reliable(keypoint):
                x, y, _ = keypoint
                zone = zone_at(x, y)
            else:
                zone = None

            fired = timer.update(zone, now)

            if fired:
                if fired in BOX_INGREDIENT:
                    command = Command(player_id=player["player_id"],
                                      role=player["role"],
                                      action=GRAB,
                                      item=BOX_INGREDIENT[fired])
                    print(command)
                else:
                    # Other zones get their real commands later, in players.py
                    print(f"P{player['player_id']} held a hand in {fired}")

            # Hold bar above the wrist: grey background, green fill
            if zone is not None:
                bar_x, bar_y = x - 40, y - 30
                filled = int(80 * timer.progress(now))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 80, bar_y + 10),
                              (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 10),
                              (0, 255, 0), -1)

    cv2.imshow("Dwell test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()
