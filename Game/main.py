"""Pizza Kitchen - Task 4 (Player Interaction and Game Control).

HOW TO RUN
    python main.py              webcam game  (q = quit, r = restart)
    python main.py --selftest   check the game rules, no camera needed
    python main.py --camera 1   use another camera (if you get a black picture)

WHAT HAPPENS EVERY FRAME (the pipeline), and in which file
    webcam frame
      -> prepare_frame()        mirror + resize to 1280 x 720       zones.py
      -> FakeInput.read()       keypoints for 2 players              fake_input.py (later: Tasks 2 + 3)
      -> zone_at + DwellTimer   "a hand was held in a zone for 1 s"  zones.py
      -> zone_to_command()      turn that into a Command             players.py
      -> to_rule_command()      translate it for the rules           players.py
      -> GameState.apply()      allowed right now? update the game   game_state.py
      -> draw_everything()      visual feedback on the frame         feedback.py

THE FILES
    main.py          this file: settings and the game loop
    commands.py      the Command message and all action/item names (team interface)
    zones.py         screen zones, DwellTimer, Cooldown, prepare_frame
    fake_input.py    TEMPORARY keypoints (MediaPipe Pose per screen half)
    game_objects.py  players, Ingredient, Pizza, Order, Oven, Bank
    game_state.py    the game rules (GameState) and the abilities
    players.py       zones -> commands, and camera commands -> rule commands
    detectors.py     keyboard stand-ins for motions/poses (detectors go here)
    feedback.py      all drawing
    selftest.py      rule tests without a camera
"""
import argparse      # reads options typed after "python main.py" (--selftest, --camera)
import sys           # sys.exit() stops the program with a message; sys.platform = which OS
import time          # the clock: time.monotonic() is used for every timer in the game

import cv2           # OpenCV: webcam and the game window

# Our own modules: each file is one part of the game (see THE FILES above).
from commands import CHEF_ID, SOUS_CHEF_ID
from zones import prepare_frame, zone_at, DwellTimer
from fake_input import FakeInput
from game_state import GameState
from players import WRISTS, is_reliable, to_rule_command, zone_to_command, hold_seconds_for
from detectors import key_to_command
from feedback import ASSET_DIR, ensure_placeholder_assets, report_assets, draw_everything
from selftest import selftest


# =====================================================================
# SETTINGS
# =====================================================================
CAMERA_INDEX = 0            # 0 = first camera.
LOST_AFTER_SECONDS = 0.5    # a player missing this long counts as "left the frame"
WINDOW_NAME = "Pizza Kitchen"


# =====================================================================
# MAIN LOOP
# =====================================================================
def apply_command(state, command):
    """Send one camera Command through the bridge to the rules and print the result."""
    rule = to_rule_command(command)
    if rule is None:
        print(f"No rule for {command.role} {command.action}")
        return False
    ok = state.apply(rule)
    print(f"P{command.player_id} {command.action} {command.item or ''} -> {state.last_message}")
    return ok          # True = the rules accepted it


def run_game(camera_index):
    # ---- Setup (runs once) ----------------------------------------------
    # Open the webcam. DirectShow often fixes a black picture on Windows;
    # elsewhere use the default.
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, api)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_index}. Try: python main.py --camera 1")

    # Make sure every ingredient picture exists, and report which are placeholders.
    made = ensure_placeholder_assets()
    if made:
        print(f"Created {made} placeholder pictures in {ASSET_DIR}")
    report_assets()

    # The keypoint source (temporary; later the real Task 2 + 3 tracker).
    tracker = FakeInput()

    def new_game(now):
        """Fresh game state, dwell timers and 'last seen' times."""
        state = GameState()
        state.update(now)
        # One DwellTimer per hand per player, keys like (2, "left_wrist")
        timers = {(pid, wrist): DwellTimer() for pid in (CHEF_ID, SOUS_CHEF_ID) for wrist in WRISTS}
        last_seen = {CHEF_ID: now, SOUS_CHEF_ID: now}
        holding_hands = {}            # {player id: "left_wrist" or "right_wrist"}
        return state, timers, last_seen, holding_hands

    # A function returns several values at once as a tuple; unpack them into 4 variables.
    state, timers, last_seen, holding_hands = new_game(time.monotonic())
    print("Game started. Hold a hand in a zone for 1 s to use it.")
    print("Keys: F chop | U spin | O sprinkle | Z/X Sous abilities | N/M Chef abilities | R restart | Q quit")

    # ---- The game loop: everything below runs once per webcam frame ------
    while True:
        # Read one frame. ret ("return") is False when no picture came.
        ret, frame = cap.read()
        if not ret:
            print("No picture from the camera. Is another app (Teams, Zoom...) using it?")
            break

        # ONE time value for the whole frame, so every part agrees on "now".
        # monotonic = a clock that never jumps back (unlike the wall clock).
        now = time.monotonic()
        frame = prepare_frame(frame)          # mirror + resize, always first
        players = tracker.read(frame)         # keypoints for both players

        # --- 1. Turn hands into commands -------------------------------------
        commands = []
        hold_bars = []
        for player in players:
            pid = player["player_id"]

            # A player only counts as "lost" after LOST_AFTER_SECONDS without
            # being seen, so a single missed frame does not pause their actions.
            if player["visible"]:
                last_seen[pid] = now
            present = now - last_seen[pid] < LOST_AFTER_SECONDS
            if state.players[pid].visible != present:
                state.set_visible(pid, present)

            # Check both hands of this player.
            for wrist in WRISTS:
                # a) Which zone is the hand in? None if it is unsure or in no zone.
                keypoint = player["keypoints"].get(wrist)
                zone = None
                if player["visible"] and is_reliable(keypoint):
                    x, y, _ = keypoint
                    zone = zone_at(x, y)

                # b) Feed it to this hand's own dwell timer. It "fires" once
                #    when the hand stayed long enough (1 s, or 2 s to sprinkle).
                timer = timers[(pid, wrist)]
                fired = timer.update(zone, now, hold_seconds_for(player, zone, state))
                if zone is not None:
                    hold_bars.append((x, y, timer.progress(now)))   # draw later
                # c) A completed hold becomes a Command (what it means depends on the game).
                if fired:
                    command = zone_to_command(player, fired, state, now)
                    if command is not None:
                        commands.append((command, wrist))   # remember WHICH hand did it

        # --- 2. Apply them to the game rules ---------------------------------
        # Both players at the bank in the same frame: the rules let the first one win.
        for command, wrist in commands:
            accepted = apply_command(state, command)
            pid = command.player_id
            # The hand that just picked something up is the hand that holds it.
            if accepted and state.players[pid].holding is not None:
                holding_hands[pid] = wrist
        for event in state.update(now):       # baked, burnt, game_over, ...
            print("EVENT:", event)

        # --- 3. Visual feedback ----------------------------------------------
        draw_everything(frame, players, state, hold_bars, holding_hands)
        cv2.imshow(WINDOW_NAME, frame)

        # --- 4. Keyboard -----------------------------------------------------
        # waitKey(1) waits at most 1 ms for a key AND lets OpenCV update the
        # window (without it, imshow shows nothing). "& 0xFF" keeps only the last
        # 8 bits of the key code, which works the same on every OS.
        key_code = cv2.waitKey(1) & 0xFF
        if key_code == 255:                   # no key pressed
            continue                          # skip the rest, go to the next frame
        key = chr(key_code).lower()           # number -> letter, e.g. 102 -> "f"
        if key == "q":
            break
        if key == "r":
            state, timers, last_seen, holding_hands = new_game(time.monotonic())
            print("New game!")
            continue
        # Game keys go through the SAME path as camera actions: Command -> rules.
        command = key_to_command(key, time.monotonic())
        if command is not None:
            apply_command(state, command)

    # ---- Clean up: free the camera and close the window -----------------
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


# This block only runs when you start THIS file ("python main.py"), not when
# another file imports it ("from main import run_game"). So teammates can
# reuse the classes without the game starting by itself.
if __name__ == "__main__":
    # Read the options typed after "python main.py".
    parser = argparse.ArgumentParser(description="Pizza Kitchen (Task 4)")
    parser.add_argument("--selftest", action="store_true",
                        help="check the game rules without a camera")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="camera number (0 = first camera)")
    args = parser.parse_args()               # args.selftest is True/False, args.camera a number

    # Either run the rule tests, or the real game.
    if args.selftest:
        selftest()
    else:
        run_game(args.camera)
