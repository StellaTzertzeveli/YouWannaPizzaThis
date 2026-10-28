import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from zones import *
from game_state import GameState
from motion_detectors.ChopDetector import ChopDetector

latest_hand_result = None

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_callback,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Create zones here
zones = make_zones()

# Create a game state here
state = GameState()

# Create a chop detector here
chop_detector = ChopDetector()
chop_detector.start_detection()

cv2.namedWindow("Zones", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Zones", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                 int(time.time() * 1000))

    # Check if the palm of any detected hand is inside the bank zone
    now = time.time()
    palms = []
    if latest_hand_result is not None:
        for hand in latest_hand_result.hand_landmarks:
            palm = hand[9]  # middle-finger knuckle = roughly the centre of the palm
            palms.append(palm)
            cv2.circle(frame, (int(palm.x * w), int(palm.y * h)), 8, (0, 255, 0), -1)
            chop_detector.update(hand)
            # Hand every new chop to the game state, then reset the detector's count
        for _ in range(chop_detector.get_chop_count()):
            state.on_chop()
            chop_detector.reset_chop_count()

    state.update(now)
    for zone in zones:
        inside = any(zone.contains(p.x, p.y) for p in palms)
        if zone.update(inside, now):
            # Tell game state that the zone was triggered, so it can update the held item or surface items accordingly
            state.on_hold(zone.name)
            print(f"{zone.name} triggered! Holding: {state.holding}")
        zone.draw(frame, progress=zone.progress(now))
    # Draw the HUD (not interactible)
    draw_hud(frame)
    # Draw the game zones (interactible)
    state.draw(frame, zones, palms)
    cv2.imshow("Zones", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        state.on_spin()
    if key == ord('q'):
        break
    if key == ord(' '):
        state.toggle_timer()
        

cap.release()
cv2.destroyAllWindows()