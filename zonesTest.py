import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from zones import *
from images import draw_fill, blend_person
from game_state import GameState
from motion_detectors.ChopDetector import ChopDetector
from motion_detectors.SpinDetector import SpinDetector
from motion_detectors.CrossDetector import CrossDetector
from motion_detectors.ThumbsupDetector import ThumbsupDetector

ABILITY_COOLDOWN = 5  # seconds an ability stays on cooldown after it's used

latest_hand_result = None
latest_pose_result = None
latest_mask = None

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result

def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result

def segment_callback(result, output_image, timestamp_ms):
    global latest_mask
    latest_mask = np.array(result.confidence_masks[-1].numpy_view())

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_callback,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=pose_callback,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

segmenter_options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path="selfie_segmenter.tflite"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_confidence_masks=True,
    output_category_mask=False,
    result_callback=segment_callback,
)
segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)

# Create zones here
zones = make_zones()

# Create a game state here
state = GameState()

# Create a chop detector here
chop_detector = ChopDetector()
chop_detector.start_detection()

# Create a spin detector here
spin_detector = SpinDetector()
spin_detector.start_detection()

# Create a cross detector here (arms forming an X, read from pose landmarks)
cross_detector = CrossDetector()
cross_detector.start_detection()
last_cross_use = -ABILITY_COOLDOWN

# Create a thumbs-up detector here (read from hand landmarks)
thumbsup_detector = ThumbsupDetector()
thumbsup_detector.start_detection()
last_thumbsup_use = -ABILITY_COOLDOWN

cv2.namedWindow("Zones", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Zones", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    hand_landmarker.detect_async(mp_image, timestamp)
    pose_landmarker.detect_async(mp_image, timestamp)
    segmenter.segment_async(mp_image, timestamp)

    camera = frame.copy()
    draw_fill(frame, "background", 0, 0, w, h)
    if latest_mask is not None:
        blend_person(frame, camera, latest_mask)

    # Check if the palm of any detected hand is inside the bank zone
    now = time.time()
    palms = []
    if latest_hand_result is not None:
        for hand in latest_hand_result.hand_landmarks:
            palm = hand[9]  # middle-finger knuckle = roughly the centre of the palm
            palms.append(palm)
            cv2.circle(frame, (int(palm.x * w), int(palm.y * h)), 8, (0, 255, 0), -1)
            chop_detector.update(hand)
            spin_detector.update(hand)
            thumbsup_detector.update(hand)

    if latest_pose_result is not None:
        for body in latest_pose_result.pose_landmarks:
            cross_detector.update(body)

    # Hand every new chop to the game state, then reset the detector's count
    for _ in range(chop_detector.get_chop_count()):
        state.on_chop()
    chop_detector.reset_chop_count()

    if state.holding != "DOUGH":
        spin_detector.count = 0
    if spin_detector.count >= spin_detector.max_value:
        state.on_spin()
        spin_detector.count = 0

    # Cross ability: fires once per completed X-hold, then sits on cooldown
    if cross_detector.state:
        cross_detector.state = False
        if now - last_cross_use >= ABILITY_COOLDOWN:
            print(f"Cross ability used! On cooldown for {ABILITY_COOLDOWN}s.")
            last_cross_use = now

    # Thumbs-up ability: same fire-once + cooldown pattern
    if thumbsup_detector.state:
        thumbsup_detector.state = False
        if now - last_thumbsup_use >= ABILITY_COOLDOWN:
            print(f"Thumbs-up ability used! On cooldown for {ABILITY_COOLDOWN}s.")
            last_thumbsup_use = now

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
    if key == ord('q'):
        break
    if key == ord(' '):
        state.toggle_timer()

cap.release()
cv2.destroyAllWindows()