import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from motion_detection import new_chop_state, CHOPPED, LIFTING_UPWARDS, CHOPPING_DOWNWARDS, draw_taskbar
CHOP_STATE = LIFTING_UPWARDS
latest_hand_result = None
latest_pose_result = None

display_pose = False
chop_amount = 0
hand_results = []

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result


def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result


# ---------------------HAND LANDMARKER----------------------
hand_model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
# 3. Configure Pose Landmarker

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=hand_model_path
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_callback
)
hand_landmarker = vision.HandLandmarker.create_from_options(
    hand_options
)

#-------------------------------pose landmarker--------------------------------
pose_model_path = "pose_landmarker.task"

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=pose_model_path
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=pose_callback,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)

LIMB_INDICES = {
    "left_wrist": 16,
    "right_wrist": 15,
}

# 4. Open Webcam Capture
cap = cv2.VideoCapture(0)


def body_part_to_coords(part):
    return part.x, part.y, part.z

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # A. Preprocessing: Mirror image horizontally & convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # B. Convert NumPy frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # SEND RESULTS TO BOTH MODELS
    frame_timestamp_ms = int(time.time() * 1000)
    hand_landmarker.detect_async(
        mp_image,
        frame_timestamp_ms
    )

    pose_landmarker.detect_async(
        mp_image,
        frame_timestamp_ms
    )
    # HAND RESULTS
    if latest_hand_result is not None:
        for hand in latest_hand_result.hand_landmarks:
            hand_dict = {0: 'wrist', 8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip'}
            result_dict = {}
            for num, finger_str in hand_dict.items():
                body_part = hand[num]
                result_dict[finger_str] = body_part_to_coords(body_part)
            result_dict['is_chop_frame'] = False
            hand_results.append(result_dict)
            if len(hand_results) > 60:
                hand_results.pop(0)
                old_state = CHOP_STATE
                CHOP_STATE = new_chop_state(hand_results, CHOP_STATE)
                if old_state != CHOP_STATE and CHOP_STATE == CHOPPED:
                    hand_results[-1]['is_chop_frame'] = True
                    chop_amount += 1

    # D. Process Detected Body Landmarks (Right Wrist Index = 16)
    if latest_pose_result is not None:
        if len(latest_pose_result.pose_landmarks) > 0:

            landmarks = latest_pose_result.pose_landmarks[0]

            h, w, _ = frame.shape

            for limb_str, limb_idx in LIMB_INDICES.items():
                limb = landmarks[limb_idx]

                cx = int(limb.x * w)
                cy = int(limb.y * h)
                if display_pose:
                    cv2.circle(
                        frame,
                        (cx, cy),
                        15,
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"{limb_str}: ({cx}, {cy})",
                        (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

    # E. Display Camera Feed
    frame = draw_taskbar(frame, chop_amount)
    cv2.imshow("MediaPipe Pose Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()