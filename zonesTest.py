import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from zones import Zone

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
bank = Zone("BANK", 0.45, 0.25, 0.55, 0.75, (0, 200, 255))

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
    hand_inside = False
    if latest_hand_result is not None:
        for hand in latest_hand_result.hand_landmarks:
            palm = hand[9]  # middle-finger knuckle = roughly the centre of the palm
            if bank.contains(palm.x, palm.y):
                hand_inside = True

    if bank.update(hand_inside, now):
        print("BANK triggered!")

    bank.draw(frame, progress=bank.progress(now))
    cv2.imshow("Zones", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()