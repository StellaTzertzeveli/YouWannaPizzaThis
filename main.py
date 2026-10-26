import cv2
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1

from player import Player
from Task_4.tracking import MP_Tracking

players = [
    Player("Chef"),
    Player("Sous Chef")
]

mp_tracking = MP_Tracking()

cap = cv2.VideoCapture(0)
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    tracks = mp_tracking.get_ids(frame)

    for i,track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        if track.time_since_update > 0:
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"ID {track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    cv2.imshow("Out", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the video capture and writing object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()