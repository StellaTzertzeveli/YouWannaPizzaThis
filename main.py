import cv2
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_filters.face_class import FaceFilter

from player import Player

players = [
    Player("Chef"),
    Player("Sous Chef")
]
face_filter = FaceFilter()


cap = cv2.VideoCapture(0)
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

tracker = DeepSort(
    max_age=30,
    n_init=5,
    max_cosine_distance=0.4,
    embedder=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (MTCNN) - also crops faces to FaceNet's 160x160 input
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    device=device
)

# Face embeddings (FaceNet)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

MIN_CONFIDENCE = 0.7

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    detections = []
    embeddings = []
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(img)

    if boxes is not None:
        keep = probs >= MIN_CONFIDENCE
        boxes, probs = boxes[keep], probs[keep]

    if boxes is not None and len(boxes) > 0:
        # Cropped + standardized faces, shape (N, 3, 160, 160)
        faces = mtcnn.extract(img, boxes, save_path=None)

        with torch.inference_mode():
            embeds = facenet(faces.to(device)).cpu().numpy()

        for box, confidence, embedding in zip(boxes, probs, embeds):
            x1, y1, x2, y2 = box.tolist()

            # DeepSORT expects [left, top, width, height]
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                float(confidence),
                "Face"
            ))

            embeddings.append(embedding)

    # Always update, so tracks age out while no face is visible
    tracks = tracker.update_tracks(
        detections,
        embeds=embeddings
    )

    tracked_players = []
    active_track_ids = []

    for track in tracks:
        if track.time_since_update > 0:
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        tracked_players.append((track_id, (x1, y1, x2, y2)))
        active_track_ids.append(track_id)

    frame = face_filter.process_frame(frame, tracked_players, active_track_ids)

    for track_id, bbox in tracked_players:
        x1, y1, x2, y2 = bbox
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