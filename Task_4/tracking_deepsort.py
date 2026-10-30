import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

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

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)
app.prepare(ctx_id=0, det_size=(320, 320))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    detections = []
    embeddings = []

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.tolist()
        confidence = float(face.det_score)

        # DeepSORT expects [left, top, width, height]
        detections.append((
            [x1, y1, x2 - x1, y2 - y1],
            confidence,
            "Face"
        ))

        embeddings.append(face.embedding)

    if detections:
        tracks = tracker.update_tracks(
            detections,
            embeds=embeddings
        )
    else:
        tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > 0:
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        #Check assigned IDs

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"ID {track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    cv2.imshow("Out",frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the video capture and writing object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
