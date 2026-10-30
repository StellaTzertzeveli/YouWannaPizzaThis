import cv2
from insightface.app import FaceAnalysis
import numpy as np

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)
app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture(0)
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_count = 0

faces = []

prev_face = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count == 0:
        faces = app.get(frame)

        if len(faces) == 0:
            continue

        for i, face in enumerate(faces):
            embedding = face.embedding

            if len(prev_face) == 0:
                prev_face = embedding
                continue

            curr_embedding = np.asarray(embedding, np.float32)
            prev_embedding = np.asarray(prev_face, np.float32)

            cosine_similarity = np.dot(curr_embedding,prev_embedding) / (np.linalg.norm(curr_embedding) * np.linalg.norm(prev_embedding))

            if cosine_similarity <= 0.3:
                print("New Face!")
                print(cosine_similarity)
        frame_count +=1

    frame_count += 1

    cv2.imshow("Out", app.draw_on(frame, faces))

    if frame_count >= 5:
        frame_count = 0


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the video capture and writing object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
