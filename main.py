import cv2
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1

#classes
from face_filters.face_class import FaceFilter
from player import Player
from Task_4.tracking import MP_Tracking

players = [
    Player("Chef"),
    Player("Sous Chef")
]

face_filter = FaceFilter()

mp_tracker = MP_Tracking()

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

MIN_CONFIDENCE = 0.9

pre_ids = [0,0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tracks = mp_tracker.get_ids(frame)

    tracked_players = []
    active_track_ids = []

    for i,track in enumerate(tracks):
        if track.time_since_update > 0:
            continue

        if i >= 2:
            continue

        track_id = int(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        bbox = [(x1, y1, x2, y2)]

        if players[i].id == 0:
            players[i].id = track_id
            players[i].bbox = bbox
            pre_ids[i] = track_id

        if track_id != players[i].id:
            #Locate player index
            if pre_ids.count(track_id) > 0:
                loc_of_id = pre_ids.index(track_id)
                if loc_of_id != i:
                    if loc_of_id == 1:
                        players[0].id = track_id
                        players[0].bbox = bbox
                        active_track_ids.append(players[0].name)
                        pre_ids[0] = track_id
                    if loc_of_id == 0:
                        players[1].id = track_id
                        players[1].bbox = bbox
                        active_track_ids.append(players[1].name)
                        pre_ids[1] = track_id
            else:
                players[i].id = track_id
                players[i].bbox = bbox
                active_track_ids.append(players[i].name)

        else:
            players[i].bbox = bbox
            active_track_ids.append(players[i].name)

        tracked_players.append((track_id, (x1, y1, x2, y2)))

    frame = face_filter.process_frame(frame, tracked_players, active_track_ids)

    for track_id, bbox in tracked_players:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        player_index = 0

        for index, player in enumerate(players):
            if player.id == track_id:
                player_index = index

        cv2.putText(
            frame, f"{players[player_index].name} ID: {players[player_index].id} {players[player_index].bbox}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    cv2.imshow("Out", frame)

#for intergration with main main we need to trigger reactions differently than below
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        face_filter.cycle_reaction(0)
    elif key == ord("2"):
        face_filter.cycle_reaction(1)

# When everything done, release the video capture and writing object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
