import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

class MP_Tracking:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=5,
            max_cosine_distance=0.4,
            embedder=None
        )

        self.face_recog = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]    
        )
        
        self.face_recog.prepare(ctx_id=0, det_size=(320, 320))
        
    def get_ids(self,frame):
        detections = []
        embeddings = []
        
        ids = []
        
        faces = self.face_recog.get(frame)
        
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
            tracks = self.tracker.update_tracks(
                detections,
                embeds=embeddings
            )
        else:
            tracks = []
        
        for i,track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            if track.time_since_update > 0:
                continue

            if i >= 2:
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            
            ids.append(([x1,y1,x2,y2], int(track_id)))
            
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            
        return ids