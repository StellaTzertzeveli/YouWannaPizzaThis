from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id for "person"


class _Track:
    # Same shape as a deep_sort_realtime Track, so the rest of the game doesn't need to change.
    def __init__(self, track_id, ltrb):
        self.track_id = track_id
        self.time_since_update = 0  # YOLO+ByteTrack only reports currently-visible detections, so always fresh
        self._ltrb = ltrb

    def to_ltrb(self):
        return self._ltrb


class MP_Tracking:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run if not already present

    def get_ids(self, frame):
        result = self.model.track(
            frame, persist=True, classes=[PERSON_CLASS], conf=0.5, imgsz=320,
            tracker="bytetrack.yaml", verbose=False,
        )[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()
        return [_Track(int(tid), tuple(box)) for tid, box in zip(ids, boxes)]
