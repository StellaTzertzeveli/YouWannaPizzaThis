import cv2
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1

class MP_Tracking:
    def __init__(self):    
        self.tracker = DeepSort(
            max_age=30,
            n_init=5,
            max_cosine_distance=0.4,
            embedder=None
        )

        #sanity chack for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Bounding Box and Confidence
        self.mtcnn = MTCNN(
            image_size=160, #Crops to 160x160 input
            margin=20,
            keep_all=True,
            device=self.device
        )

        # Face embeddings
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        
        self.min_confidence = 0.9 #Minimum confidece for a face to be detected
        
    def get_ids(self,frame):
        detections = []
        embeddings = []

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img)

        if boxes is not None:
            keep = probs >= self.min_confidence
            boxes, probs = boxes[keep], probs[keep]

        if boxes is not None and len(boxes) > 0:
            # Cropped + standardized faces, shape (N, 3, 160, 160)
            faces = self.mtcnn.extract(img, boxes, save_path=None)

            with torch.inference_mode():
                embeds = self.facenet(faces.to(self.device)).cpu().numpy()

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
        tracks = self.tracker.update_tracks(
            detections,
            embeds=embeddings
        )
        
        return tracks