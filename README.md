some general info should go hereee


About face filters and player identification:


**main.py finds and tracks people**
-MTCNN returns face boxes and confidences for each detected face. Boxes below 0.7 are dropped.
-FaceNet (InceptionResnetV1) turns each face crop into a 512-number embedding. The same person gives similar embeddings, so it works as a fingerprint.
-DeepSORT takes all detections and embeddings and predicts each track’s motion with a Kalman filter,
-and matches new detections to tracks by appearance (cosine distance under 0.4) and overlap
-if 30 frames pass without a match, the track is deleted
-the result is tracked_players = [(track_id, bbox), ...], passed to process_frame

-each face mediapipe detects is matched to the nearest DeepSort bounding box
-each person gets a hat/ is assigned to be chef/ sous-chef depending on the ID.
-the earliest assigned ID is the chef (because it's more senior) and the second one is the sous-chef



The **FaceFilter** class:
-finds facial landmarks using MediaPipe, 
-works out how the head is rotated in 3D, 
-renders a 3D hat for the chef and rat for the sous-chef, 
-draws 2D reaction PNGs.