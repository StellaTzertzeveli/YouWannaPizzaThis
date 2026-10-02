import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]    
)
app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture("YouWannaPizzaThis\\Task_4\\Singluar_Face_Tracking.mp4")
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_count = 0

faces = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:  
        break
        
    if frame_count == 0:
        faces = app.get(frame)
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