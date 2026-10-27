import cv2

# Defines a zone, it has a name, coordinates (x1, y1) for the top-left corner and 
# (x2, y2) for the bottom-right corner, and a color for visualization.
class Zone:
    def __init__(self, name, x1, y1, x2, y2, color):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    # When a point is inbetween the coordinates of the edges of the zone, it returns true
    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    # Draws the zone boundaries, if active line thickens, and the name of the zone is displayed
    # in the top left corner of the zone
    def draw(self, frame, active=False):
        h, w, _ = frame.shape
        top_left = (int(self.x1 * w), int(self.y1 * h))
        bottom_right = (int(self.x2 * w), int(self.y2 * h))
        thickness = 6 if active else 2
        cv2.rectangle(frame, top_left, bottom_right, self.color, thickness)
        cv2.putText(frame, self.name, (top_left[0] + 5, top_left[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
    
# Initialize zones with their respective coordinates and colors
if __name__ == "__main__":
    bank = Zone("BANK", 0.45, 0.25, 0.55, 0.75, (0, 200, 255))
    