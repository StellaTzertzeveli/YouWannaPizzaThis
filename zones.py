import cv2

# Defines a zone, it has a name, coordinates (x1, y1) for the top-left corner and 
# (x2, y2) for the bottom-right corner, and a color for visualization.
class Zone:
    def __init__(self, name, x1, y1, x2, y2, color, hold_time=1.0):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.hold_time = hold_time
        self.enter_time = None
        self.fired = False

    # When a point is inbetween the coordinates of the edges of the zone, it returns true
    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    # Updates the state of the zone based on whether a hand is inside it and the current time
    def update(self, inside, now):
        # If the hand is not inside the zone, reset the enter time and fired state
        if not inside:
            self.enter_time = None
            self.fired = False
            return False
        # If the hand is inside the zone, check if it has been there long enough to fire
        if self.enter_time is None:
            self.enter_time = now
        # If the hand has been inside the zone for longer than the hold time 
        # and the event didn't fire yet, fire the event
        if not self.fired and (now - self.enter_time) >= self.hold_time:
            self.fired = True
            return True
        # If the hand is inside the zone but hasn't been there long enough, return False
        return False
    
    # Returns the progress of how long the hand has been inside the zone
    # relative to the hold time
    def progress(self, now):
        if self.enter_time is None:
            return 0.0
        # If the hand has been inside the zone for longer than the hold time, return 1.0
        return min((now-self.enter_time)/self.hold_time, 1.0)

    # Draws the zone boundaries, if active line thickens, and the name of the zone is displayed
    # in the top left corner of the zone
    def draw(self, frame, progress=0.0):
        h, w, _ = frame.shape
        x1, y1 = int(self.x1 * w), int(self.y1 * h)
        x2, y2 = int(self.x2 * w), int(self.y2 * h)
        thickness = 6 if progress > 0 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, thickness)
        cv2.putText(frame, self.name, (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        if progress > 0:
            bar_end = x1 + int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 - 15), (bar_end, y2), self.color, -1)
            
# Initialize zones with their respective coordinates and colors
if __name__ == "__main__":
    bank = Zone("BANK", 0.45, 0.25, 0.55, 0.75, (0, 200, 255))
    
    