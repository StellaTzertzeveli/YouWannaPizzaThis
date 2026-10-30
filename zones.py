import cv2
from images import draw_fill


BANK_SLOTS = 4
# List of touples containing the name and color of each ingredient zone
INGREDIENTS = [
    ("CHEESE", (0, 220, 255)),
    ("PEPPERONI", (0, 0, 220)),
    ("MUSHROOMS", (140, 180, 210)),
    ("PEPPERS", (0, 200, 0)),
]
    
# Defines a zone, it has a name, coordinates (x1, y1) for the top-left corner and 
# (x2, y2) for the bottom-right corner, and a color for visualization.
class Zone:
    def __init__(self, name, x1, y1, x2, y2, color, hold_time=1.0, sprite=None):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.hold_time = hold_time
        self.enter_time = None
        self.fired = False
        self.sprite = sprite or name.lower().replace(" ", "_")  # background picture, e.g. "chopping_counter"

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
        draw_fill(frame, self.sprite, x1, y1, x2, y2)  # background picture, if there is one
        thickness = 6 if progress > 0 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, thickness)
        # Shrink the text until it fits inside the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        if progress > 0:
            bar_end = x1 + int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 - 15), (bar_end, y2), self.color, -1)
            
def make_zones():
    zones = []

    # Bank: slots stacked from top to bottom in the middle, one item each
    slot_height = (0.98 - 0.24) / BANK_SLOTS
    for i in range(BANK_SLOTS):
        top = 0.24 + i * slot_height
        zones.append(Zone(f"BANK {i + 1}", 0.45, top + 0.01, 0.55, top + slot_height - 0.01, (0, 200, 255)))

    # Four ingredient boxes side by side along the top of the left half
    box_width = 0.45 / len(INGREDIENTS)
    for i, (name, color) in enumerate(INGREDIENTS):
        left = i * box_width
        zones.append(Zone(name, left + 0.01, 0.02, left + box_width - 0.01, 0.20, color, sprite="ingredient_box"))

    # Chopping counter along the bottom of the left half
    zones.append(Zone("CHOPPING COUNTER", 0.01, 0.72, 0.44, 0.98, (60, 120, 180)))

    # Delivery stand and trash on the far left, between the ingredient boxes and the counter
    zones.append(Zone("DELIVERY", 0.01, 0.28, 0.12, 0.48, (200, 80, 160), sprite="delivery"))
    zones.append(Zone("TRASH", 0.01, 0.51, 0.12, 0.70, (150, 150, 150)))

    # Chef's side: a mirror image of the Sous Chef's side
    zones.append(Zone("TOPPING STATION", 0.56, 0.72, 0.99, 0.98, (40, 80, 230)))
    zones.append(Zone("OVEN", 0.88, 0.28, 0.99, 0.48, (0, 90, 255)))
    zones.append(Zone("DOUGH", 0.88, 0.51, 0.99, 0.70, (170, 210, 240), sprite="ingredient_box"))

    return zones
            
def draw_hud(frame):
    # Things that are only shown on screen
    h, w, _ = frame.shape
    # Make art fill the recipe box
    draw_fill(frame, "recipe", int(0.58 * w), int(0.02 * h), int(0.99 * w), int(0.20 * h))
    # Recipe box above the Chef, empty for now
    cv2.rectangle(frame, (int(0.58 * w), int(0.02 * h)), (int(0.99 * w), int(0.20 * h)), (255, 255, 255), 2)
    # Timer above the bank, a white circle for now, to move it more to he left you have to change the x coordinate of the
    # circle center, to move it more down you have to change the y coordinate of the circle center, to make it bigger you have to change the radius of the circle
    cv2.circle(frame, (int(0.51 * w), int(0.12 * h)), int(0.07 * h), (255, 255, 255), -1)
    
# Initialize zones with their respective coordinates and colors
if __name__ == "__main__":
    bank = Zone("BANK", 0.45, 0.25, 0.55, 0.75, (0, 200, 255))
    
    