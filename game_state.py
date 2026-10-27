import cv2
from zones import INGREDIENTS

# List of ingredient names
INGREDIENT_NAMES = [name for name, color in INGREDIENTS]

# List of surfaces in the game
SURFACES = ["CHOPPING COUNTER", "TOPPING STATION", "OVEN", "BANK", "DELIVERY"]

class GameState:
    def __init__(self):
        # Initialize the game state with no held ingredient
        self.holding = None
        # Start the game with nothing on the surfaces
        self.items = {name: None for name in SURFACES}
        
    def on_hold(self, zone_name):
        # If the player is hovering over a zone that is either an ingredient or dough
        if zone_name in INGREDIENT_NAMES or zone_name == "DOUGH":
            # If the player is not already holding something
            if self.holding == None:
                # Let the player hold the ingredient or dough
                self.holding = zone_name
            # If the player is hovering over trash
        elif zone_name == "TRASH":
                # Let player throw awy the item
                self.holding = None
            # If the player is hovering over a surface
        elif zone_name in SURFACES:
                on_surface = self.items[zone_name]
                # If there is nothing on the surface and the player is holding something
                if on_surface is None and self.holding is not None:
                    # Place the held item on the surface and stop holding it
                    self.items[zone_name] = self.holding
                    self.holding = None
                # If there is something on the surface and the player isnt holding anything
                elif on_surface is not None and self.holding is None:
                    # Pick up the item from the surface and make the zone stop holding it
                    self.holding = on_surface
                    self.items[zone_name] = None
                    
    def draw(self, frame, zones):
        h, w, _ = frame.shape
        for zone in zones:
            item = self.items.get(zone.name)
            if item is not None:
                cv2.putText(frame, item, (int(zone.x1 * w) + 5, int(zone.y1 * h) + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Holding: {self.holding}", (int(0.58 * w), int(0.74 * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
