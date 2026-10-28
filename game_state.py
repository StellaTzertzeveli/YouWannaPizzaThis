import cv2
from zones import INGREDIENTS


# List of ingredient names
INGREDIENT_NAMES = [name for name, color in INGREDIENTS]

# List of surfaces in the game
SURFACES = ["CHOPPING COUNTER", "TOPPING STATION", "OVEN", "BANK", "DELIVERY"]

# Game settings
CHOPS_NEEDED = 4
BAKE_TIME = 5
BURN_TIME = 10

ACCEPTS = {
    "TOPPING STATION": ["DOUGH_SPUN"],
    "OVEN": ["PIZZA"],
    "DELIVERY": ["BAKED_PIZZA"]
    
}

class GameState:
    def __init__(self):
        # Initialize the game state with no held ingredient
        self.holding = None
        # Start the game with nothing on the surfaces
        self.items = {name: None for name in SURFACES}
        # Initialize the number of chops to 0
        self.chops = 0
        # Initialize oven timer to 0
        self.oven_start = 0
        
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
                    if zone_name in ACCEPTS and self.holding not in ACCEPTS[zone_name]:
                        # If the surface is a station that only accepts certain items, and the player is holding an item that is not accepted, do nothing
                        return
                    # Place the held item on the surface and stop holding it
                    self.items[zone_name] = self.holding
                    self.holding = None
                    if zone_name == "CHOPPING COUNTER":
                        self.chops = 0       
                    if zone_name == "TOPPING STATION":
                        self.items[zone_name] = "PIZZA"
                        # a freshly placed ingredient starts unchopped
                # If there is something on the surface and the player isnt holding anything
                elif on_surface is not None and self.holding is None:
                    # Pick up the item from the surface and make the zone stop holding it
                    self.holding = on_surface
                    self.items[zone_name] = None
    # When items are on the chopping counter, the player can chop them. After a certain number of chops, the item becomes chopped.
    def on_chop(self):
        item = self.items["CHOPPING COUNTER"]
        if item in INGREDIENT_NAMES:
            self.chops += 1
            if self.chops >= CHOPS_NEEDED:
                self.items["CHOPPING COUNTER"] = item + "_CHOPPED"
                self.chops = 0
    
    # When the player is holding dough, they can spin it to make it pizza dough.
    def on_spin(self):
        if self.holding == "DOUGH":
            self.holding = "DOUGH_SPUN"
            
    def update(self, now):
        item = self.items["OVEN"]
        if item is None:
            self.oven_start = None
            return
        if self.oven_start is None:
            self.oven_start = now
        time_in_oven = now - self.oven_start
        if item == "PIZZA" and time_in_oven >= BAKE_TIME:
            self.items["OVEN"] = "BAKED_PIZZA"
        elif item == "BAKED_PIZZA" and time_in_oven >= BURN_TIME:
            self.items["OVEN"] = "BURNT_PIZZA"
        
        
    
    def draw(self, frame, zones):
        h, w, _ = frame.shape
        for zone in zones:
            item = self.items.get(zone.name)
            if item is not None:
                cv2.putText(frame, item, (int(zone.x1 * w) + 5, int(zone.y1 * h) + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Holding: {self.holding}", (int(0.58 * w), int(0.74 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if zone.name == "CHOPPING COUNTER" and item in INGREDIENT_NAMES:
                    cv2.putText(frame, f"Chops: {self.chops}/{CHOPS_NEEDED}", (int(zone.x1 * w) + 5, int(zone.y1 * h) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
