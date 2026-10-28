import math

import cv2
from matplotlib import text
from zones import *
import random

import zones

# List of ingredient names
INGREDIENT_NAMES = [name for name, color in INGREDIENTS]

# List of chopped ingredient names
CHOPPED_NAMES = [name + "_CHOPPED" for name in INGREDIENT_NAMES]

# List of bank slots
BANK_NAMES = [f"BANK {i + 1}" for i in range(BANK_SLOTS)]

# List of surfaces in the game
SURFACES = ["CHOPPING COUNTER", "TOPPING STATION", "OVEN", "DELIVERY"] + BANK_NAMES

# Gameplay settings
CHOPS_NEEDED = 4
BAKE_TIME = 5
BURN_TIME = 10
CORRECT_BONUS = 15
WRONG_PENALTY = 5
BURNT_PENALTY = 10

# Game Time
GAME_TIME = 90

ACCEPTS = {
    "TOPPING STATION": ["DOUGH_SPUN"],
    "OVEN": ["PIZZA"],
    "DELIVERY": ["PIZZA_BAKED","PIZZA_BURNT"]
    
}

# Pizza class to keep track of pizza stage + ingredients used
class Pizza:
    def __init__(self):
        self.stage = "PIZZA"
        self.toppings = []
    def __str__(self):
        return " ".join([self.stage] + self.toppings)

def kind(item):
    if isinstance(item, Pizza):
        return item.stage
    return item

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
        # Initialize the order list to an empty list
        self.order = []
        self.new_order()
        # Initialize the game start time
        self.time_left = GAME_TIME
        self.timer_running = False
        self.last_now = None
        
    def new_order(self):
        amount = random.randint(1, 3)
        self.order = random.sample(INGREDIENT_NAMES, amount)
        
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
                if zone_name == "TOPPING STATION" and isinstance(on_surface, Pizza) and self.holding in CHOPPED_NAMES:
                    topping = self.holding.replace("_CHOPPED", "")
                    if topping not in on_surface.toppings:
                        on_surface.toppings.append(topping)
                        self.holding = None
                    return
                # If there is nothing on the surface and the player is holding something
                if on_surface is None and self.holding is not None:
                    if zone_name in ACCEPTS and kind(self.holding) not in ACCEPTS[zone_name]:   
                        return
                    # Place the held item on the surface and stop holding it
                    self.items[zone_name] = self.holding
                    self.holding = None
                    if zone_name == "CHOPPING COUNTER":
                        self.chops = 0       
                    if zone_name == "TOPPING STATION":
                        self.items[zone_name] = Pizza()     
                    if zone_name == "DELIVERY":
                        self.deliver(self.items[zone_name])
                        self.items[zone_name] = None
                # If there is something on the surface and the player isnt holding anything
                elif on_surface is not None and self.holding is None:
                    # Pick up the item from the surface and make the zone stop holding it
                    self.holding = on_surface
                    self.items[zone_name] = None
                    
                    
    def deliver(self, pizza):
        if pizza.stage == "PIZZA_BURNT":
            self.time_left -= BURNT_PENALTY
        elif sorted(pizza.toppings) == sorted(self.order):
            self.time_left += CORRECT_BONUS
        else:
            self.time_left -= WRONG_PENALTY
        self.time_left = max(self.time_left, 0)
        self.new_order()
                    
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
    
    def toggle_timer(self):
        self.timer_running = not self.timer_running
    
    # Logic for how long the pizza has to stay in the oven
    def update(self, now):
        # Called every frame: things that happen over time, without a hand doing anything
        if self.timer_running and self.last_now is not None:
            self.time_left = max(self.time_left - (now - self.last_now), 0)
        self.last_now = now

        pizza = self.items["OVEN"]
        if pizza is None:
            self.oven_start = None  # oven empty: stop the clock
            return
        if self.oven_start is None:
            self.oven_start = now  # pizza just went in: start the clock
        time_in_oven = now - self.oven_start
        if pizza.stage == "PIZZA" and time_in_oven >= BAKE_TIME:
            pizza.stage = "PIZZA_BAKED"
        elif pizza.stage == "PIZZA_BAKED" and time_in_oven >= BURN_TIME:
            pizza.stage = "PIZZA_BURNT"

    def draw(self, frame, zones):
        h, w, _ = frame.shape
        for zone in zones:
            item = self.items.get(zone.name)
            if item is not None:
                cv2.putText(frame, str(item), (int(zone.x1 * w) + 5, int(zone.y1 * h) + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if zone.name == "CHOPPING COUNTER" and item in INGREDIENT_NAMES:
                    cv2.putText(frame, f"Chops: {self.chops}/{CHOPS_NEEDED}", (int(zone.x1 * w) + 5, int(zone.y1 * h) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Recipe: one line per topping, inside the recipe box
        for i, topping in enumerate(self.order):
            cv2.putText(frame, topping, (int(0.58 * w) + 5, int(0.02 * h) + 45 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Seconds left, centred in the timer circle
        text = str(math.ceil(self.time_left))
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cx, cy = int(0.51 * w), int(0.12 * h)
        cv2.putText(frame, text, (cx - text_w // 2, cy + text_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f"Holding: {self.holding}", (int(0.58 * w), int(0.74 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)