import math
import random
import cv2
from zones import INGREDIENTS, BANK_SLOTS
from images import draw_image


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
PIZZA_SCALE = 1.6
ITEM_SIZE = 0.1 # Scaled on screen size
TOPPING_SCALE = 0.6

# To make them fit on pizza
TOPPING_OFFSETS = {
    "CHEESE": (0.1, 0.15),
    "PEPPERONI": (0.1, 0.15),
    "MUSHROOMS": (0.1, 0.15),
    "PEPPERS": (0.1, 0.15),
}

# Game Time
GAME_TIME = 90

# Surfaces that only accept certain items. Surfaces not listed here accept anything.
ACCEPTS = {
    "TOPPING STATION": ["DOUGH_SPUN", "PIZZA"],
    "OVEN": ["PIZZA"],
    "DELIVERY": ["PIZZA_BAKED", "PIZZA_BURNT"],
    "CHOPPING COUNTER": INGREDIENT_NAMES,
}


# Pizza class to keep track of pizza stage + ingredients used
class Pizza:
    def __init__(self):
        self.stage = "PIZZA"  # PIZZA -> PIZZA_BAKED -> PIZZA_BURNT
        self.toppings = []    # e.g. ["CHEESE", "PEPPERS"]

    def __str__(self):
        return " ".join([self.stage] + self.toppings)


def kind(item):
    # The name the rules look at: a pizza's stage, or the item itself if it is just a name
    if isinstance(item, Pizza):
        return item.stage
    return item


def image_name(item, chops=0):
    # Which picture shows this item, e.g. "CHEESE" with 2 chops -> "cheese_cut_2"
    if item in INGREDIENT_NAMES:
        if chops > 0:
            return f"{item.lower()}_cut_{chops}"
        return f"{item.lower()}_raw"
    if item in CHOPPED_NAMES:
        return item.replace("_CHOPPED", "").lower() + "_sliced"
    return item.lower()


def draw_item(frame, item, x, y, size, chops=0):
    # Draws an item as a picture in a size x size square; falls back to its name if there is no picture
    if isinstance(item, Pizza):
        big = int(size * PIZZA_SCALE)
        x, y, size = x - (big - size) // 2, y - (big - size) // 2, big  # grow around the same centre
        if draw_image(frame, "pizza", x, y, size):
            t = int(size * TOPPING_SCALE)
            tx, ty = x + (size - t) // 2, y + (size - t) // 2  # centred on the pizza
            for topping in item.toppings:
                dx, dy = TOPPING_OFFSETS.get(topping, (0, 0))
                draw_image(frame, topping.lower() + "_sliced", tx + int(dx * size), ty + int(dy * size), t)
            if item.stage != "PIZZA":
                cv2.putText(frame, item.stage.replace("PIZZA_", ""), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            return
    elif draw_image(frame, image_name(item, chops), x, y, size):
        return
    cv2.putText(frame, str(item), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
def oven_progress(time_in_oven):
    # How full the oven bar is (0 to 1) and its colour (B, G, R): fills up green while baking, then turns from green to red until burnt
    if time_in_oven < BAKE_TIME:
        return time_in_oven / BAKE_TIME, (0, 255, 0)
    if time_in_oven < BURN_TIME:
        p = (time_in_oven - BAKE_TIME) / (BURN_TIME - BAKE_TIME)
        return 1.0, (0, int(255 * (1 - p)), int(255 * p))
    return 1.0, (0, 0, 255)


class GameState:
    def __init__(self):
        # Initialize the game state with no held ingredient
        self.holding = None
        # Start the game with nothing on the surfaces
        self.items = {name: None for name in SURFACES}
        # Initialize the number of chops to 0
        self.chops = 0
        # When the pizza went into the oven (None = oven empty)
        self.oven_start = None
        # Initialize the game timer, paused until the space bar is pressed
        self.time_left = GAME_TIME
        self.timer_running = False
        self.last_now = None
        # Initialize the order list and pick the first recipe
        self.order = []
        self.new_order()

    def new_order(self):
        amount = random.randint(1, 3)
        self.order = random.sample(INGREDIENT_NAMES, amount)

    def on_hold(self, zone_name):
        # If the player is hovering over a zone that is either an ingredient or dough
        if zone_name in INGREDIENT_NAMES or zone_name == "DOUGH":
            # If the player is not already holding something
            if self.holding is None:
                # Let the player hold the ingredient or dough
                self.holding = zone_name
        # If the player is hovering over trash
        elif zone_name == "TRASH":
            # Let player throw away the item
            self.holding = None
        # If the player is hovering over a surface
        elif zone_name in SURFACES:
            on_surface = self.items[zone_name]
            # Chopped ingredient over the pizza on the topping station: add it as a topping
            if zone_name == "TOPPING STATION" and isinstance(on_surface, Pizza) and self.holding in CHOPPED_NAMES:
                topping = self.holding.replace("_CHOPPED", "")
                if topping not in on_surface.toppings:
                    on_surface.toppings.append(topping)
                    self.holding = None
                return
            # If there is nothing on the surface and the player is holding something
            if on_surface is None and self.holding is not None:
                if zone_name in ACCEPTS and kind(self.holding) not in ACCEPTS[zone_name]:
                    return  # not allowed here: nothing happens
                # Place the held item on the surface and stop holding it
                self.items[zone_name] = self.holding
                self.holding = None
                if zone_name == "CHOPPING COUNTER":
                    self.chops = 0  # a freshly placed ingredient starts unchopped
                if zone_name == "TOPPING STATION":
                    self.items[zone_name] = Pizza()  # sauce is added instantly
                if zone_name == "DELIVERY":
                    self.deliver(self.items[zone_name])
                    self.items[zone_name] = None  # the pizza is gone: served
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

    # Called every frame: things that happen over time, without a hand doing anything
    def update(self, now):
        # Game timer: count down only while running
        if self.timer_running and self.last_now is not None:
            self.time_left = max(self.time_left - (now - self.last_now), 0)
        self.last_now = now

        # Oven: bake after BAKE_TIME, burn after BURN_TIME
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
            
    def draw_oven_bar(self, frame, zone):
        if self.items["OVEN"] is None or self.oven_start is None or self.last_now is None:
            return
        h, w, _ = frame.shape
        fill, color = oven_progress(self.last_now - self.oven_start)
        x1, x2 = int(zone.x1 * w) + 5, int(zone.x2 * w) - 5
        y = int((zone.y1 + zone.y2) / 2 * h)
        half = max(4, int(0.015 * h))
        cv2.rectangle(frame, (x1, y - half), (x2, y + half), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y - half), (x1 + int((x2 - x1) * fill), y + half), color, -1)
        cv2.rectangle(frame, (x1, y - half), (x2, y + half), (255, 255, 255), 1)

    def draw(self, frame, zones, palms=()):
        h, w, _ = frame.shape
        for zone in zones:
            item = self.items.get(zone.name)
            if zone.name in INGREDIENT_NAMES or zone.name == "DOUGH":
                item = zone.name  # a box always shows what you can grab from it
            if zone.name == "OVEN":
                self.draw_oven_bar(frame, zone)
                item = None  # the pizza is inside the oven, so it is hidden
            if item is not None:
                if item is Pizza:
                    size = int(ITEM_SIZE * h * PIZZA_SCALE)
                else:
                    size = int(ITEM_SIZE * h)
                cx, cy = int((zone.x1 + zone.x2) / 2 * w), int((zone.y1 + zone.y2) / 2 * h)  # centre of the zone
                chops = self.chops if zone.name == "CHOPPING COUNTER" else 0
                draw_item(frame, item, cx - size // 2, cy - size // 2, size, chops)

        # Recipe: one picture per topping, side by side inside the recipe box
        size = int(0.18 * h) - 35
        for i, topping in enumerate(self.order):
            draw_item(frame, topping, int(0.58 * w) + 5 + i * (size + 5), int(0.02 * h) + 32, size)
        # Seconds left, centred in the timer circle
        text = str(math.ceil(self.time_left))
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cx, cy = int(0.51 * w), int(0.12 * h)
        cv2.putText(frame, text, (cx - text_w // 2, cy + text_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)    
        # The held item on the hand
        if self.holding is not None and len(palms) > 0:
            palm = palms[0] 
            size = int(0.12 * h)
            draw_item(frame, self.holding, int(palm.x * w) - size // 2, int(palm.y * h) - size // 2, size)
