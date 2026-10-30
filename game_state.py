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
BAKE_TIME = 10
BURN_TIME = 25
CORRECT_BONUS = 30
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

# Player Abillities

PLAYER_ABILITIES = {
    "SOUS": ["KNIFE_FRENZY", "RUSH_DELIVERY"],
    "CHEF": ["HOT_OVEN", "STUBBORN"],
}

ABILITY_COOLDOWN = {
    "KNIFE_FRENZY": 40,
    "RUSH_DELIVERY": 30,
    "HOT_OVEN": 40,
    "STUBBORN": 30,
}

# Hot Oven and Knife Frenzy
ABILITY_DURATION = 20

FRENZY_CHOPS = 3
HOT_OVEN_SPEED = 3

# Game Time
GAME_TIME = 90

# Surfaces that only accept certain items. Surfaces not listed here accept anything.
ACCEPTS = {
    "TOPPING STATION": ["DOUGH_SPUN", "PIZZA"],
    "OVEN": ["PIZZA"],
    "DELIVERY": ["PIZZA_BAKED", "PIZZA_BURNT"],
    "CHOPPING COUNTER": INGREDIENT_NAMES,
}

# Which role a zone belongs to. A zone left out (the bank, trash's... no, trash IS listed) is open to both players.
ZONE_ROLES = {
    "DOUGH": "CHEF",
    "OVEN": "CHEF",
    "TOPPING STATION": "CHEF",
    "CHOPPING COUNTER": "SOUS",
    "TRASH": "SOUS",
    "DELIVERY": "SOUS",
}
for _name in INGREDIENT_NAMES:
    ZONE_ROLES[_name] = "SOUS"


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


def image_name(item, chops=0, spun=0):
    # Which picture shows this item, e.g. "CHEESE" with 2 chops -> "cheese_cut_2"
    if item in INGREDIENT_NAMES:
        if chops > 0:
            return f"{item.lower()}_cut_{chops}"
        return f"{item.lower()}_raw"
    if item == "DOUGH":
        if spun==1:
            return f"{item.lower()}_spun"
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
        # Initialize the game state with no held ingredient. Each player has their own hand,
        # so this is one slot per role, not one shared slot for the whole game.
        self.holding = {role: None for role in PLAYER_ABILITIES}
        # Start the game with nothing on the surfaces
        self.items = {name: None for name in SURFACES}
        # Initialize the number of chops to 0
        self.chops = 0
        # Initialize dough as not spun
        self.spun = 0
        # How long the pizza has been in the oven (0 = oven empty)
        self.oven_time = 0.0
        # When each ability was last used (None = never), for cooldowns and durations
        self.last_used = {name: None for name in ABILITY_COOLDOWN}
        # Initialize the game timer, running from the start
        self.time_left = GAME_TIME
        self.timer_running = True
        self.last_now = None
        self.score = 0  # +1 per pizza delivered that exactly matches the order
        self.game_over = False
        # Initialize the order list and pick the first recipe
        self.order = []
        self.new_order()


    def new_order(self):
        amount = random.randint(1, 3)
        self.order = random.sample(INGREDIENT_NAMES, amount)

    def on_hold(self, zone_name, role=None):
        if role is None:
            return  # no tracked player to credit this trigger to, so there's no hand to hold anything
        # Zones restricted to one role reject the other player (bank/unlisted zones stay open to both)
        required = ZONE_ROLES.get(zone_name)
        if required is not None and role != required:
            return
        holding = self.holding[role]
        # If the player is hovering over a zone that is either an ingredient or dough
        if zone_name in INGREDIENT_NAMES or zone_name == "DOUGH":
            # If the player is not already holding something
            if holding is None:
                # Let the player hold the ingredient or dough
                self.holding[role] = zone_name
        # If the player is hovering over trash
        elif zone_name == "TRASH":
            # Let player throw away the item
            self.holding[role] = None
        # If the player is hovering over a surface
        elif zone_name in SURFACES:
            on_surface = self.items[zone_name]
            # Chopped ingredient over the pizza on the topping station: add it as a topping
            if zone_name == "TOPPING STATION" and isinstance(on_surface, Pizza) and holding in CHOPPED_NAMES:
                topping = holding.replace("_CHOPPED", "")
                if topping not in on_surface.toppings:
                    on_surface.toppings.append(topping)
                    self.holding[role] = None
                return
            # If there is nothing on the surface and the player is holding something
            if on_surface is None and holding is not None:
                if zone_name in ACCEPTS and kind(holding) not in ACCEPTS[zone_name]:
                    return  # not allowed here: nothing happens
                # Place the held item on the surface and stop holding it
                self.items[zone_name] = holding
                self.holding[role] = None
                if zone_name == "CHOPPING COUNTER":
                    self.chops = 0  # a freshly placed ingredient starts unchopped
                if zone_name == "TOPPING STATION":
                    self.items[zone_name] = Pizza()  # sauce is added instantly
                if zone_name == "DELIVERY":
                    self.deliver(self.items[zone_name])
                    self.items[zone_name] = None  # the pizza is gone: served
            # If there is something on the surface and the player isnt holding anything
            elif on_surface is not None and holding is None:
                # Pick up the item from the surface and make the zone stop holding it
                self.holding[role] = on_surface
                self.items[zone_name] = None

    def deliver(self, pizza):
        if pizza.stage == "PIZZA_BURNT":
            self.time_left -= BURNT_PENALTY
        elif sorted(pizza.toppings) == sorted(self.order):
            self.time_left += CORRECT_BONUS
            self.score += 1
        else:
            self.time_left -= WRONG_PENALTY
        self.time_left = max(self.time_left, 0)
        self.new_order()

    # When items are on the chopping counter, the player can chop them. After a certain number of chops, the item becomes chopped.
    def on_chop(self):
        item = self.items["CHOPPING COUNTER"]
        if item in INGREDIENT_NAMES:
            self.chops += FRENZY_CHOPS if self.ability_active("KNIFE_FRENZY") else 1
            if self.chops >= CHOPS_NEEDED:
                self.items["CHOPPING COUNTER"] = item + "_CHOPPED"
                self.chops = 0

    # When the player is holding dough, they can spin it to make it pizza dough.
    def on_spin(self, role="CHEF"):  # only the chef ever holds dough (DOUGH is a CHEF-only zone)
        if self.holding[role] == "DOUGH":
            self.holding[role] = "DOUGH_SPUN"

    # Seeing if the ability cooldown is over
    def ability_ready(self, name):
        last = self.last_used[name]
        # shortcut: was "and", which crashed on a never-used ability (last is None) instead of allowing it; needs "or" so "never used" counts as ready
        return last is None or (self.last_now or 0) - last >= ABILITY_COOLDOWN[name]

    # Whether the ability's effect is currently active (its short duration, not the whole cooldown)
    def ability_active(self, name):
        last = self.last_used[name]
        # shortcut: was comparing against ABILITY_COOLDOWN[name], which kept the effect "active" for the entire 10-20s cooldown instead of just ABILITY_DURATION seconds
        return last is not None and (self.last_now or 0) - last < ABILITY_DURATION


    def on_ability(self, player, number):
        name = PLAYER_ABILITIES[player][number-1]
        if not self.ability_ready(name):
            return False
        if name == "RUSH_DELIVERY":
            if not self.rush_delivery(player):
                return False
        if name == "STUBBORN":
            self.new_order()
        self.last_used[name] = self.last_now or 0
        return True

    def rush_delivery(self, role):
        if kind(self.holding[role]) == "PIZZA_BAKED":
            pizza, self.holding[role] = self.holding[role], None
            self.deliver(pizza)
            return True
        for name in SURFACES:
            if kind(self.items[name]) == "PIZZA_BAKED":
                pizza, self.items[name] = self.items[name], None
                self.deliver(pizza)
                return True
        return False


    def toggle_timer(self):
        if not self.game_over:
            self.timer_running = not self.timer_running

    # Called every frame: things that happen over time, without a hand doing anything
    def update(self, now):
                # Seconds since the previous frame
        dt = 0 if self.last_now is None else now - self.last_now
        self.last_now = now

        if self.game_over:
            return

        # Game timer: count down only while running
        if self.timer_running:
            self.time_left = max(self.time_left - dt, 0)
            if self.time_left <= 0:
                self.timer_running = False
                self.game_over = True
                return

        # Oven: bake after BAKE_TIME, burn after BURN_TIME
        pizza = self.items["OVEN"]
        if pizza is None:
            self.oven_time = 0.0  # oven empty: reset the clock
            return
        self.oven_time += dt * (HOT_OVEN_SPEED if self.ability_active("HOT_OVEN") else 1)
        if pizza.stage == "PIZZA" and self.oven_time >= BAKE_TIME:
            pizza.stage = "PIZZA_BAKED"
        elif pizza.stage == "PIZZA_BAKED" and self.oven_time >= BURN_TIME:
            pizza.stage = "PIZZA_BURNT"

    def draw_oven_bar(self, frame, zone):
        if self.items["OVEN"] is None:
            return
        h, w, _ = frame.shape
        fill, color = oven_progress(self.oven_time)
        x1, x2 = int(zone.x1 * w) + 5, int(zone.x2 * w) - 5
        y = int((zone.y1 + zone.y2) / 2 * h)
        half = max(4, int(0.015 * h))
        cv2.rectangle(frame, (x1, y - half), (x2, y + half), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y - half), (x1 + int((x2 - x1) * fill), y + half), color, -1)
        cv2.rectangle(frame, (x1, y - half), (x2, y + half), (255, 255, 255), 1)

    def draw(self, frame, zones, palms=(), palm_roles=()):
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
        # Score, just below the timer circle
        score_text = f"Score: {self.score}"
        (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, score_text, (cx - score_w // 2, cy + int(0.07 * h) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Each player's own held item, drawn on their own tracked hand (skipped for a frame if that
        # player's hand briefly drops out of tracking)
        size = int(0.12 * h)
        for role, item in self.holding.items():
            if item is None:
                continue
            palm = next((p for p, r in zip(palms, palm_roles) if r == role), None)
            if palm is None:
                continue
            draw_item(frame, item, int(palm.x * w) - size // 2, int(palm.y * h) - size // 2, size)
