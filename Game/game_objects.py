"""
Game objects for Pizza Kitchen: Ingredient, Pizza, Order, Oven, Bank.
No game rules in here (who may do what) - that lives in game_state.py.
Run this file directly to check the objects on their own.
"""
import random
from dataclasses import dataclass, field

# =====================================================================
# SETTINGS for the objects
# =====================================================================
TOPPINGS = ["cheese", "pepperoni", "mushrooms", "peppers"]
CHOP_STRIKES = 5             # TBD: strikes needed to chop one ingredient
BAKE_TIME = 5.0              # seconds in the oven until baked
BURN_TIME = 10.0             # seconds in the oven until burnt
BANK_CAPACITY = 4            # TBD: max items on the bank

PIZZA_STAGES = ["dough", "spun", "sauced", "topped", "baking", "baked", "burnt"]


# =====================================================================
@dataclass
class Ingredient:
    type: str
    state: str = "raw"       # "raw" or "chopped"
    strikes: int = 0         # how many chop strikes it has had

    def __post_init__(self):
        if self.type not in TOPPINGS:
            raise ValueError(f"unknown ingredient: {self.type}")

    def __str__(self):
        if self.state == "raw":
            return f"raw {self.type} ({self.strikes}/{CHOP_STRIKES})"
        return f"chopped {self.type}"


@dataclass
class Pizza:
    stage: str = "dough"
    toppings: list = field(default_factory=list)

    def __post_init__(self):
        if self.stage not in PIZZA_STAGES:
            raise ValueError(f"unknown stage: {self.stage}")

    def __str__(self):
        return f"pizza[{self.stage}, toppings={self.toppings}]"


@dataclass
class Order:
    toppings: list           # 1 to 3 different toppings

    def __post_init__(self):
        if not 1 <= len(self.toppings) <= 3:
            raise ValueError("an order has 1 to 3 toppings")
        if len(set(self.toppings)) != len(self.toppings):
            raise ValueError("no duplicate toppings in an order")

    @staticmethod
    def random(rng=None):
        rng = rng or random
        n = rng.randint(1, 3)
        return Order(rng.sample(TOPPINGS, n))      # sample = no duplicates

    def is_satisfied_by(self, pizza):
        """True if the pizza has exactly the toppings the order asks for."""
        return sorted(pizza.toppings) == sorted(self.toppings)


class Oven:
    """Holds one pizza. update(now) makes it baked at 5 s and burnt at 10 s."""

    def __init__(self):
        self.pizza = None
        self.elapsed = 0.0       # oven seconds so far
        self.last_now = None

    def put(self, pizza, now):
        if self.pizza is not None:
            return False
        self.pizza = pizza
        self.elapsed = 0.0
        self.last_now = now
        return True

    def take(self):
        pizza = self.pizza
        self.pizza, self.elapsed, self.last_now = None, 0.0, None
        return pizza

    def update(self, now, speed=1.0):
        """Call every frame with the current time (speed > 1 = Hot Oven).
        Returns a list of what just happened: [], ['baked'], ['burnt'] or ['baked', 'burnt']."""
        happened = []
        if self.pizza is None:
            return happened
        dt = max(0.0, now - self.last_now)
        self.last_now = now
        self.elapsed += dt * speed
        if self.pizza.stage == "baking" and self.elapsed >= BAKE_TIME:
            self.pizza.stage = "baked"
            happened.append("baked")
        if self.elapsed >= BURN_TIME:
            self.pizza.stage = "burnt"
            happened.append("burnt")
        return happened


class Bank:
    """Shared shelf between the two players."""

    def __init__(self, max_items=BANK_CAPACITY):
        self.items = []
        self.max_items = max_items

    def put(self, item):
        """Returns False if the bank is full."""
        if len(self.items) >= self.max_items:
            return False
        self.items.append(item)
        return True

    def take(self, item):
        """Removes that item. Returns False if it is not on the bank."""
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def is_full(self):
        return len(self.items) >= self.max_items


# =====================================================================
# Quick check: run this file on its own
# =====================================================================
if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(200):
        o = Order.random(rng)
        assert 1 <= len(o.toppings) <= 3 and len(set(o.toppings)) == len(o.toppings)
    print("Order: 200 random orders OK (1-3 toppings, no duplicates)")

    p = Pizza(toppings=["cheese", "peppers"])
    assert Order(["peppers", "cheese"]).is_satisfied_by(p)
    assert not Order(["cheese"]).is_satisfied_by(p)
    print("Order.is_satisfied_by OK")

    oven = Oven()
    pz = Pizza(stage="baking")
    oven.put(pz, now=0.0)
    assert oven.update(4.9) == [] and pz.stage == "baking"
    assert oven.update(5.0) == ["baked"] and pz.stage == "baked"
    assert oven.update(9.9) == []
    assert oven.update(10.0) == ["burnt"] and pz.stage == "burnt"
    print("Oven: baked at 5 s, burnt at 10 s OK")

    bank = Bank(max_items=2)
    a, b, c = Ingredient("cheese"), Ingredient("peppers"), Ingredient("mushrooms")
    assert bank.put(a) and bank.put(b) and not bank.put(c)
    assert bank.take(a) and not bank.take(a) and bank.put(c)
    print("Bank: put/take/max items OK")
    print("ALL OBJECT CHECKS PASSED")