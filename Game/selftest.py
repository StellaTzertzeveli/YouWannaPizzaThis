"""Self test: checks the game rules without a camera.

    python selftest.py         (or: python main.py --selftest)

Plays 4 pizzas through the SAME path the camera uses:
Command -> to_rule_command -> GameState.apply. Run it after every change.
"""
from commands import (CHEF, SOUS_CHEF, CHEF_ID, SOUS_CHEF_ID, Command, GRAB, PLACE,
                      CHOP_STRIKE, SPIN_DONE, SPRINKLE_DONE, OVEN_IN, OVEN_OUT,
                      PASS_TO_BANK, TAKE_FROM_BANK, DELIVER, ABILITY_1, ABILITY_2,
                      CHEESE, PEPPERONI, DOUGH, PIZZA)
from zones import (zone_at, DwellTimer, HOLD_SECONDS, BOX_CHEESE, BOX_DOUGH, BANK,
                   OVEN, TOPPING_STATION)
from game_objects import Ingredient
from game_state import GameState
from players import (to_rule_command, zone_to_command, hold_seconds_for,
                     SPRINKLE_SECONDS)
from feedback import hand_position


# =====================================================================
# SELF TEST   (python main.py --selftest, no camera needed)
# Plays 4 pizzas through the SAME path the camera uses:
# Command -> to_rule_command -> GameState.apply
# =====================================================================
def selftest():
    # "assert condition" stops the program with an AssertionError if the condition
    # is False. Every assert below is one small check of how the game must behave;
    # if they all pass, the rules work. Run it after every change.

    # Zones and dwell timer
    assert zone_at(200, 50) == BOX_CHEESE
    assert zone_at(640, 300) == BANK
    assert zone_at(640, 50) is None
    timer = DwellTimer()
    assert timer.update(BOX_CHEESE, 0.0) is None
    assert timer.update(BOX_CHEESE, 0.5) is None
    assert timer.update(BOX_CHEESE, 1.0) == BOX_CHEESE
    assert timer.update(BOX_CHEESE, 2.0) is None        # still there: no second grab
    timer.update(None, 2.1)                             # hand leaves...
    timer.update(BOX_CHEESE, 2.2)                       # ...and comes back
    assert timer.update(BOX_CHEESE, 3.2) == BOX_CHEESE
    print("Zones + DwellTimer OK")

    # Palm = a bit past the wrist, along the forearm
    arm = {"keypoints": {"right_elbow": (400, 400, 0.9), "right_wrist": (400, 300, 0.9),
                         "left_wrist": (100, 100, 0.1)}}            # left wrist unsure
    assert hand_position(arm, "right_wrist") == (400, 275)
    assert hand_position(arm, "left_wrist") is None
    print("Hand position OK")

    # The bridge
    rule = to_rule_command(Command(player_id=SOUS_CHEF_ID, role=SOUS_CHEF, action=GRAB, item=CHEESE))
    assert rule.action == "grab_raw" and rule.arg == CHEESE
    rule = to_rule_command(Command(player_id=CHEF_ID, role=CHEF, action=PASS_TO_BANK, item=PIZZA))
    assert rule.action == "place_on_bank"
    rule = to_rule_command(Command(player_id=CHEF_ID, role=CHEF, action=ABILITY_2))
    assert rule.action == "ability" and rule.arg == "stubborn"
    print("Bridge (Command -> RuleCommand) OK")

    # Zone -> command depends on the situation
    state = GameState(seed=1)
    sous_player = {"player_id": SOUS_CHEF_ID, "role": SOUS_CHEF}
    chef_player = {"player_id": CHEF_ID, "role": CHEF}
    assert zone_to_command(sous_player, BANK, state, 0.0).action == TAKE_FROM_BANK
    state.sous.holding = Ingredient(CHEESE, state="chopped")
    assert zone_to_command(sous_player, BANK, state, 0.0).action == PASS_TO_BANK
    assert zone_to_command(chef_player, BOX_CHEESE, state, 0.0) is None   # Chef can't use boxes
    assert zone_to_command(chef_player, OVEN, state, 0.0).action == OVEN_IN
    assert zone_to_command(chef_player, BOX_DOUGH, state, 0.0).item == DOUGH
    assert zone_to_command(sous_player, BOX_DOUGH, state, 0.0) is None    # dough is Chef-only
    assert zone_to_command(chef_player, TOPPING_STATION, state, 0.0) is None   # empty hands: nothing
    state.chef.holding = state.pizza                                       # holding the dough
    assert zone_to_command(chef_player, TOPPING_STATION, state, 0.0).action == PLACE
    state.chef.holding = Ingredient(PEPPERONI, state="chopped")            # holding a topping
    assert zone_to_command(chef_player, TOPPING_STATION, state, 0.0).action == SPRINKLE_DONE
    assert hold_seconds_for(chef_player, TOPPING_STATION, state) == SPRINKLE_SECONDS
    assert zone_to_command(chef_player, BANK, state, 0.0).action == PASS_TO_BANK
    state.chef.holding = None
    assert hold_seconds_for(chef_player, TOPPING_STATION, state) == HOLD_SECONDS

    # A 2 s hold does not fire after 1 s, and the hold bar is half full
    timer = DwellTimer()
    timer.update(TOPPING_STATION, 0.0, 2.0)
    assert timer.update(TOPPING_STATION, 1.0, 2.0) is None
    assert abs(timer.progress(1.0) - 0.5) < 1e-9
    assert timer.update(TOPPING_STATION, 2.0, 2.0) == TOPPING_STATION
    print("Zone -> command OK")

    # Dough box: the Chef must grab the dough before spinning or placing it
    state = GameState(seed=1)

    def chef_try(action, item=None):
        return state.apply(to_rule_command(
            Command(player_id=CHEF_ID, role=CHEF, action=action, item=item, timestamp=0.0)))

    assert state.dough_in_box()
    assert not chef_try(SPIN_DONE)                      # no dough in hands yet
    assert not chef_try(PLACE, DOUGH)
    assert chef_try(GRAB, DOUGH) and not state.dough_in_box()
    assert not chef_try(GRAB, DOUGH)                    # only one dough per order
    assert chef_try(SPIN_DONE) and chef_try(PLACE, DOUGH)
    assert state.chef.holding is None and state.pizza.stage == "sauced"
    print("Dough box OK")

    # Chef can put a topping back on the bank, and take it again
    state.chef.holding = Ingredient(state.order.toppings[0], state="chopped")
    assert chef_try(PASS_TO_BANK, state.chef.holding.type)
    assert state.chef.holding is None and len(state.bank.items) == 1
    assert chef_try(TAKE_FROM_BANK) and isinstance(state.chef.holding, Ingredient)
    print("Chef bank (put back / take) OK")

    # Full game: 3 pizzas delivered, 1 burnt, every ability used once.
    # Instead of the real clock we use a fake one (clock[0]) that we move forward
    # ourselves, so a test of "60 seconds of play" runs in a fraction of a second.
    state = GameState(seed=1)
    clock = [0.0]          # a list so the inner functions can change it
    log = []

    def do(player_id, role, action, item=None):
        clock[0] += 0.2
        command = Command(player_id=player_id, role=role, action=action,
                          item=item, timestamp=clock[0])
        ok = state.apply(to_rule_command(command))
        assert ok, f"{action} refused: {state.last_message}"

    def sous(action, item=None):
        do(SOUS_CHEF_ID, SOUS_CHEF, action, item)

    def chef(action, item=None):
        do(CHEF_ID, CHEF, action, item)

    def wait(seconds):
        clock[0] += seconds
        log.extend(state.update(clock[0]))

    def build_pizza():
        for topping in list(state.order.toppings):
            sous(GRAB, topping)
            while state.sous.holding.state != "chopped":
                sous(CHOP_STRIKE)
            sous(PASS_TO_BANK, topping)
        chef(GRAB, DOUGH)
        chef(SPIN_DONE)
        chef(PLACE, DOUGH)
        for _ in list(state.order.toppings):
            chef(TAKE_FROM_BANK)
            chef(SPRINKLE_DONE)
        assert state.pizza.stage == "topped"

    def bake_and_deliver():
        chef(OVEN_IN); wait(6)
        chef(OVEN_OUT); chef(PASS_TO_BANK, PIZZA)
        sous(TAKE_FROM_BANK, PIZZA); sous(DELIVER, PIZZA)

    state.update(0.0)
    sous(ABILITY_1)                                     # Pizza 1: Knife Frenzy
    build_pizza(); bake_and_deliver()
    build_pizza(); chef(ABILITY_1)                      # Pizza 2: Hot Oven...
    chef(OVEN_IN); wait(3); sous(ABILITY_2)             # ...then Rush Delivery
    build_pizza(); chef(OVEN_IN); wait(11)              # Pizza 3: burns
    chef(ABILITY_2)                                     # Pizza 4: Stubborn, new order
    build_pizza(); bake_and_deliver()

    log.extend(state.update(clock[0]))
    print(state)
    assert state.score == 3, state.score
    assert "pizza_burnt" in log
    print("\nSELFTEST PASSED: 3 delivered, 1 burnt, all 4 abilities used.")


if __name__ == "__main__":
    selftest()
