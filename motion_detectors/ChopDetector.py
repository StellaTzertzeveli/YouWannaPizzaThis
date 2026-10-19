from motion_detectors.ActionDetector import ActionDetector
class ChopDetector(ActionDetector):

    def __init__(self):
        super().__init__()
        self.CHOPPING_DOWNWARDS = 999
        self.LIFTING_UPWARDS = 998
        self.CHOPPED = 997
        self.mp_dict = {0: 'wrist', 8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip'}

        self.state = self.LIFTING_UPWARDS
        self.chop_count = 0

        self.max_value = 30 #for bar graph

    def get_chop_count(self):
        return self.chop_count

    def get_display_value(self):
        return self.get_chop_count()

    def reset_chop_count(self):
        self.chop_count = 0

    def update_history(self, hand):
        super().update_history(hand)
        self.history[-1]['is_chop_frame'] = False

    def update_state(self, should_print):
        #Here we check if the state of the detector should be changed!
        sum_hand_history = [
            tuple(
                sum(coords[i] for coords in (frame[finger_str] for finger_str in self.mp_dict.values())) / 5
                for i in range(3)
            )
            for frame in self.history
        ]
        wrist_history = [frame['wrist'] for frame in self.history]
        for history in [sum_hand_history, wrist_history]:
            old_x, old_y, old_z = history[-3]
            new_x, new_y, new_z = history[-1]

            dx = new_x - old_x
            dy = new_y - old_y
            dz = new_z - old_z
            speed = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            if should_print:
                print(f'state = {self.state}, speed = {speed}, dy = {dy}, dx = {dx}')

            if self.state == self.LIFTING_UPWARDS:
                CHOP_SPEED = 0.05
                if speed > CHOP_SPEED and abs(dy) > abs(dx) and dy > 0:
                    self.state = self.CHOPPING_DOWNWARDS
                    break

            elif self.state == self.CHOPPING_DOWNWARDS:
                if dy <= 0:
                    self.state = self.CHOPPED
                    self.chop_count += 1
                    latest_result = self.history[-1]
                    latest_result['is_chop_frame'] = True
                    break

            elif self.state == self.CHOPPED:
                chop_history_ints = [i for i in range(len(self.history)) if self.history[i]['is_chop_frame'] == True]
                chop_frames = [history[i] for i in chop_history_ints]
                LIFT_DIST = 0.10  # amount of distance need between the end point of a chop and when it has lifted
                LIFT_SPEED = 0.05
                if len(chop_frames) == 0:
                    # NO CHOP FRAME IN HISTORY
                    self.state = self.LIFTING_UPWARDS
                    break
                else:  # THERE IS A CHOP FRAME
                    chop_frame = chop_frames[-1]
                    _, chop_y, _ = chop_frame
                    if new_y >= chop_y + LIFT_DIST or dy <= LIFT_SPEED:
                        self.state = self.LIFTING_UPWARDS
                        break
            else:
                continue