from motion_detectors.motion_detection import draw_taskbar

class ChopDetector:

    def __init__(self):
        self.CHOPPING_DOWNWARDS = 999
        self.LIFTING_UPWARDS = 998
        self.CHOPPED = 997
        self.hand_dict = {0: 'wrist', 8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip'}

        self.state = self.LIFTING_UPWARDS
        self.chop_count = 0
        self.hand_history = []

    def get_chop_count(self):
        return self.chop_count

    def reset_chop_count(self):
        self.chop_count = 0

    def display_bar_graph(self, frame):
        return draw_taskbar(frame, self.chop_count)

    def update(self, new_hand, should_print=False):
        self.update_history(new_hand)
        if len(self.hand_history) >= 60:
            self.update_state(should_print)

    def update_history(self, hand):
        result_dict = {}
        for num, finger_str in self.hand_dict.items():
            body_part = hand[num]
            result_dict[finger_str] = self.body_part_to_coords(body_part)
        result_dict['is_chop_frame'] = False
        self.hand_history.append(result_dict)
        if len(self.hand_history) > 60:
            self.hand_history.pop(0)

    def body_part_to_coords(self, part):
        return part.x, part.y, part.z

    def update_state(self, should_print):
        #Here we check if the state of the detector should be changed!
        sum_hand_history = [
            tuple(
                sum(coords[i] for coords in (frame[finger_str] for finger_str in self.hand_dict.values())) / 5
                for i in range(3)
            )
            for frame in self.hand_history
        ]
        wrist_history = [frame['wrist'] for frame in self.hand_history]
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
                    latest_hand_result = self.hand_history[-1]
                    latest_hand_result['is_chop_frame'] = True
                    break

            elif self.state == self.CHOPPED:
                chop_history_ints = [i for i in range(len(self.hand_history)) if self.hand_history[i]['is_chop_frame'] == True]
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