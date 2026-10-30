#This class detects if someone is making an 'X' with their arms to use the ability.
from motion_detectors.ActionDetector import ActionDetector
import numpy as np
class CrossDetector(ActionDetector):
    def __init__(self):
        super().__init__()
        self.lowest_angle = 10  # widened from 30-120: real crossed arms rarely land in a narrow band
        self.highest_angle = 160
        self.hold_time = 1 #Time that the player has to hold the X before the ability is used.
        self.mp_dict = {13: 'LEFT ELBOW', 14: 'RIGHT ELBOW', 15: 'LEFT WRIST', 16: 'RIGHT WRIST'}
        self.state = False
        self.counter = 0

        self.cross_frames = 0 #The consecutive amount of frames where the player had the X input
        self.max_value = 30  

    def stop_detection(self):
        self.should_detect = False
        self.counter = 0
        self.state = False
        self.cross_frames = 0

    def update_state(self, should_print):
        vectors = self.vector_history[-1]

        for limb_str in self.mp_dict.values():
            if not limb_str in vectors.keys():
                return
        #create a vector for each arm
        left_elb = vectors['LEFT ELBOW']
        right_elb = vectors['RIGHT ELBOW']
        left_wrist = vectors['LEFT WRIST']
        right_wrist = vectors['RIGHT WRIST']

        left_arm = left_wrist - left_elb
        right_arm = right_wrist - right_elb
        angle = self.angle_between(left_arm, right_arm)
        #check the angle
        is_crossed = False
        if self.lowest_angle < angle < self.highest_angle:
            #Horizontal buffer
            if right_wrist[0] > left_wrist[0]:
                vertical_buffer = 0.18  # was 0.1: too strict about wrists being level
                #vertical buffer
                if abs(right_wrist[1]-left_wrist[1]) < vertical_buffer:
                    is_crossed = True

        # shortcut: without this reset, a single stray frame that happens to look X-ish (reaching,
        # stirring, anything) would sit in self.counter forever, so scattered incidental matches
        # across an entire session could add up to a false trigger instead of requiring a real
        # continuous hold (same bug fixed in ThumbsupDetector).
        self.counter = self.counter + 1 if is_crossed else 0

        if self.counter >= self.max_value:
            self.state = True
            self.counter = 0

        if should_print:
            print(f'X has been formed for {self.counter} frames \n Degrees: {angle}')


    def angle_between(self, v, u):
        cos_angle = np.dot(v, u) / (
                np.linalg.norm(v) * np.linalg.norm(u)
        )

        return np.degrees(
            np.arccos(np.clip(cos_angle, -1.0, 1.0))
        )

    def get_counter(self):
        return self.counter

    def get_display_value(self):
        return self.get_counter()
