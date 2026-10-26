#This class detects if someone is making an 'X' with their arms to use the ability.
from motion_detectors.ActionDetector import ActionDetector
import numpy as np
class CrossDetector(ActionDetector):
    def __init__(self):
        super().__init__()
        self.lowest_angle = 30
        self.highest_angle = 90 + self.lowest_angle
        self.hold_time = 2 #Time that the player has to hold the X before the ability is used.
        self.mp_dict = {13: 'LEFT ELBOW', 14: 'RIGHT ELBOW', 15: 'LEFT WRIST', 16: 'RIGHT WRIST'}
        self.state = False
        self.counter = 0

        self.cross_frames = 0 #The consecutive amount of frames where the player had the X input
        self.max_value = 60

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
        if self.lowest_angle < angle < self.highest_angle:
            #Horizontal buffer
            if right_wrist[0] > left_wrist[0]:
                vertical_buffer = 0.1
                #vertical buffer
                if abs(right_wrist[1]-left_wrist[1]) < vertical_buffer:
                    self.counter += 1

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