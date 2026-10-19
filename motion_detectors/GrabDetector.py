from motion_detectors.ActionDetector import ActionDetector
import numpy as np
class GrabDetector(ActionDetector):
    def __init__(self, detection_type):
        super().__init__()
        self.mp_dict = {8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip'}

        self.state = False

        self.high_threshold = 0.10 #threshold for grabbing
        self.low_threshold = 0.20 #threshold for letting go

        self.counter = 0

        self.max_value = 60 #for bar graph
        if not detection_type in ['grab', 'loose']:
            raise ValueError(f'Unknown detection type {detection_type} in GrabDetector, should be grabbing or loose')
        self.detection_type = detection_type

    def update_state(self, should_print):
        vectors = self.vector_history[-1]

        for limb_str in self.mp_dict.values():
            if not limb_str in vectors.keys():
                return
        # create a vector for each arm
        fingers = [vectors[limb_str] for limb_str in self.mp_dict.values()]

        #if distance between any two fingers is too big, no grabbing happens
        max_dist = max( [np.linalg.norm(u-v) for u in fingers for v in fingers] )
        if self.detection_type == 'grab':
            detection_bool =  max_dist < self.high_threshold
        else: #loose
            detection_bool = max_dist > self.low_threshold

        if not detection_bool:
            self.counter = 0
            return

        self.counter += 1
        if self.counter > self.max_value:
            self.state = True
            self.counter = 0


        if should_print:
            print(f'{self.detection_type} is {self.state}')

    def reset_state(self):
        self.counter = 0
        self.state = False

    def get_state(self):
        return self.state

    def get_display_value(self):
        return self.counter