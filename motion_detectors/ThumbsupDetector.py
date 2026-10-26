from motion_detectors.ActionDetector import ActionDetector
import numpy as np
class ThumbsupDetector(ActionDetector):

    def __init__(self):
        super().__init__()
        self.mp_dict = {0: 'wrist', 2: 'thumb_base', 3: 'thumb_knuckle', 4: 'thumb_tip', 8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip', 5: 'index_knuckle', 17: 'pinky_knuckle'}
        self.state = False
        self.count = 0
        self.max_value = 60

        self.dist_threshold_wrist = 0.30 #minimum distance between secondary fingers and the wrist
        self.angle_threshold_thumb = 35

    def stop_detection(self):
        self.should_detect = False
        self.state = False
        self.count = 0

    def update_state(self, should_print):
        vectors = self.vector_history[-1]

        for limb_str in self.mp_dict.values():
            if not limb_str in vectors.keys():
                return
        # create a vector for each arm
        fingers = [vectors[limb_str] for limb_str in self.mp_dict.values()]
        wrist = fingers[0]
        secondary_fingers = fingers[4:8]
        knuckles = fingers[8:10]


        #first we check if the secondary fingers are close to the wrist

        max_dist = max( [np.linalg.norm(u-wrist) for u in secondary_fingers] )
        if max_dist > self.dist_threshold_wrist:
            return

        #check if the thumb is pointing in the same direction as hand is
        thumb_base = fingers[1]
        thumb_tip = fingers[3]

        hand_direction = knuckles[0] - knuckles[1] #index knuckle - pinky knuckle

        thumb_diff_vector = thumb_tip - thumb_base
        angle = self.angle_between(thumb_diff_vector, hand_direction)
        # print(angle, angle < self.angle_threshold_thumb)
        if abs(angle) > self.angle_threshold_thumb:
            return


        self.count += 1

        if self.count > self.max_value:
            self.state = True
            self.count = 0

    def angle_between(self, v1, v2):
        v1, v2 = np.asarray(v1), np.asarray(v2)
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def get_display_value(self):
        return self.count