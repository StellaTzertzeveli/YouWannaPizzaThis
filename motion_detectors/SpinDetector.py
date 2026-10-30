from motion_detectors.ActionDetector import ActionDetector
import numpy as np
class SpinDetector(ActionDetector):
    def _init_(self):
        super()._init_()
        self.mp_dict = {0: 'wrist', 8: 'index_tip'}
        self.direction = None
        self.direction_count = 0
        self.count = 0
        self.speed_threshold = 0.15
        self.max_value = 30
        self.max_direction_count = 5

    def vector_direction(self, dx):
        if dx > 0:
            return 'right'
        else:
            return 'left'

    def stop_detection(self):
        self.direction = None
        self.count = 0

    def update_state(self, should_print):
        vectors = self.vector_history[-1]
        old_vectors = self.vector_history[-2]
        for limb_str in self.mp_dict.values():
            if not limb_str in vectors.keys() or not limb_str in old_vectors.keys():
                return
        old_tip = old_vectors['index_tip']
        new_tip = vectors['index_tip']
        direction_vector = new_tip - old_tip
        dx, dy = direction_vector
        direction = self.vector_direction(dx)

        if self.direction is None:
            self.direction = direction
            return
        print(self.direction, direction)
        speed = np.linalg.norm(direction_vector)
        if speed < self.speed_threshold:
            return
        if direction == self.direction: #still going the same direction
            self.direction_count += 1

        self.direction = direction
        self.direction_count = 0

        if self.direction_count >= self.max_direction_count:
            return
        self.count += 1


    def get_display_value(self):
        return self.count