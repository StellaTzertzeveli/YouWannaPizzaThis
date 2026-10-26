from motion_detectors.ActionDetector import ActionDetector

class SpinDetector(ActionDetector):

    def __init__(self):
        super().__init__()
        self.mp_dict = {0: 'wrist', 8: 'index_tip'}
        self.direction = None
        self.count = 0
        self.speed_threshold = 0.05

    def vector_direction(self, vector):
        if vector[0] > 0:
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
        old_tip = old_vectors[0]
        new_tip = vectors[0]
        direction_vector = new_tip - old_tip
        direction = self.vector_direction(direction_vector)

        if self.direction is None:
            self.direction = direction

        if direction == self.direction: #still going the same direction
            pass
        else: #changing direction
            pass

        self.count += 1


    def get_display_value(self):
        raise NotImplementedError("Subclass must implement")