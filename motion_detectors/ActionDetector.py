import cv2
import numpy as np
class ActionDetector:
    def __init__(self):
        self.history = []
        self.vector_history = []
        self.mp_dict = {}
        self.max_value = -99
        self.should_detect = False
        pass

    def update(self, new_mp, should_print=False):
        self.update_history(new_mp)
        if len(self.history) >= 60 and self.should_detect:
            self.update_state(should_print)

    def start_detection(self):
        self.should_detect = True

    def stop_detection(self):
        raise NotImplementedError("Subclass must implement update_state")

    def update_state(self, should_print):
        raise NotImplementedError("Subclass must implement update_state")

    def get_display_value(self):
        raise NotImplementedError("Subclass must implement")

    def update_history(self, limb):
        result_dict = {}
        vectors = {}
        for idx, limb_str in self.mp_dict.items():
            body_part = limb[idx]
            coords = self.body_part_to_coords(body_part)
            result_dict[limb_str] = coords
            vector = np.array((coords[0], coords[1]))
            vectors[limb_str] = vector

        self.history.append(result_dict)
        self.vector_history.append(vectors)

        if len(self.history) > 60:
            self.history.pop(0)
            self.vector_history.pop(0)

    def body_part_to_coords(self, part):
        return part.x, part.y, part.z

    def display_bar_graph(self, frame):
        value = self.get_display_value()
        h, w, _ = frame.shape
        # Position and size
        x = 50
        y = 50
        bar_width = 300
        bar_height = 30

        # Clamp value
        value = max(0, min(value, self.max_value))

        # Calculate filled width
        fill_width = int((value / self.max_value) * bar_width)

        # Background
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (50, 50, 50),
            -1
        )

        # Filled portion
        cv2.rectangle(
            frame,
            (x, y),
            (x + fill_width, y + bar_height),
            (0, 255, 0),
            -1
        )

        # Border
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (255, 255, 255),
            2
        )

        # Value text
        cv2.putText(
            frame,
            f"{value}/{self.max_value}",
            (x + bar_width + 15, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame