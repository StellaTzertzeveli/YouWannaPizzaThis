"""TEMPORARY stand-in for Tasks 2 and 3: keypoints for two players.

Replace FakeInput with the real tracker when it is ready. As long as the
real tracker's read(frame) returns the same format, nothing else changes.
"""
import sys           # sys.exit(): stop with a clear message
import cv2           # colour conversion for MediaPipe

from commands import CHEF, SOUS_CHEF, CHEF_ID, SOUS_CHEF_ID
from zones import FRAME_WIDTH


# =====================================================================
# FAKE INPUT - TEMPORARY stand-in for Tasks 2 + 3
#
# Cuts the picture in half (left = Sous Chef, right = Chef) and runs one
# MediaPipe Pose per half. When Tasks 2 and 3 deliver, replace FakeInput
# with their code; as long as read() returns the same format, nothing
# else has to change.
#
# read() returns a list with one dictionary per player:
#   {"player_id": 1, "role": "CHEF", "visible": True,
#    "keypoints": {"nose": (x, y, confidence), "left_wrist": (...), ...}}
# x, y are pixels in the prepared 1280 x 720 frame. If "visible" is False,
# "keypoints" is empty.
# =====================================================================
KEYPOINT_INDEX = {            # MediaPipe Pose numbers its 33 body points
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
HALF = FRAME_WIDTH // 2       # 1280 // 2 = 640



class FakeInput:
    """Finds two players in a frame and returns their keypoints."""

    def __init__(self):
        # Imported here (not at the top) so --selftest works without mediapipe.
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            # MediaPipe 0.10.30 and newer removed mp.solutions (the old Pose API).
            sys.exit(f"Your MediaPipe ({mp.__version__}) no longer has mp.solutions.pose.\n"
                     "Use the same Python + MediaPipe install that runs fake_input.py,\n"
                     "or ask Task 2 which pose library the team uses.")
        # One Pose detector per half: each remembers its own person between frames.
        self.pose_left = mp.solutions.pose.Pose(model_complexity=1)
        self.pose_right = mp.solutions.pose.Pose(model_complexity=1)

    def read(self, frame):
        """frame: a PREPARED frame. Returns [chef, sous_chef]."""
        # Cut the image in two with numpy "slicing": frame[rows, columns].
        # ":" alone = all rows; ":HALF" = from the start up to HALF; "HALF:" = from HALF to the end.
        left_half = frame[:, :HALF]      # all rows, columns 0 .. 639
        right_half = frame[:, HALF:]     # all rows, columns 640 .. 1279
        # Run pose detection on each half. The right half needs x_offset=HALF so its
        # x positions can be shifted back to full-frame coordinates afterwards.
        sous_chef = self._detect(self.pose_left, left_half, 0, SOUS_CHEF_ID, SOUS_CHEF)
        chef = self._detect(self.pose_right, right_half, HALF, CHEF_ID, CHEF)
        return [chef, sous_chef]

    def _detect(self, pose, image, x_offset, player_id, role):
        # Step 1: start with an "empty" player: not visible, no keypoints.
        # If nobody is found, this is what we return.
        player = {"player_id": player_id, "role": role, "visible": False, "keypoints": {}}

        # Step 2: run MediaPipe Pose on this half of the image.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # OpenCV = BGR, MediaPipe = RGB
        result = pose.process(rgb)
        if result.pose_landmarks is None:              # nobody in this half
            return player

        # Step 3: copy the 7 body points we need into the agreed format.
        height, width = image.shape[:2]                # shape = (rows, columns, colours)
        for name, index in KEYPOINT_INDEX.items():
            landmark = result.pose_landmarks.landmark[index]
            # MediaPipe gives fractions (0.0 to 1.0); turn them into pixels
            # and shift the right half back into place.
            x = int(landmark.x * width) + x_offset
            y = int(landmark.y * height)
            # landmark.visibility = how sure MediaPipe is (0.0 to 1.0): our "confidence".
            player["keypoints"][name] = (x, y, landmark.visibility)

        player["visible"] = True                       # Step 4: mark the player as found
        return player

    def close(self):
        # MediaPipe holds memory and a model; free it when the game ends.
        self.pose_left.close()
        self.pose_right.close()
