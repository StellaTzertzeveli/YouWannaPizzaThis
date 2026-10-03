"""Temporary stand-in for Tasks 2 and 3 (THROWAWAY test code).

Gives Task 4 keypoints for two players in the agreed format, so we can
build and test zones and detectors before the real tracking is ready.

How it cheats:
- It cuts the picture in half: left half = Sous Chef, right half = Chef.
  (Task 3 will do real identity tracking; here a player who walks to the
   other side simply becomes the other player.)
- It runs one MediaPipe Pose per half (MediaPipe Pose finds one person).

When Tasks 2 and 3 deliver, replace FakeInput with their code. As long as
their output has the same format, nothing else in Task 4 has to change.

-------------------------------------------------------------------------
AGREED FORMAT: read() returns a list with one dictionary per player:

    {
        "player_id": 1,              # 1 = Chef, 2 = Sous Chef
        "role": "CHEF",              # CHEF or SOUS_CHEF (from commands.py)
        "visible": True,             # False = player lost this frame
        "keypoints": {
            "nose":           (x, y, confidence),
            "left_shoulder":  (x, y, confidence),
            "right_shoulder": (x, y, confidence),
            "left_elbow":     (x, y, confidence),
            "right_elbow":    (x, y, confidence),
            "left_wrist":     (x, y, confidence),
            "right_wrist":    (x, y, confidence),
        },
    }

x, y:        pixels in the prepared (mirrored, 1280 x 720) frame,
             so they can go straight into zone_at(x, y).
confidence:  0.0 to 1.0, how sure MediaPipe is that the point is visible.
If "visible" is False, "keypoints" is empty.
-------------------------------------------------------------------------
"""
import cv2
import mediapipe as mp

from commands import CHEF, SOUS_CHEF
from player_signals.zones import FRAME_WIDTH, prepare_frame, zone_at, ZONE_RECTS

# Player IDs (as in the meeting notes: Player 1 = Chef, Player 2 = Sous Chef)
CHEF_ID = 1
SOUS_CHEF_ID = 2

# MediaPipe Pose numbers its 33 body points; these are the ones we need.
KEYPOINT_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

# Below this confidence a keypoint should not be trusted.
CONFIDENCE_THRESHOLD = 0.5

# Where the picture is cut in two.
HALF = FRAME_WIDTH // 2      # // divides and drops the decimals: 1280 // 2 = 640


def is_reliable(keypoint):
    """True if a keypoint exists and MediaPipe is confident enough about it.

    Example:
        wrist = player["keypoints"].get("right_wrist")
        if is_reliable(wrist):
            x, y, confidence = wrist
    """
    return keypoint is not None and keypoint[2] >= CONFIDENCE_THRESHOLD


class FakeInput:
    """Finds two players in a frame and returns their keypoints."""

    def __init__(self):
        # One Pose detector per half. Each remembers its own person between
        # frames, so they must not be shared.
        # model_complexity=1 is the "full" model that comes inside the
        # mediapipe package, so it works without internet. (0 is faster but
        # is downloaded the first time you use it.)
        self.pose_left = mp.solutions.pose.Pose(model_complexity=1)
        self.pose_right = mp.solutions.pose.Pose(model_complexity=1)

    def read(self, frame):
        """frame: a PREPARED frame (after prepare_frame). Returns 2 players."""
        left_half = frame[:, :HALF]      # all rows, columns 0 .. 639
        right_half = frame[:, HALF:]     # all rows, columns 640 .. 1279

        sous_chef = self._detect(self.pose_left, left_half, 0, SOUS_CHEF_ID, SOUS_CHEF)
        chef = self._detect(self.pose_right, right_half, HALF, CHEF_ID, CHEF)
        return [chef, sous_chef]

    def _detect(self, pose, image, x_offset, player_id, role):
        """Run one Pose detector on one half and build the player dictionary."""
        player = {
            "player_id": player_id,
            "role": role,
            "visible": False,
            "keypoints": {},
        }

        # OpenCV stores colours as BGR, MediaPipe expects RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks is None:    # nobody found in this half
            return player

        height, width = image.shape[:2]
        for name, index in KEYPOINT_INDEX.items():
            landmark = result.pose_landmarks.landmark[index]
            # MediaPipe gives x and y as fractions (0.0 to 1.0) of the image.
            # Turn them into pixels, and shift the right half back into place.
            x = int(landmark.x * width) + x_offset
            y = int(landmark.y * height)
            player["keypoints"][name] = (x, y, landmark.visibility)

        player["visible"] = True
        return player

    def close(self):
        """Free MediaPipe's resources when you are done."""
        self.pose_left.close()
        self.pose_right.close()


# ---------------------------------------------------------------------------
# Demo: webcam window showing keypoints, zones and which zone each wrist is in.
#   python fake_input.py      (press q to quit)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = FakeInput()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = prepare_frame(frame)          # mirror + resize, always first
        players = tracker.read(frame)

        # Zones in yellow, and the line where the picture is cut in two
        for (x1, y1, x2, y2) in ZONE_RECTS.values():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.line(frame, (HALF, 0), (HALF, frame.shape[0]), (255, 255, 255), 1)

        for player in players:
            if not player["visible"]:
                continue                      # skip to the next player

            for name, keypoint in player["keypoints"].items():
                x, y, confidence = keypoint
                # Green = reliable, red = unsure
                colour = (0, 255, 0) if is_reliable(keypoint) else (0, 0, 255)
                cv2.circle(frame, (x, y), 6, colour, -1)

            # Label each wrist, plus the zone it is in (if any)
            for wrist_name, short in (("left_wrist", "L"), ("right_wrist", "R")):
                wrist = player["keypoints"][wrist_name]
                if is_reliable(wrist):
                    x, y, _ = wrist
                    text = f"P{player['player_id']} {short} {zone_at(x, y) or ''}"
                    cv2.putText(frame, text, (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Fake input test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
