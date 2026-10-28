import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender
from PIL import Image
from math import atan, atan2, degrees
from pathlib import Path

HATS = [
    #hat for the chef and rat for sous chef
    {
        "obj": "3D_models/chef_hat.obj",
        "texture": "textures/hat.jpeg",
        "width": 190.0,
        "offset": [0.0, 70.0, -50.0],
        "color": [0.9, 0.7, 0.25, 1.0],
    },
    {
        "obj": "3D_models/rat.obj",
        "texture": "textures/rat.png",
        "width": 190.0,
        "offset": [0.0, 70.0, -70.0],
        "color": [0.25, 0.45, 0.9, 1.0],
    },
]

REACTION_PNG_PATHS = {
    "angry": "reactions/eyebrows.png",
    "sad": "reactions/tears.png",
}

REACTION_CYCLE = [None, "angry", "sad"]
player_reactions = [None, None]

HAT_WIDTH = 190.0
HAT_OFFSET = np.array([0.0, 70.0, -80.0])

def load_reaction_png(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

def overlay_png(frame, png, center, target_width, roll_degrees=0.0):
    target_width = int(round(target_width))
    if target_width <= 0:
        return

    source_h, source_w = png.shape[:2]
    target_height = max(1, int(round(source_h * target_width / source_w)))
    resized = cv2.resize(png, (int(target_width), target_height))

    #rotate the PNG with the face
    matrix = cv2.getRotationMatrix2D( (target_width / 2.0, target_height / 2.0), -roll_degrees,1.0,)
    rotated = cv2.warpAffine(resized,matrix,(target_width, target_height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))

    frame_h, frame_w = frame.shape[:2]
    center_x, center_y = (int(round(value)) for value in center)
    x0 = center_x - target_width // 2
    y0 = center_y - target_height // 2
    x1 = x0 + target_width
    y1 = y0 + target_height

    #clip overlay to visible
    visible_x0 = max(0, x0)
    visible_y0 = max(0, y0)
    visible_x1 = min(frame_w, x1)
    visible_y1 = min(frame_h, y1)

    if visible_x0 >= visible_x1 or visible_y0 >= visible_y1:
        return

    overlay = rotated[int(visible_y0 - y0):int(visible_y1 - y0),
              int(visible_x0 - x0):int(visible_x1 - x0)]

    roi = frame[visible_y0:visible_y1, visible_x0:visible_x1]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (
        overlay[:, :, :3].astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def draw_face_reaction(frame, face, reaction, reaction_images, width, height):
    def point(index):
        landmark = face.landmark[index]
        return np.array([landmark.x * width, landmark.y * height])

    eye_left = point(33)
    eye_right = point(263)
    eye_center = (eye_left + eye_right) / 2.0
    roll = degrees(atan2(
        eye_right[1] - eye_left[1],
        eye_right[0] - eye_left[0],
    ))

    face_left = point(234)
    face_right = point(454)
    face_width = np.linalg.norm(face_right - face_left)

    if reaction == "angry":
        brow_left = point(105)
        brow_right = point(334)
        center = (brow_left + brow_right) / 2.0
        target_width = np.linalg.norm(brow_right - brow_left) * 1.6
    elif reaction == "sad":
        center = eye_center + np.array([0, face_width * 0.30])
        target_width = face_width * 0.8
    else:
        return

    overlay_png(frame,reaction_images[reaction],center,target_width,roll)

def load_hat(obj_path, texture_path, target_width, color):
    mesh = trimesh.load(obj_path, force="mesh", process=False)

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a mesh from {obj_path}")

    uv = getattr(mesh.visual, "uv", None)
    material = None

    if texture_path and uv is None:
        print(f"Warning: {obj_path} has no UV coordinates, texture ignored")

    if uv is not None and texture_path:
        texture_bgr = cv2.imread(texture_path)
        if texture_bgr is None:
            raise FileNotFoundError(f"Could not load {texture_path}")

        texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
        # texture_rgb = cv2.flip(texture_rgb, 0)   # uncomment if the texture looks upside down

        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=pyrender.Texture(
                source=texture_rgb,
                source_channels="RGB",
            ),
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )

    bounds = mesh.bounds
    center_x = (bounds[0, 0] + bounds[1, 0]) / 2.0
    base_y = bounds[0, 1]
    center_z = (bounds[0, 2] + bounds[1, 2]) / 2.0
    mesh.vertices -= np.array([center_x, base_y, center_z])

    model_width = mesh.bounds[1, 0] - mesh.bounds[0, 0]
    if model_width <= 0:
        raise ValueError(f"The model in {obj_path} has zero width.")

    mesh.apply_scale(target_width / model_width)
    return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)


def get_image_point(face_landmarks, index, width, height):
    landmark = face_landmarks.landmark[index]
    return [landmark.x * width, landmark.y * height]


def estimate_head_pose(face_landmarks, width, height, camera_matrix):
    # Approximate 3D face points in millimeters, corresponding to the landmarks below.
    # This is a generic face shape; individual face proportions vary.
    face_3d = np.array([
        [0.0,   0.0,   0.0],     # nose tip: landmark 1
        [0.0, -63.0, -12.0],     # chin: landmark 152
        [-43.0, 32.0, -26.0],    # outer eye corner: landmark 33
        [43.0,  32.0, -26.0],    # outer eye corner: landmark 263
        [-28.0, -28.0, -24.0],   # mouth corner: landmark 61
        [28.0,  -28.0, -24.0],   # mouth corner: landmark 291
    ], dtype=np.float64)

    landmark_indices = [1, 152, 33, 263, 61, 291]
    face_2d = np.array([
        get_image_point(face_landmarks, index, width, height)
        for index in landmark_indices
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d,
        face_2d,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation, _ = cv2.Rodrigues(rotation_vector)
    face_pose = np.eye(4, dtype=np.float64)
    face_pose[:3, :3] = rotation
    face_pose[:3, 3] = translation_vector.reshape(3)
    return face_pose

class FaceFilter:
    def __init__(self):
        self.player_reactions = [None, None]
        self.renderer = None
        self.renderer_size = None
        self.hat_assignments = {}

        asset_dir = Path(__file__).resolve().parent

        self.hat_meshes = [
            load_hat(
                str(asset_dir / hat["obj"]),
                str(asset_dir / hat["texture"]) if hat["texture"] else None,
                hat["width"],
                hat["color"],
            )
            for hat in HATS
        ]

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.reaction_images = {
            name: load_reaction_png(str(asset_dir / path))
            for name, path in REACTION_PNG_PATHS.items()
        }

    def process_frame(self, frame, tracked_players=(), active_track_ids=None):
        """Add the hats and reactions to a BGR frame and return it."""
        height, width = frame.shape[:2]

        # at the start of process_frame
        if active_track_ids is not None:
            active = {str(t) for t in active_track_ids}
            self.hat_assignments = {k: v for k, v in self.hat_assignments.items() if k in active}
            matched_faces = []  # (face, hat_index) pairs, used for reactions below

        if self.renderer_size != (width, height):
            if self.renderer is not None:
                self.renderer.delete()

            self.renderer = pyrender.OffscreenRenderer(width, height)
            self.renderer_size = (width, height)

        focal_length = float(width)
        camera_matrix = np.array([
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        faces = results.multi_face_landmarks or []
        faces = sorted(faces, key=lambda face: face.landmark[1].x)

        if faces:
            scene = pyrender.Scene(
                bg_color=[0.0, 0.0, 0.0, 0.0],
                ambient_light=[0.35, 0.35, 0.35],
            )

            camera_yfov = 2.0 * atan(height / (2.0 * focal_length))
            scene.add(
                pyrender.PerspectiveCamera(
                    yfov=camera_yfov,
                    aspectRatio=width / height,
                ),
                pose=np.eye(4),
            )

            light_pose = np.eye(4)
            light_pose[:3, 3] = [0.0, 2.0, 2.0]
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
                pose=light_pose,
            )

            cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
            hats_added = 0
            print("faces:", len(faces), "tracked:", len(tracked_players), "hats:", hats_added)
            unmatched_players = list(tracked_players)

            for face in faces:
                face_pose = estimate_head_pose(
                    face, width, height, camera_matrix
                )
                if face_pose is None:
                    continue

                face_x = face.landmark[1].x * width
                face_y = face.landmark[1].y * height
                match_index = self._match_tracked_player(
                    face_x, face_y, unmatched_players
                )
                if match_index is None:
                    continue

                track_id, _ = unmatched_players.pop(match_index)
                track_key = str(track_id)

                if track_key not in self.hat_assignments:
                    used = set(self.hat_assignments.values())
                    free = [i for i in range(len(HATS)) if i not in used]
                    if not free:
                        continue
                    self.hat_assignments[track_key] = free[0]

                hat_index = self.hat_assignments[track_key]
                matched_faces.append((face, hat_index))

                hat = HATS[hat_index]
                hat_mesh = self.hat_meshes[hat_index]
                hat_offset = np.eye(4, dtype=np.float64)
                hat_offset[:3, 3] = hat["offset"]

                hat_pose = cv_to_gl @ face_pose @ hat_offset
                scene.add(hat_mesh, pose=hat_pose)
                hats_added += 1

            if hats_added:
                rendered, _ = self.renderer.render(
                    scene,
                    flags=pyrender.RenderFlags.RGBA,
                )

                rendered_bgr = cv2.cvtColor(
                    rendered[:, :, :3],
                    cv2.COLOR_RGB2BGR,
                )
                alpha = rendered[:, :, 3:4].astype(np.float32) / 255.0

                frame = (
                    rendered_bgr.astype(np.float32) * alpha
                    + frame.astype(np.float32) * (1.0 - alpha)
                ).astype(np.uint8)

        for face, player_index in matched_faces:
            reaction = self.player_reactions[player_index]
            if reaction is not None:
                draw_face_reaction(
                    frame,
                    face,
                    reaction,
                    self.reaction_images,
                    width,
                    height,
                )

        return frame

    @staticmethod
    def _match_tracked_player(face_x, face_y, tracked_players):
        best_index = None
        best_distance = float("inf")

        for index, (_, bbox) in enumerate(tracked_players):
            x1, y1, x2, y2 = bbox
            box_width = max(x2 - x1, 1)
            box_height = max(y2 - y1, 1)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            distance = (
                ((face_x - center_x) / box_width) ** 2
                + ((face_y - center_y) / box_height) ** 2
            )

            if distance < best_distance:
                best_index = index
                best_distance = distance

        if best_distance > 1.0:
            return None
        return best_index

    def cycle_reaction(self, player_index):
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1")

        current = self.player_reactions[player_index]
        next_index = (REACTION_CYCLE.index(current) + 1) % len(REACTION_CYCLE)
        self.player_reactions[player_index] = REACTION_CYCLE[next_index]