import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender
from PIL import Image
from math import atan, degrees


HATS = [
    #a different hat for the chef and for the chous chef
    {
        "obj": "3D_models/chef_hat.obj",
        "texture": None,
        "width": 190.0,
        "offset": [0.0, 70.0, -50.0],
        "color": [0.9, 0.7, 0.25, 1.0],
    },
    {
        "obj": "3D_models/Ushanka.obj",
        "texture": None,
        "width": 190.0,
        "offset": [0.0, 70.0, -60.0],
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
    roll = degrees(np.atan2(
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

    if uv is not None and texture_path:
        texture_bgr = cv2.imread(texture_path)
        if texture_bgr is None:
            raise FileNotFoundError(f"Could not load {texture_path}")

        texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
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


def main():
    hat_meshes = [
        load_hat(hat["obj"], hat["texture"], hat["width"], hat["color"])
        for hat in HATS
    ]

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    reaction_images = {name: load_reaction_png(path)
                       for name, path in REACTION_PNG_PATHS.items()}

    renderer = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            height, width = frame.shape[:2]

            focal_length = float(width)
            camera_matrix = np.array([
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            if renderer is None:
                renderer = pyrender.OffscreenRenderer(width, height)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            faces = []
            if results.multi_face_landmarks:
                faces = sorted(
                    results.multi_face_landmarks,
                    key=lambda face: face.landmark[1].x,)

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

                # Leftmost detected face gets HATS[0]; next face gets HATS[1].
                for face, hat, hat_mesh in zip(faces, HATS, hat_meshes):
                    face_pose = estimate_head_pose(face, width, height, camera_matrix)
                    if face_pose is None:
                        continue

                    hat_offset = np.eye(4, dtype=np.float64)
                    hat_offset[:3, 3] = hat["offset"]

                    hat_pose_cv = face_pose @ hat_offset
                    hat_pose_gl = cv_to_gl @ hat_pose_cv
                    scene.add(hat_mesh, pose=hat_pose_gl)
                    hats_added += 1

                if hats_added:
                    rendered, _ = renderer.render(
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

            for player_index, face in enumerate(faces[:2]):
                reaction = player_reactions[player_index]
                if reaction is not None:
                    draw_face_reaction(frame, face, reaction, reaction_images, width, height

                    )

            cv2.imshow("3D face filter - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key in (ord('1'), ord('2')):
                player_index = key - ord('1')
                current = player_reactions[player_index]
                next_index = (REACTION_CYCLE.index(current) + 1) % len(REACTION_CYCLE)
                player_reactions[player_index] = REACTION_CYCLE[next_index]
                print(f"Player {player_index + 1}: {player_reactions[player_index] or 'no reaction'}")

    finally:
        camera.release()
        face_mesh.close()
        if renderer is not None:
            renderer.delete()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()