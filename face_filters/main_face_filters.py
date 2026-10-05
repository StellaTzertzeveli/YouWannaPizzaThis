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

# These values are initial fit adjustments in the approximate face-model units.
# Adjust them if the hat is too high, low, close, or far from the forehead.
HAT_WIDTH = 190.0
HAT_OFFSET = np.array([0.0, 70.0, -10.0])


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

    renderer = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            height, width = frame.shape[:2]

            # Approximate webcam intrinsics. Real calibration can improve alignment.
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

            if results.multi_face_landmarks:
                faces = sorted(
                    results.multi_face_landmarks,
                    key=lambda face: face.landmark[1].x,
                )

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

            cv2.imshow("3D face filter - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        face_mesh.close()
        if renderer is not None:
            renderer.delete()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()