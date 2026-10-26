# YouWannaPizzaThis

Real-time face tracking from a webcam. Faces are detected with **MTCNN**, turned into identity embeddings with **FaceNet**, and tracked across frames with **DeepSORT**, so each person keeps a stable ID while moving around.

## How it works

```
webcam frame
   │
   ├─ MTCNN ──────────► face boxes + confidence
   │                     (detections below 0.9 are dropped)
   │
   ├─ MTCNN.extract ──► 160×160 face crops
   │
   ├─ FaceNet ────────► 512-d embedding per face
   │                     (InceptionResnetV1, VGGFace2 weights)
   │
   └─ DeepSORT ───────► track IDs
                         (motion via Kalman filter + appearance via
                          cosine distance between embeddings)
```

## Requirements

- Python 3.10–3.12
- A webcam
- Optional: NVIDIA GPU with CUDA (detection and embeddings run on GPU automatically if available)

## Setup

Create and activate a virtual environment (Windows / PowerShell):

```powershell
python -m venv IPCV
IPCV\Scripts\Activate.ps1
```

Install from the pinned requirements:

```powershell
pip install -r requirements.txt --no-deps
```

> **Why `--no-deps`?** `facenet-pytorch` 2.6.0 declares outdated pins
> (torch 2.2, numpy < 2, Pillow 10.2) that conflict with current versions.
> The code itself runs fine on newer versions, so dependencies are installed
> explicitly instead of letting pip resolve them. pip will still print a
> dependency-conflict warning — it can be ignored.

### Installing from scratch (without `requirements.txt`)

```powershell
pip install opencv-python deep-sort-realtime pillow requests tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU
# pip install torch torchvision                                                  # CPU only
pip install facenet-pytorch --no-deps
pip freeze > requirements.txt
```

Pick the CUDA build that matches your driver on [pytorch.org](https://pytorch.org/get-started/locally/).
If `requirements.txt` contains a CUDA torch build (e.g. `torch==...+cu124`), add this as its first line:

```
--extra-index-url https://download.pytorch.org/whl/cu124
```

The FaceNet weights are downloaded automatically on the first run.

## Usage

```powershell
python main.py
```

A window titled **Out** shows the camera feed with a box and ID per tracked face.
Press **Q** to quit.

## Tuning

| Parameter | Where | Default | Effect |
|---|---|---|---|
| `MIN_CONFIDENCE` | script | `0.9` | Drop weaker face detections |
| `margin` | `MTCNN(...)` | `20` | Pixels of context around each face crop |
| `min_face_size` | `MTCNN(...)` | `20` | Raise (40–80) to skip small faces and speed up detection |
| `max_cosine_distance` | `DeepSort(...)` | `0.4` | Lower = stricter identity matching, fewer ID swaps but more new IDs |
| `n_init` | `DeepSort(...)` | `5` | Frames a face must be seen before its track is confirmed |
| `max_age` | `DeepSort(...)` | `30` | Frames a track survives without a detection |

## Troubleshooting

- **`ModuleNotFoundError` inside `facenet_pytorch`** — a dependency was skipped by `--no-deps`; install the named package (e.g. `pip install tqdm`).
- **Runs on CPU despite a GPU** — check `torch.cuda.is_available()`; reinstall torch with the CUDA index URL above.
- **`Activate.ps1` cannot be loaded** — run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Credits & license notes

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN and InceptionResnetV1 (VGGFace2 pretrained weights)
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) — DeepSORT tracker

Check the licenses of the pretrained weights before any non-academic use.