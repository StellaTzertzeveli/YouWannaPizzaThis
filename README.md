# YouWannaPizzaThis

IPCV group 17 assignment. A cooperative two-player webcam pizza game. Two people stand in
front of one camera, get tracked and turned into a Chef and a Sous Chef, and have to build,
bake and deliver pizzas that match an order before the timer runs out.


## Concept

One webcam watches two players. Hand, body and face tracking (MediaPipe) plus person
tracking (YOLOv8 + ByteTrack) turn each player into a character wearing a 3D hat:
the player on screen-right becomes the **Chef** (chef hat), the player on screen-left
becomes the **Sous Chef** (rat hat). Decided once, the first time both are seen together,
and kept stable even if someone briefly steps out of frame.

Together you run a kitchen: grab ingredients, chop them, spin dough, assemble a pizza,
bake it, and deliver it before the clock hits zero. Roles are split on purpose, so you have to cooperate: the Chef can't chop, the ingredients and the Sous Chef can't use the oven or spin the dough.

## Requirements

- Python 3.10+
- Everything in `requirements.txt`:
  ```
  opencv-python
  mediapipe==0.10.21
  numpy<2
  pillow
  trimesh
  pyrender
  ultralytics
  lap>=0.5.12
  ```
  (`pyrender` pulls in `PyOpenGL` itself, which `face_filters/face_class.py` needs.)
- Model files, loaded by relative path from wherever you launch the game:
  - `hand_landmarker.task`, `pose_landmarker.task`, `selfie_segmenter.tflite` — already in this folder.
  - `face_landmarker.task` — **not included** (too big to ship casually); download it from
    [Google's MediaPipe model page](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
    and drop it in this same folder before running.
  - `yolov8n.pt` — downloaded automatically by `ultralytics` the first time you run the game, if not already present (needs internet that first time).

## How to run

```bash
cd YouWannaPizzaThis
pip install -r requirements.txt
python pizzaGame.py
```

Run it from inside this folder. Assets and model files are loaded by relative path.
The game opens a fullscreen window reading from webcam index 1, pizzaGame.py - line 279 (`cv2.VideoCapture(1, cv2.CAP_DSHOW)`). if your webcam is a different index change the 1 to 0.

Get both players standing so the camera can see their upper body and both hands.

## Controls (keyboard)

| Key | Action |
|---|---|
| `Q` | Quit the game |
| `R` | Restart (resets timer, score, zones, abilities, tracked identities, reactions) |



## Who can do what (role restrictions)

| Zone | Allowed role |
|---|---|
| Dough, Oven, Topping Station | **Chef only** |
| Ingredient boxes, Chopping Counter, Trash, Delivery | **Sous Chef only** |
| Bank (x4) | Either player |


## How to do each action

- **Pick up / place / use a zone:** hold your tracked hand's palm over the zone for about
  1 second.
- **Chop:** with a raw ingredient on the Chopping Counter, mime a quick downward
  stroke followed by lifting back up. 4 chops turns it into the chopped version.
- **Spin dough** pick up Dough, then spin your hand (alternating left/right). This spins it into Dough (Spun),  the only form the Topping Station will accept.
- **Deliver / bake / top:** just hold the item over the right zone like any other pickup/placement
  — e.g. hold a chopped topping over a pizza on the Topping Station to add it; hold a baked pizza
  over Delivery to serve it.

## Abilities

Each player has two abilities, triggered by gesture rather than by standing in a zone:

- **Cross-arms** (ability slot 1): cross your forearms into an X in front of your chest and hold
  for roughly 3 seconds.
- **Thumbs-up** (ability slot 2): make a  thumbs-up and hold for about 3 seconds

| Role | Cross-arms ability | Thumbs-up ability |
|---|---|---|
| Sous Chef | **Knife Frenzy** — each chop counts 3x for 15s | **Rush Delivery** — instantly delivers the first ready (baked) pizza found anywhere on the counters, skipping the Delivery stand |
| Chef | **Hot Oven** — oven bakes/burns 3x faster for 15s | **Stubborn** — instantly rerolls the current order |

Cooldowns: Knife Frenzy & Hot Oven 22.5s, Rush Delivery 30s, Stubborn 40s.

Triggering a thumbs-up ability also puts a face reaction on that player for 10 seconds:
**Stubborn → angry eyebrows on the Chef**, **Rush Delivery → sad tears on the Sous Chef**

## How to score a point

1. The **Recipe box** shows the current order: 1–3 toppings, picked at random.
2. Build a pizza: Dough → spin it (Chef) → place on Topping Station (becomes a bare pizza) →
   Sous Chef chops the needed ingredients and adds each chopped topping at the Topping Station →
   Chef moves the topped pizza to the Oven → once baked, move it to Delivery.
3. **Deliver a pizza whose toppings exactly match the order → +1 score, +30 seconds on the clock.**
4. Deliver the wrong toppings → −5 seconds, no score.
5. Deliver a burnt pizza → −10 seconds, no score.
6. A new random order is picked after every delivery (right or wrong).

The oven bakes a pizza after 10 seconds in it, and burns it 25 seconds after that if it isn't
taken out. Hot Oven speeds both up 3x while active.


## Classes (what lives where)

| File | Class / role |
|---|---|
| `player.py` | `Player` — a tracked person's slot: name, track id, bounding box, missing-frame counter |
| `zones.py` | `Zone` — one interactive hotspot (bounds, color, hold-to-trigger timing, progress bar); `make_zones()` builds the kitchen layout; `draw_hud()` draws the recipe box/timer background |
| `game_state.py` | `GameState` — all game rules: what's held/placed where, chopping, spinning, delivering, scoring, abilities, timer; `Pizza` — a pizza's stage (raw/baked/burnt) and toppings |
| `motion_detectors/ActionDetector.py` | `ActionDetector` — shared base class: rolling 60-frame landmark history, start/stop switch |
| `motion_detectors/ChopDetector.py` | `ChopDetector` — detects the chop gesture |
| `motion_detectors/SpinDetector.py` | `SpinDetector` — detects the dough-spin shake |
| `motion_detectors/CrossDetector.py` | `CrossDetector` — detects crossed-arms (ability slot 1) |
| `motion_detectors/ThumbsupDetector.py` | `ThumbsupDetector` — detects thumbs-up (ability slot 2) |
| `motion_detectors/GrabDetector.py` | `GrabDetector` — a grab/release gesture detector; not currently wired into the game |
| `face_filters/face_class.py` | `FaceFilter` — tracks faces, renders each player's 3D hat with pyrender, overlays the angry/sad reaction images |
| `Task_4/tracking.py` | `MP_Tracking` — wraps YOLOv8 + ByteTrack person detection/tracking behind a simple `get_ids(frame)` call |
| `images.py` | sprite-loading/drawing helpers (`draw_image`, `draw_fill`, transparent-PNG cropping+caching) |
| `pizzaGame.py` | the game itself — sets up the three MediaPipe landmarkers (hands/pose/segmentation) and ties every class above together in the main loop |

`main.py` and `MediaPipe.py` are earlier standalone prototypes (DeepSort+FaceNet tracking,
an older MediaPipe wiring) kept for reference only — they're not part of the actual game and
`main.py` no longer matches `FaceFilter`'s current interface, so it won't run as-is.
