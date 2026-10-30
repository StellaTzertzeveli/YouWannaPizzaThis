import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from zones import *
from images import draw_fill, blend_person
from game_state import GameState, PLAYER_ABILITIES
from motion_detectors.ChopDetector import ChopDetector
from motion_detectors.SpinDetector import SpinDetector
from motion_detectors.CrossDetector import CrossDetector
from motion_detectors.ThumbsupDetector import ThumbsupDetector
from face_filters.face_class import FaceFilter
from player import Player
from Task_4.tracking import MP_Tracking

latest_hand_result = None
latest_pose_result = None
latest_mask = None

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result

def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result

def segment_callback(result, output_image, timestamp_ms):
    global latest_mask
    latest_mask = np.array(result.confidence_masks[-1].numpy_view())

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=4,  # up to 2 per player; one_hand_per_player() below picks each player's closest hand
    result_callback=hand_callback,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=2,  # both players, so each one's cross-arms gesture can be told apart
    result_callback=pose_callback,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

segmenter_options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path="selfie_segmenter.tflite"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_confidence_masks=True,
    output_category_mask=False,
    result_callback=segment_callback,
)
segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)

# Create zones here
zones = make_zones()

# Create a game state here
state = GameState()

# Create a chop detector here (shared: either player chopping advances the same chopping counter)
chop_detector = ChopDetector()
chop_detector.start_detection()

# Create a spin detector here (shared, same reason)
spin_detector = SpinDetector()
spin_detector.start_detection()

# Player detection (face-embedding + DeepSORT tracking) and face filters
mp_tracker = MP_Tracking()
face_filter = FaceFilter()
players = [Player("Chef"), Player("Sous Chef")]
ROLES = ["CHEF", "SOUS"]  # players[i] uses PLAYER_ABILITIES[ROLES[i]]
ability_popups = []  # (player_index, text, expire_time): "X used ability" banner, shown for 1s on that player's side
REACTION_DURATION = 10  # seconds a triggered face reaction (angry/sad) stays on before clearing
reaction_expire = [0, 0]  # per player_reactions slot: timestamp when the reaction should clear

# Abilities are per-player, so each player gets their own cross/thumbs-up detector instance
cross_detectors = [CrossDetector(), CrossDetector()]
thumbsup_detectors = [ThumbsupDetector(), ThumbsupDetector()]
for d in cross_detectors + thumbsup_detectors:
    d.start_detection()


MISSING_FRAMES_LIMIT = 45  # ~1.5s of continuous absence before a player's slot frees up for someone else


def update_players(tracks):
    # Match this frame's DeepSORT tracks to the two player slots, keeping each player on the same slot across re-detections.
    tracked_players = []
    active_track_ids = []

    # The very first time both players are visible together with neither slot assigned yet, assign
    # by screen position instead of whichever order the tracker happened to list them in: the player
    # on screen-left becomes the Sous Chef (players[1]), the player on screen-right becomes the Chef (players[0]).
    visible = [t for t in tracks if t.time_since_update == 0][:2]
    if len(visible) == 2 and players[0].id == 0 and players[1].id == 0:
        visible.sort(key=lambda t: t.to_ltrb()[0])  # ascending x1: left to right
        for slot, track in zip((1, 0), visible):
            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            players[slot].id = track_id
            players[slot].bbox = [(x1, y1, x2, y2)]
            players[slot].missing_frames = 0
            active_track_ids.append(players[slot].name)
            tracked_players.append((track_id, (x1, y1, x2, y2)))
        return tracked_players, active_track_ids

    # Match each of this frame's tracks to a player by track_id alone -- never by the position it
    # happens to appear at in the tracker's list, which isn't guaranteed stable frame to frame and
    # was previously causing the two players' roles to swap whenever that order changed.
    seen_slots = set()
    for track in visible:
        track_id = int(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        bbox = [(x1, y1, x2, y2)]

        slot = next((i for i, p in enumerate(players) if p.id == track_id), None)
        if slot is None:
            # A person we haven't seen before: give them the first free slot, if any.
            slot = next((i for i, p in enumerate(players) if p.id == 0), None)
            if slot is None:
                continue  # both slots already taken by the two tracked players; ignore anyone else
            players[slot].id = track_id

        players[slot].bbox = bbox
        players[slot].missing_frames = 0
        seen_slots.add(slot)
        active_track_ids.append(players[slot].name)
        tracked_players.append((track_id, (x1, y1, x2, y2)))

    # A player who wasn't seen this frame: once they've been gone long enough that the tracker has
    # genuinely lost them (not just a brief occlusion), free their slot. Without this, walking out of
    # frame and coming back with a new track_id (ByteTrack doesn't remember people forever) finds both
    # slots looking permanently "taken" by a track_id that no longer exists, and never gets re-assigned.
    for i, player in enumerate(players):
        if i in seen_slots or player.id == 0:
            continue
        player.missing_frames += 1
        if player.missing_frames > MISSING_FRAMES_LIMIT:
            player.id = 0
            player.bbox = None
            player.missing_frames = 0

    return tracked_players, active_track_ids


def nearest_player(x, y):
    # Which player this point (a palm or a nose) belongs to, so a gesture gets credited to the right player.
    # Prefer whoever's own tracked bbox actually contains the point, so a hand reaching toward the other
    # player's side doesn't get stolen just because it's closer to that player's box centre.
    containing = []
    for i, player in enumerate(players):
        if player.id == 0 or not player.bbox:
            continue
        x1, y1, x2, y2 = player.bbox[0]
        if x1 <= x <= x2 and y1 <= y <= y2:
            containing.append(i)
    if len(containing) == 1:
        return containing[0]

    # Nobody's box contains it (or both do, since boxes overlap) -> fall back to nearest centre,
    # restricted to the containing players when there are any.
    candidates = containing if containing else range(len(players))
    best_index, best_dist = None, float("inf")
    for i in candidates:
        player = players[i]
        if player.id == 0 or not player.bbox:
            continue
        x1, y1, x2, y2 = player.bbox[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist < best_dist:
            best_index, best_dist = i, dist
    return best_index


def one_hand_per_player(hands, w, h):
    # If a player raises both hands, keep only the one closest to their tracked position,
    # so each player ever contributes at most one hand's worth of gestures per frame.
    best = {}  # player_index -> (dist_sq, hand, palm)
    for hand in hands:
        palm = hand[9]
        x, y = palm.x * w, palm.y * h
        player_index = nearest_player(x, y)
        if player_index is None:
            continue
        x1, y1, x2, y2 = players[player_index].bbox[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if player_index not in best or dist < best[player_index][0]:
            best[player_index] = (dist, hand, palm)
    return {i: (hand, palm) for i, (dist, hand, palm) in best.items()}


def ability_display_name(role, number):
    return PLAYER_ABILITIES[role][number - 1].replace("_", " ").title()


def player_side(player, w):
    # Which half of the screen this player is actually on right now, so the popup lands on their
    # side regardless of which role (Chef/Sous) is currently assigned to left vs. right.
    if not player.bbox:
        return "left"
    x1, _, x2, _ = player.bbox[0]
    return "left" if (x1 + x2) / 2 < w / 2 else "right"


def draw_ability_popups(frame, popups, now):
    # "<Name> used <Ability>!" banner for 1 second on whichever side of the screen that player is on.
    popups[:] = [p for p in popups if p[2] > now]
    h, w, _ = frame.shape
    for i, (side, text, _) in enumerate(popups):
        y = int(0.3 * h) + i * 40
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = 20 if side == "left" else w - text_w - 20
        cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# Each role's two ability labels are stacked above this one station, regardless of what the ability actually does.
ABILITY_ROLE_STATION = {
    "SOUS": "CHOPPING COUNTER",
    "CHEF": "TOPPING STATION",
}


def draw_ability_labels(frame, zones):
    zones_by_name = {zone.name: zone for zone in zones}
    h, w, _ = frame.shape
    for role, abilities in PLAYER_ABILITIES.items():
        zone = zones_by_name.get(ABILITY_ROLE_STATION[role])
        if zone is None:
            continue
        cx = int((zone.x1 + zone.x2) / 2 * w)
        base_y = int(zone.y1 * h) - 8

        for number, name in enumerate(abilities, start=1):
            label = ability_display_name(role, number)
            color = (0, 200, 0) if state.ability_ready(name) else (0, 0, 220)  # green = ready, red = on cooldown
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # ability 1 sits right above the zone, ability 2 stacks above ability 1
            y = max(text_h + 5, base_y - (number - 1) * (text_h + 12))
            cv2.putText(frame, label, (cx - text_w // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_game_over(frame, score):
    h, w, _ = frame.shape
    dim = (frame.astype(np.float32) * 0.4).astype(np.uint8)
    frame[:] = dim
    for text, scale, dy in [("GAME OVER", 1.4, -30), (f"Score: {score}", 1.0, 20), ("Press R to restart, Q to quit", 0.7, 70)]:
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(frame, text, (w // 2 - text_w // 2, h // 2 + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def reset_game():
    global state, zones, ability_popups
    state = GameState()
    zones = make_zones()
    ability_popups = []
    face_filter.player_reactions = [None, None]
    reaction_expire[0] = 0
    reaction_expire[1] = 0
    for player in players:
        player.id = 0
        player.bbox = None
        player.missing_frames = 0
    for detector in [chop_detector, spin_detector] + cross_detectors + thumbsup_detectors:
        detector.stop_detection()
        detector.start_detection()


cv2.namedWindow("Zones", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Zones", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    now = time.time()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    hand_landmarker.detect_async(mp_image, timestamp)
    pose_landmarker.detect_async(mp_image, timestamp)
    segmenter.segment_async(mp_image, timestamp)

    camera = frame.copy()
    draw_fill(frame, "background", 0, 0, w, h)
    if latest_mask is not None:
        blend_person(frame, camera, latest_mask)

    for i in range(2):
        if face_filter.player_reactions[i] is not None and now >= reaction_expire[i]:
            face_filter.player_reactions[i] = None

    # Player detection: who is who (Chef / Sous Chef), then draw their hats/reactions and bounding boxes
    tracks = mp_tracker.get_ids(camera)
    tracked_players, active_track_ids = update_players(tracks)
    role_by_track_id = {player.id: ROLES[i] for i, player in enumerate(players) if player.id != 0}
    frame = face_filter.process_frame(frame, tracked_players, role_by_track_id)
    for track_id, bbox in tracked_players:
        x1, y1, x2, y2 = bbox
        for player in players:
            if player.id == track_id:
                cv2.putText(frame, player.name, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

    # Check if the palm of any detected hand is inside the bank zone
    state.update(now)  # done first so ability_ready()/on_ability() below see this frame's time, not last frame's
    palms = []
    palm_roles = []  # role of whichever player's hand each palm belongs to (parallel to palms)
    if not state.game_over:
        if latest_hand_result is not None:
            for player_index, (hand, palm) in one_hand_per_player(latest_hand_result.hand_landmarks, w, h).items():
                palms.append(palm)
                cv2.circle(frame, (int(palm.x * w), int(palm.y * h)), 8, (0, 255, 0), -1)
                role = ROLES[player_index]
                palm_roles.append(role)
                # Sous chef chops, chef spins the dough
                if role == "SOUS":
                    chop_detector.update(hand)
                if role == "CHEF":
                    spin_detector.update(hand)
                thumbsup_detectors[player_index].update(hand)

        if latest_pose_result is not None:
            for body in latest_pose_result.pose_landmarks:
                nose = body[0]
                player_index = nearest_player(nose.x * w, nose.y * h)
                if player_index is not None:
                    cross_detectors[player_index].update(body)

    # Hand every new chop to the game state, then reset the detector's count
    for _ in range(chop_detector.get_chop_count()):
        state.on_chop()
    chop_detector.reset_chop_count()

    if state.holding["CHEF"] != "DOUGH":  # only the chef ever holds dough
        spin_detector.count = 0
    if spin_detector.count >= spin_detector.max_value:
        state.on_spin("CHEF")
        spin_detector.count = 0

    # Abilities: cross-arms is each player's ability 1, thumbs-up is ability 2 (fires once per completed gesture, then sits on GameState's own per-ability cooldown)
    for i, role in enumerate(ROLES):
        if cross_detectors[i].state:
            cross_detectors[i].state = False
            if state.on_ability(role, 1):
                name = ability_display_name(role, 1)
                print(f"{players[i].name} used {name}!")
                ability_popups.append((player_side(players[i], w), f"{players[i].name}: {name}!", now + 1.0))
        if thumbsup_detectors[i].state:
            thumbsup_detectors[i].state = False
            if state.on_ability(role, 2):
                name = ability_display_name(role, 2)
                print(f"{players[i].name} used {name}!")
                ability_popups.append((player_side(players[i], w), f"{players[i].name}: {name}!", now + 1.0))
                # Stubborn (Chef) -> angry eyebrows, Rush Delivery (Sous) -> sad tears
                if role == "CHEF":
                    face_filter.player_reactions[0] = "angry"
                    reaction_expire[0] = now + REACTION_DURATION
                elif role == "SOUS":
                    face_filter.player_reactions[1] = "sad"
                    reaction_expire[1] = now + REACTION_DURATION

    for zone in zones:
        inside_roles = [r for p, r in zip(palms, palm_roles) if zone.contains(p.x, p.y)]
        if zone.update(len(inside_roles) > 0, now):
            # Tell game state that the zone was triggered, so it can update the held item or surface items accordingly
            # (role-restricted zones silently reject the wrong player's hand inside on_hold)
            role = inside_roles[0] if inside_roles else None
            state.on_hold(zone.name, role)
            print(f"{zone.name} triggered! {role} holding: {state.holding.get(role)}")
        zone.draw(frame, progress=zone.progress(now))
    # Draw the HUD (not interactible)
    draw_hud(frame)
    # Draw the game zones (interactible)
    state.draw(frame, zones, palms, palm_roles)
    draw_ability_labels(frame, zones)
    draw_ability_popups(frame, ability_popups, now)
    if state.game_over:
        draw_game_over(frame, state.score)
    cv2.imshow("Zones", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        reset_game()
    if key == ord(' '):
        state.toggle_timer()
    if key == ord('1'):
        face_filter.cycle_reaction(0)
    if key == ord('2'):
        face_filter.cycle_reaction(1)

cap.release()
cv2.destroyAllWindows()
