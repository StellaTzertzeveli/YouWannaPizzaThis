import cv2
CHOPPING_DOWNWARDS = 999
LIFTING_UPWARDS = 998
CHOPPED = 997

def print_chop_state(state):
    return {997: 'CHOPPED', 998: 'LIFTING_UPWARDS', 999: 'CHOPPING_DOWNWARDS'}[state]

def new_chop_state(hand_history, CHOP_STATE, should_print=False):
    hand_dict = {0: 'wrist', 8: 'index_tip', 12: 'middle_tip', 16: 'ring_tip', 20: 'pinky_tip'}
    sum_hand_history = [
        tuple(
            sum(coords[i] for coords in (frame[finger_str] for finger_str in hand_dict.values())) / 5
            for i in range(3)
        )
        for frame in hand_history
    ]
    wrist_history = [frame['wrist'] for frame in hand_history]
    for history in [sum_hand_history, wrist_history]:
        old_x, old_y, old_z = history[-3]
        new_x, new_y, new_z = history[-1]

        dx = new_x - old_x
        dy = new_y - old_y
        dz = new_z - old_z
        speed = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        if should_print:
            print(f'state = {print_chop_state(CHOP_STATE)}, speed = {speed}, dy = {dy}, dx = {dx}')

        if CHOP_STATE == LIFTING_UPWARDS:
            CHOP_SPEED = 0.05
            if speed > CHOP_SPEED and abs(dy) > abs(dx) and dy > 0:

                NEW_STATE = CHOPPING_DOWNWARDS
                return NEW_STATE

        if CHOP_STATE == CHOPPING_DOWNWARDS:
            if dy <= 0:
                NEW_STATE = CHOPPED
                return NEW_STATE

        if CHOP_STATE == CHOPPED:
            chop_history_ints = [i for i in range(len(hand_history)) if hand_history[i]['is_chop_frame'] == True]
            chop_frames = [history[i] for i in chop_history_ints]
            LIFT_DIST = 0.10 # amount of distance need between the end point of a chop and when it has lifted
            LIFT_SPEED = 0.05
            if len(chop_frames) == 0:
                #NO CHOP FRAME IN HISTORY
                NEW_STATE = LIFTING_UPWARDS
                return NEW_STATE
            else: #THERE IS A CHOP FRAME
                chop_frame = chop_frames[-1]
                _, chop_y, _ = chop_frame
                if new_y >= chop_y + LIFT_DIST or dy <= LIFT_SPEED:
                    NEW_STATE = LIFTING_UPWARDS
                    return NEW_STATE

    return CHOP_STATE



def draw_taskbar(frame, value, max_value=30):
    h, w, _ = frame.shape

    # Position and size
    x = 50
    y = 50
    bar_width = 300
    bar_height = 30

    # Clamp value
    value = max(0, min(value, max_value))

    # Calculate filled width
    fill_width = int((value / max_value) * bar_width)

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
        f"{value}/{max_value}",
        (x + bar_width + 15, y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame