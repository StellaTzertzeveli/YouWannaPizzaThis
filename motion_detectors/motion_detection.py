import cv2
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