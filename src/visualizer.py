import cv2

COLOR = (0, 200, 255)
BOX_COLOR = (55, 55, 55)
FONT = cv2.FONT_HERSHEY_DUPLEX
THICKNESS = 1
PADDING = 4


def _put_text_box(frame, text, x, y, font_scale=0.6):
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, THICKNESS)
    top = y - th - PADDING
    left = x
    bottom = y + baseline + PADDING
    right = x + tw + PADDING * 2
    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, -1)
    cv2.putText(
        frame,
        text,
        (x + PADDING, y),
        FONT,
        font_scale,
        COLOR,
        THICKNESS,
        cv2.LINE_AA,
    )


def draw_tracks(frame, tracks):
    for tid, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2, cv2.LINE_AA)
        label_y = max(y1 - 6, 20)
        _put_text_box(frame, f"#{tid}", x1, label_y, font_scale=0.5)

    _put_text_box(frame, f"Drohnen: {len(tracks)}", 10, 34, font_scale=0.65)
    return frame
