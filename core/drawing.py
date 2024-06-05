from typing import Any

import cv2


def scale_down_frame_if_larger_than_720p(frame) -> Any:
    if frame.shape[0] > 720 or frame.shape[1] > 1280:
        scale_percent = 720 / max(frame.shape[0], frame.shape[1])
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame
    else:
        return frame


def show_frame(frame, window_title: str = "Frame Preview") -> bool:
    cv2.imshow(window_title, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    return False


def close_frame_preview():
    cv2.destroyAllWindows()
