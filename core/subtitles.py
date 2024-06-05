from typing import Any

import cv2

from core.classes import FiresideSegment


def segment_value_for_current_time(current_time: float, segments: list[FiresideSegment]) -> str | None:
    if segments is None:
        return None
    else:
        for index, segment in enumerate(segments):
            if segment.start <= current_time <= segment.end:
                return segment.value
            elif segment.start > current_time:
                if segment.value == segments[index - 1].value:
                    return segment.value
                else:
                    return None


def concatenate_subtitles(speaker_name: str = None, text: str = None):
    if text is None:
        return ""
    else:
        if speaker_name is None:
            return text
        else:
            return f"{speaker_name}: {text}"


def draw_subtitles(frame: Any, speaker_name: str, text: str):
    video_height = frame.shape[0]
    video_width = frame.shape[1]

    current_subtitle = concatenate_subtitles(speaker_name, text)

    if current_subtitle != "":
        cv2.putText(
            img=frame,
            text=current_subtitle,
            org=(21, int(video_height) - 19),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(18, 18, 18),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            img=frame,
            text=current_subtitle,
            org=(20, int(video_height) - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )
