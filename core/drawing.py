from typing import Any

import cv2
from cv2 import VideoCapture

from classes import FiresideSegment
from core.faces import (
    get_face_detection_model,
    get_face_recognition_model,
    highlight_faces,
    label_faces
)
from core.videos import get_video_metadata


def draw_subtitles(
        frame: Any,
        current_time: float,
        transcription_segments: list[FiresideSegment] = None,
        diarization_segments: list[FiresideSegment] = None,
        video_height: int = 0
) -> None:
    current_text: str = ""
    current_speaker: str = ""

    if transcription_segments:
        for segment in transcription_segments:
            if segment.start <= current_time <= segment.end:
                current_text = segment.value
                break
            else:
                current_text = ""

    if diarization_segments:
        for segment in diarization_segments:
            if segment.start <= current_time <= segment.end:
                current_speaker = segment.value
                break
            else:
                current_speaker = ""

    if current_speaker != "" and current_text != "":
        current_subtitle = f"{current_speaker}: {current_text}"
    elif diarization_segments is None and current_text != "":
        current_subtitle = current_text
    else:
        current_subtitle = ""

    if current_subtitle != "":
        cv2.putText(
            img=frame,
            text=current_subtitle,
            org=(21, int(video_height) - 19),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(18, 18, 18),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            img=frame,
            text=current_subtitle,
            org=(20, int(video_height) - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )


def update_frames(
        video_capture: VideoCapture,
        should_highlight_faces: bool = True,
        should_label_faces: bool = True,
        transcription_segments: list[FiresideSegment] = None,
        diarization_segments: list[FiresideSegment] = None
) -> list[Any]:
    if should_highlight_faces:
        print("Initializing face detection...")
        face_detection_model = get_face_detection_model()
    else:
        face_detection_model = None

    if should_label_faces:
        print("Initializing face recognition...")
        face_recognition_model = get_face_recognition_model()
    else:
        face_recognition_model = None

    frames = []
    frames_per_second, _, video_height = get_video_metadata(video_capture)
    frame_count: int = 0
    print("Drawing new frames...")
    while True:
        is_frame_read_successfully, frame = video_capture.read()
        if not is_frame_read_successfully:
            break

        if face_detection_model:
            highlight_faces(frame, face_detection_model)

        if face_recognition_model:
            label_faces(frame, face_recognition_model)

        draw_subtitles(
            frame=frame,
            current_time=frame_count / frames_per_second,
            transcription_segments=transcription_segments,
            diarization_segments=diarization_segments,
            video_height=video_height
        )

        frames.append(frame)
        frame_count += 1
        print(".", end="")

    print("")
    return frames
