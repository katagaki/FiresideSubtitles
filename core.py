import json
import os
import wave
from typing import Any

import cv2
import moviepy.editor as movie_editor
import numpy as np
import requests
import torch
import whisper
from cv2 import VideoCapture, VideoWriter
from pyannote.audio import Pipeline
from pyannote.database.util import load_lab

from classes import FiresideSegment


def get_video_metadata(video_capture: VideoCapture) -> (float, float, float):
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return frames_per_second, video_width, video_height


def open_video_file(video_filename: str) -> VideoCapture:
    print("Opening video file...")
    video_capture = VideoCapture(video_filename)
    if not video_capture.isOpened():
        raise IOError(f"Could not open video file '{video_filename}'")
    frames_per_second, video_width, video_height = get_video_metadata(video_capture)
    print(f"Video framerate: {frames_per_second}")
    print(f"Video width: {video_width}")
    print(f"Video height: {video_height}")
    return video_capture


def extract_audio_from_video(video_filename: str, output_filename: str) -> int:
    if not os.path.exists(output_filename):
        video_clip = movie_editor.VideoFileClip(video_filename)
        video_clip.audio.write_audiofile(output_filename,
                                         bitrate="16k",
                                         ffmpeg_params=["-ac", "1"])
    audio_file = wave.open(output_filename, "rb")
    audio_sample_rate = audio_file.getframerate()
    audio_file.close()
    print(f"Audio sample rate: {audio_sample_rate}")
    return audio_sample_rate


def transcribe(audio_filename: str, output_filename: str) -> list[FiresideSegment]:
    print("Transcribing...")
    if not os.path.exists(output_filename):
        whisper_model = whisper.load_model(
            name="medium.en",
            device="cpu",
            download_root="models/whisper",
            in_memory=False
        )
        transcription = whisper_model.transcribe(audio_filename, verbose=True)
        with open(output_filename, "w") as json_file:
            json.dump(transcription, json_file, indent=4)
    else:
        with open(output_filename, "r") as json_file:
            transcription = json.load(json_file)
    transcription_segments: list[FiresideSegment] = []
    for segment in transcription["segments"]:
        transcription_segments.append(
            FiresideSegment(
                start=segment["start"],
                end=segment["end"],
                value=segment["text"]
            )
        )
    return transcription_segments


def diarize(audio_filename: str, output_filename: str, hugging_face_token: str) -> list[FiresideSegment]:
    print("Diarizing...")
    if not os.path.exists(output_filename):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hugging_face_token,
            cache_dir="models/pyannote"
        )
        pipeline.to(torch.device("mps"))
        diarization = pipeline(audio_filename)
        lab_string = diarization.to_lab()
        with open(output_filename, "w") as lab_file:
            lab_file.write(lab_string)
    else:
        diarization = load_lab(output_filename)
    diarization_segments: list[FiresideSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append(
            FiresideSegment(
                start=turn.start,
                end=turn.end,
                value=speaker
            )
        )
    return diarization_segments


def update_frames(
        video_capture: VideoCapture,
        detect_faces: bool = True,
        transcription_segments: list[FiresideSegment] = None,
        diarization_segments: list[FiresideSegment] = None
) -> list[Any]:
    # Initialize face detection model
    if detect_faces:
        print("Initializing face detection...")
        required_files = [
            ("models/dnn/deploy.prototxt",
             "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/deploy.prototxt"),
            ("models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel",
             "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        ]
        if not os.path.exists("models/dnn"):
            os.mkdir("models/dnn")
        for (required_file_filename, required_file_url) in required_files:
            if not os.path.exists(required_file_filename):
                data = requests.get(
                    url=required_file_url,
                    allow_redirects=True
                )
                with open(required_file_filename, "wb") as required_file:
                    required_file.write(data.content)
        face_detection_model = cv2.dnn.readNetFromCaffe(
            "models/dnn/deploy.prototxt",
            "models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
    else:
        face_detection_model = None

    # Do face detection and draw frames
    frames = []
    frames_per_second, _, video_height = get_video_metadata(video_capture)
    frame_count: int = 0
    current_text: str = ""
    current_speaker: str = ""
    print("Drawing new frames...")
    while True:
        is_frame_read_successfully, frame = video_capture.read()
        if not is_frame_read_successfully:
            break

        current_time = frame_count / frames_per_second

        # Detect faces
        if face_detection_model:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detection_model.setInput(blob)
            detections = face_detection_model.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Get transcribed text
        if transcription_segments:
            for segment in transcription_segments:
                if segment.start <= current_time <= segment.end:
                    current_text = segment.value
                    break
                else:
                    current_text = ""

        # Get diarized speaker
        if diarization_segments:
            for segment in diarization_segments:
                if segment.start <= current_time <= segment.end:
                    current_speaker = segment.value
                    break
                else:
                    current_speaker = ""

        # Determine the subtitle to show
        if current_speaker != "" and current_text != "":
            current_subtitle = f"{current_speaker}: {current_text}"
        elif diarization_segments is None and current_text != "":
            current_subtitle = current_text
        else:
            current_subtitle = ""

        # Put subtitle on frame
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

        frames.append(frame)
        frame_count += 1
    return frames


def export_video(video_capture: VideoCapture, frames: list[Any], output_filename: str):
    print("Exporting frames to video...")
    frames_per_second, video_width, video_height = get_video_metadata(video_capture)
    video_writer = VideoWriter(
        filename=output_filename,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=frames_per_second,
        frameSize=(int(video_width), int(video_height))
    )
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def replace_audio_in_video(video_filename: str, audio_filename: str, output_filename: str):
    print("Combining video and audio...")
    output_video_clip = movie_editor.VideoFileClip(video_filename)
    original_video_audio = movie_editor.AudioFileClip(audio_filename)
    output_video_with_original_video_audio = output_video_clip.set_audio(original_video_audio)
    output_video_with_original_video_audio.write_videofile(
        filename=output_filename,
        codec="libx264",
        audio_codec="pcm_s16le"
    )
