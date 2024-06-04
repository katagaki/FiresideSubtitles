import os
from typing import Any

import cv2
from cv2 import VideoCapture, VideoWriter
from moviepy import editor as movie_editor


def get_video_metadata(video_capture: VideoCapture) -> (float, float, float):
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return frames_per_second, video_width, video_height


def open_video_file(video_filename: str) -> VideoCapture:
    video_capture = VideoCapture(video_filename)
    if not video_capture.isOpened():
        raise IOError(f"Could not open video file '{video_filename}'")
    return video_capture


def extract_audio_from_video(video_filename: str, output_filename: str):
    if not os.path.exists(output_filename):
        video_clip = movie_editor.VideoFileClip(video_filename)
        video_clip.audio.write_audiofile(output_filename,
                                         bitrate="16k",
                                         ffmpeg_params=["-ac", "1"])


def export_video(video_capture: VideoCapture, frames: list[Any], output_filename: str):
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
    output_video_clip = movie_editor.VideoFileClip(video_filename)
    original_video_audio = movie_editor.AudioFileClip(audio_filename)
    output_video_with_original_video_audio = output_video_clip.set_audio(original_video_audio)
    output_video_with_original_video_audio.write_videofile(
        filename=output_filename,
        codec="libx264",
        audio_codec="pcm_s16le"
    )
