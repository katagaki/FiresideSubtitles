import json
import os

import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.database.util import load_lab

from core.classes import FiresideSegment
from core.io import create_folder_for_file


def transcribe(audio_filename: str, output_filename: str) -> list[FiresideSegment]:
    create_folder_for_file(output_filename)
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
