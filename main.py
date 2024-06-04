import os
import wave

from dotenv import load_dotenv

from core.drawing import update_frames
from core.speech import transcribe, diarize
from core.videos import (
    open_video_file,
    extract_audio_from_video,
    export_video,
    replace_audio_in_video,
    get_video_metadata
)

load_dotenv()

if __name__ == "__main__":
    hugging_face_token = os.environ["HUGGING_FACE_TOKEN"]
    filename = "Rick Astley - Never Gonna Give You Up"
    input_video_filename = f"media/input/{filename}.mp4"
    input_audio_filename = f"media/audio/{filename}.wav"
    transcript_filename = f"media/transcripts/{filename}.whs"
    diarization_filename = f"media/transcripts/{filename}.lab"
    audioless_output_video_filename = f"media/output/.{filename}.mp4"
    final_output_video_filename = f"media/output/{filename}.mp4"

    print("Opening video file...")
    video_capture = open_video_file(
        video_filename=input_video_filename
    )
    frames_per_second, video_width, video_height = get_video_metadata(video_capture)
    print(f"Video framerate: {frames_per_second}")
    print(f"Video width: {video_width}")
    print(f"Video height: {video_height}")

    print("Extracting audio from video...")
    extract_audio_from_video(
        video_filename=input_video_filename,
        output_filename=input_audio_filename
    )
    audio_file = wave.open(input_audio_filename, "rb")
    audio_sample_rate = audio_file.getframerate()
    audio_file.close()
    print(f"Audio sample rate: {audio_sample_rate}")

    print("Transcribing...")
    transcription_segments = transcribe(
        audio_filename=input_audio_filename,
        output_filename=transcript_filename
    )

    print("Diarizing...")
    diarization_segments = diarize(
        audio_filename=input_audio_filename,
        output_filename=diarization_filename,
        hugging_face_token=hugging_face_token
    )

    frames = update_frames(
        video_capture=video_capture,
        should_highlight_faces=True,
        should_label_faces=True,
        transcription_segments=transcription_segments,
        diarization_segments=diarization_segments,
        should_show_preview=True
    )

    print("Exporting frames to video...")
    export_video(
        video_capture=video_capture,
        frames=frames,
        output_filename=audioless_output_video_filename
    )

    video_capture.release()

    print("Combining video and audio...")
    replace_audio_in_video(
        video_filename=audioless_output_video_filename,
        audio_filename=input_audio_filename,
        output_filename=final_output_video_filename
    )

    exit(0)
