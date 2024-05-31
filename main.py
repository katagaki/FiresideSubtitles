import os

from core import (
    open_video_file,
    extract_audio_from_video,
    transcribe,
    diarize,
    update_frames,
    export_video,
    replace_audio_in_video
)

if __name__ == "__main__":
    hugging_face_token = os.environ["HUGGING_FACE_TOKEN"]
    filename = "Kitchen Nightmare's Most Ridiculous Moments"
    input_video_filename = f"media/input/{filename}.mp4"
    input_audio_filename = f"media/audio/{filename}.wav"
    transcript_filename = f"media/transcripts/{filename}.whs"
    diarization_filename = f"media/transcripts/{filename}.lab"
    audioless_output_video_filename = f"media/output/.{filename}.mp4"
    final_output_video_filename = f"media/output/{filename}.mp4"

    # Open video file
    video_capture = open_video_file(
        video_filename=input_video_filename
    )

    # Get audio track of video file
    audio_sample_rate = extract_audio_from_video(
        video_filename=input_video_filename,
        output_filename=input_audio_filename
    )

    # Get transcription from OpenAI Whisper
    transcription_segments = transcribe(
        audio_filename=input_audio_filename,
        output_filename=transcript_filename
    )

    # Diarize audio file
    diarization_segments = diarize(
        audio_filename=input_audio_filename,
        output_filename=diarization_filename,
        hugging_face_token=hugging_face_token
    )

    # Draw frames
    frames = update_frames(
        video_capture=video_capture,
        detect_faces=True,
        transcription_segments=transcription_segments,
        diarization_segments=diarization_segments
    )

    # Output frames
    export_video(
        video_capture=video_capture,
        frames=frames,
        output_filename=audioless_output_video_filename
    )

    # Clean up video capture
    video_capture.release()

    # Add audio file to output video
    replace_audio_in_video(
        video_filename=audioless_output_video_filename,
        audio_filename=input_audio_filename,
        output_filename=final_output_video_filename
    )

    exit(0)
