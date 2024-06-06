import os
import pickle
import wave

from dotenv import load_dotenv

from core.classes import FiresideFaceToSpeakerMapping
from core.drawing import (
    scale_down_frame_if_larger_than_720p,
    show_frame,
    did_user_send_cancel_signal,
    close_frame_preview
)
from core.faces import get_face_detection_model, get_face_recognition_model, extract_faces_with_names, label_faces
from core.speech import transcribe, diarize
from core.subtitles import draw_subtitles, segment_value_for_current_time
from core.videos import (
    get_video_framerate,
    get_video_width,
    get_video_height,
    open_video_file,
    extract_audio_from_video,
    export_video,
    replace_audio_in_video
)
from training import train_face_model

load_dotenv()

if __name__ == "__main__":

    if os.environ.get("SHOULD_TRAIN_FACES_BEFORE_EXECUTION", "0") == "1":
        train_face_model()

    hugging_face_token = os.environ["HUGGING_FACE_TOKEN"]
    filename = os.environ["FILENAME_TO_PROCESS"]
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

    video_framerate = get_video_framerate(video_capture)
    video_width = get_video_width(video_capture)
    video_height = get_video_height(video_capture)
    print(f"Video framerate: {video_framerate}")
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

    print("Initializing face detection...")
    face_detection_model = get_face_detection_model()

    print("Initializing face recognition...")
    face_recognition_model = get_face_recognition_model()
    with open("models/faces/faces.pickle", "rb") as encodings_file:
        face_encoding_mappings = pickle.loads(encodings_file.read())

    print("Drawing new frames", end="")
    face_to_speaker_mappings: list[FiresideFaceToSpeakerMapping] = []
    frames: list = []
    video_framerate = get_video_framerate(video_capture)
    frame_number: int = 0
    is_frame_read_successfully: bool = True
    should_break_out_of_loop: bool = False
    while is_frame_read_successfully and not should_break_out_of_loop:
        is_frame_read_successfully, frame = video_capture.read()

        if is_frame_read_successfully:
            frame = scale_down_frame_if_larger_than_720p(frame)

            faces = extract_faces_with_names(
                frame=frame,
                face_recognition_model=face_recognition_model,
                face_encoding_mappings=face_encoding_mappings
            )
            faces = [face for face in faces if face.name != "Unknown"]
            label_faces(
                frame=frame,
                faces=faces
            )

            current_time = frame_number / video_framerate
            current_text = segment_value_for_current_time(current_time, transcription_segments)
            current_speaker = segment_value_for_current_time(current_time, diarization_segments)

            # Get mapping
            existing_face_to_speaker_mappings = [
                mapping for mapping in face_to_speaker_mappings if mapping.speaker_name == current_speaker
            ]
            if len(existing_face_to_speaker_mappings) == 1:
                current_speaker = existing_face_to_speaker_mappings[0].person_name
            elif len(faces) == 1:
                face_to_speaker_mappings.append(FiresideFaceToSpeakerMapping(
                    speaker_name=current_speaker,
                    person_name=faces[0].name
                ))
                current_speaker = faces[0].name
            else:
                current_speaker = None

            draw_subtitles(
                frame=frame,
                speaker_name=current_speaker,
                text=current_text
            )

            frames.append(frame)

            if os.environ.get("SHOULD_SHOW_PREVIEWS", "0") == "1":
                show_frame(frame)
                should_break_out_of_loop = did_user_send_cancel_signal()

            frame_number += 1
            print(".", end="")
    print("")

    close_frame_preview()

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
