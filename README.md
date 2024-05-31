# FiresideSubtitles

A simple Python project that provides transcription, speaker diarization, and face detection in a simple package.

More to come!

## Running

1. Create a .env file in the root of the cloned project, and add an environment variable `HUGGING_FACE_TOKEN` with your [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) as the value.
2. Accept the conditions of use for the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) pipeline on Hugging Face.
3. Create a `media` folder in the project root.
4. In the `media` folder, create an `input` folder, and put your video file in that folder.
5. Change the `filename` variable in `main.py` to point to the video file you want to process, excluding the file extension.
6. Run `main.py`.
7. Open the `output` folder in the `media` folder to see the generated output.
