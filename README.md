# FiresideSubtitles

A simple Python project that provides transcription, speaker diarization, and face detection in a simple package.

More to come!

## Preparing Environment

1. Prepare the `.env` file in the root of the cloned project with the following keys:
    - `HUGGING_FACE_TOKEN`: Specifies the [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) to use
    - `SHOULD_TRAIN_FACES_BEFORE_EXECUTION`: Specifies whether face recognition data should be retrained before execution
    - `SHOULD_SHOW_PREVIEWS`: Specifies whether a preview of the current frame will be displayed during processing
    - `FILENAME_TO_PROCESS`: Specifies the filename to process, without its extension

    ```dotenv
    HUGGING_FACE_TOKEN=xxxxx

    SHOULD_TRAIN_FACES_BEFORE_EXECUTION=1
    SHOULD_SHOW_PREVIEWS=1

    FILENAME_TO_PROCESS="Rick Astley - Never Gonna Give You Up"
    ```

2. Accept the conditions of use for the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) pipeline on Hugging Face.
3. Install [Homebrew](https://brew.sh), and run `brew install ffmpeg` in your Terminal.
4. Run `pip install -r requirements.txt` in your Terminal, from the project root.

## Preparing Video Input

1. Create a `media` folder in the project root.
2. In the `media` folder, create an `input` folder, and put your video files in that folder.

## Preparing Face Recognition Data

If you are using face recognition (`label_faces`), you will need to prepare a set of embeddings to use.

1. Create a `models` folder in the project root.
2. In the `models` folder, create a `faces` folder.
3. In the `faces` folder, insert photos of people you want to identify, separated by folders based on the person.
4. Run `training.py` to generate the embeddings required for face recognition.

## Running

1. Run `main.py`.
2. Open the `output` folder in the `media` folder to see the generated output.
