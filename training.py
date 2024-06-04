import os
import pickle

import cv2
import face_recognition
from imutils import paths


def train_face_model(images_path: str, output_model_filename: str):
    images = list(paths.list_images(images_path))

    known_encodings = []
    known_names = []
    for (index, image_path) in enumerate(images):
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)

        image_in_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        identified_face_locations = face_recognition.face_locations(image_in_rgb, model="cnn")
        print(f"Found {len(identified_face_locations)} face(s) in {image_path}")

        encodings = face_recognition.face_encodings(image_in_rgb, identified_face_locations)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    data = {
        "encodings": known_encodings,
        "names": known_names
    }
    with open(output_model_filename, "wb") as f:
        f.write(pickle.dumps(data))


if __name__ == "__main__":
    train_face_model(
        images_path="models/faces",
        output_model_filename="models/faces/faces.pickle"
    )
