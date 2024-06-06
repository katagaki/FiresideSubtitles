import os
from typing import Any

import cv2
import face_recognition
import numpy as np
import requests

from core.classes import FiresideFace


def download_file(file_url: str, save_path: str):
    data = requests.get(
        url=file_url,
        allow_redirects=True
    )
    with open(save_path, "wb") as required_file:
        required_file.write(data.content)


def get_face_detection_model():
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
            download_file(required_file_url, required_file_filename)
    face_detection_model = cv2.dnn.readNetFromCaffe(
        "models/dnn/deploy.prototxt",
        "models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )
    return face_detection_model


def get_face_recognition_model():
    required_files = [
        ("models/faces/haarcascade_frontalface_default.xml",
         "https://github.com/kipr/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml")
    ]
    if not os.path.exists("models/faces"):
        os.mkdir("models/faces")
    for (required_file_filename, required_file_url) in required_files:
        if not os.path.exists(required_file_filename):
            download_file(required_file_url, required_file_filename)
    face_recognition_model = cv2.CascadeClassifier(
        "models/faces/haarcascade_frontalface_default.xml"
    )
    return face_recognition_model


def extract_faces_without_names(frame: Any, face_detection_model: Any) -> list[FiresideFace]:
    height = frame.shape[0]
    width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            src=frame,
            dsize=(400, 400)
        ),
        scalefactor=0.7,
        size=(400, 400),
        mean=(104.0, 177.0, 123.0)
    )
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    found_faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            overlay_box = np.zeros_like(frame, np.uint8)

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            found_faces.append(FiresideFace(
                name="",
                x_start=start_x,
                x_end=end_x,
                y_start=start_y,
                y_end=end_y
            ))

    return found_faces


def extract_faces_with_names(
        frame: Any,
        face_recognition_model: Any,
        face_encoding_mappings: Any
) -> list[FiresideFace]:
    current_name = "Unknown"

    detected_faces = face_recognition_model.detectMultiScale(
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_DO_ROUGH_SEARCH
    )
    detected_face_boxes = [(y, x + w, y + h, x) for (x, y, w, h) in detected_faces]

    encodings = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detected_face_boxes)

    found_faces = []
    for index, encoding in enumerate(encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings=face_encoding_mappings["encodings"],
            face_encoding_to_check=encoding,
            tolerance=0.4
        )

        name = "Unknown"
        if True in matches:
            matched_face_indexes = [index for (index, b) in enumerate(matches) if b]
            possible_matches = {}
            for index in matched_face_indexes:
                name = face_encoding_mappings["names"][index]
                possible_matches[name] = possible_matches.get(name, 0) + 1
            name = max(possible_matches, key=possible_matches.get)
            if current_name != name:
                current_name = name

        for top, right, bottom, left in detected_face_boxes:
            found_faces.append(FiresideFace(
                name=name,
                x_start=left,
                x_end=right,
                y_start=top,
                y_end=bottom
            ))

    return found_faces


def label_faces(frame: Any, faces: list[FiresideFace]):
    for face in faces:
        if face.name != "Unknown":
            cv2.rectangle(
                img=frame,
                pt1=(face.x_start, face.y_start),
                pt2=(face.x_end, face.y_end),
                color=(0, 0, 225),
                thickness=2
            )
            cv2.putText(
                img=frame,
                text=face.name,
                org=(face.x_start, face.y_start - 15 if face.y_start - 15 > 15 else face.y_start + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.8,
                color=(0, 0, 255),
                thickness=2
            )
