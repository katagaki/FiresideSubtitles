import os
import pickle
from typing import Any

import cv2
import face_recognition
import numpy as np
import requests


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


def highlight_faces(frame: Any, face_detection_model: Any):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            src=frame,
            dsize=(300, 300)
        ),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            overlay_box = np.zeros_like(frame, np.uint8)

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(
                img=overlay_box,
                pt1=(startX, startY),
                pt2=(endX, endY),
                color=(255, 0, 0),
                thickness=cv2.FILLED
            )

            alpha = 0.8
            mask = overlay_box.astype(bool)
            frame[mask] = cv2.addWeighted(frame, alpha, overlay_box, 1 - alpha, 0)[mask]


def label_faces(frame: Any, face_recognition_model: Any):
    with open("models/faces/faces.pickle", "rb") as encodings_file:
        data = pickle.loads(encodings_file.read())

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
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings=data["encodings"],
            face_encoding_to_check=encoding,
            tolerance=0.4
        )
        name = "Unknown"

        if True in matches:
            matched_face_indexes = [index for (index, b) in enumerate(matches) if b]
            possible_matches = {}
            for index in matched_face_indexes:
                name = data["names"][index]
                possible_matches[name] = possible_matches.get(name, 0) + 1
            name = max(possible_matches, key=possible_matches.get)
            if current_name != name:
                current_name = name

        names.append(name)

    for ((top, right, bottom, left), name) in zip(detected_face_boxes, names):
        cv2.rectangle(
            img=frame,
            pt1=(left, top),
            pt2=(right, bottom),
            color=(0, 0, 225),
            thickness=2
        )
        cv2.putText(
            img=frame,
            text=name,
            org=(left, top - 15 if top - 15 > 15 else top + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.8,
            color=(0, 0, 255),
            thickness=2
        )

    return frame
