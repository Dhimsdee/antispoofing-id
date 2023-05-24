import streamlit as st
from imutils.video import VideoStream
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

def serial_detection():
    # Set up the argument parser
    model_path = 'livenessnet/LivenessNet.model'
    le_path = 'livenessnet/le.pickle'
    detector_folder = 'livenessnet/face_detector'
    confidence = 0.5
    args = {'model': model_path, 'le': le_path, 'detector': detector_folder,
            'confidence': confidence}

    result_placeholder = st.empty()
    stframe = st.empty()

    # Load the face detector
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the model and label encoder
    model = load_model(args["model"])
    le = pickle.loads(open(args["le"], "rb").read())

    # Start the video stream
    v = cv2.VideoCapture(0)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    sequence_count_real = 0
    sequence_count_fake = 0
    label = 'fake'
    is_passive_real = False

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=720)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        num_faces = 0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                num_faces += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if num_faces == 1:
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]

                    label = "{}: {:.4f}".format(label, preds[j])
                    print(label)

                    if preds[j] > 0.60 and j == 1:
                        sequence_count_real += 1
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        if sequence_count_real >= 10:
                            is_passive_real = True
                            break
                    else:
                        sequence_count_fake += 1
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        if sequence_count_fake >= 10:
                            break

                else:
                    cv2.putText(frame, "Ensure only 1 face in the frame", (20, 35),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    print("Error frame")
                    sequence_count_fake = 0
                    sequence_count_real = 0

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if is_passive_real:
            # result_placeholder.success('You are considered real')
            label = "real"
            time.sleep(2)
            break
        elif sequence_count_fake >= 10:
            # result_placeholder.error('You are considered fake. Please try again')
            label = "fake"
            time.sleep(2)
            break

    cv2.destroyAllWindows()
    vs.stop()
    stframe.empty()

    return label


