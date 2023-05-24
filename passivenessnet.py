# passivenessnet.py

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

def detect_passive_liveness():
    
    # Set up the argument parser
    model_path='livenessnet/LivenessNet.model'
    le_path='livenessnet/le.pickle'
    detector_folder='livenessnet/face_detector'
    confidence=0.5
    args = {'model':model_path, 'le':le_path, 'detector':detector_folder,
             'confidence':confidence}
    
    result_placeholder = st.empty()
    stframe = st.empty()
    
    # if st.button("Detect Liveness"):
    # Load the face detector
    #st.write("Loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the model and label encoder
    #st.write("Loading Model...")
    model = load_model(args["model"])
    le = pickle.loads(open(args["le"], "rb").read())

    # Start the video stream
    v = VideoStream(src=0).start()
    time.sleep(2.0)
    sequence_count_real = 0
    sequence_count_fake = 0
    label ='fake'

    # Loop through each frame in the video stream
    while True:
        frame = v.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
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

                    # Draw the rectangle and label on the frame
                    if preds[j] > 0.60 and j == 1:
                        sequence_count_real += 1
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    else:
                        sequence_count_fake += 1
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                
                else:
                    cv2.putText(frame, "Pastikan hanya ada 1 wajah di frame", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    print("Error frame")
                    sequence_count_fake = 0
                    sequence_count_real = 0
                        
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if sequence_count_real == 10 and sequence_count_fake < 5:
            result_placeholder.success('Anda dinyatakan real')
            label = "real"
            time.sleep(2)
            break
        elif sequence_count_real == 10 and sequence_count_fake >= 5:
            result_placeholder.error('Silakan coba lagi')
            label = "fake"
            time.sleep(2)
            break
        elif sequence_count_fake == 10:
            result_placeholder.error('Anda dinyatakan fake. Silakan coba lagi')
            label = "fake"
            time.sleep(2)
            break

    cv2.destroyAllWindows()
    v.stop()
    stframe.empty()

    return label