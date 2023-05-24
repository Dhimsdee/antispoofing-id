# f_liveness_detection.py

import cv2
import imutils
import f_utils
import dlib
import numpy as np
from profile_detection import f_detector
from emotion_detection import f_emotion_detection
from blink_detection import f_blink_detection

frontal_face_detector    = dlib.get_frontal_face_detector()
profile_detector         = f_detector.detect_face_orientation()
emotion_detector         = f_emotion_detection.predict_emotions()
blink_detector           = f_blink_detection.eye_blink_detector() 


def detect_liveness(im,COUNTER=0,TOTAL=0):
    gray = gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # face detection
    rectangles = frontal_face_detector(gray, 0)
    boxes_face = f_utils.convert_rectangles2array(rectangles,im)
    if len(boxes_face)!=0:
        areas = f_utils.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = [list(boxes_face[index])]

        # emotion detection
        _,emotion = emotion_detector.get_emotion(im,boxes_face)

        # blink detection
        COUNTER,TOTAL = blink_detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
    else:
        boxes_face = []
        emotion = []
        TOTAL = 0
        COUNTER = 0

    # profile detection
    box_orientation, orientation = profile_detector.face_orientation(gray)

    # output
    output = {
        'box_face_frontal': boxes_face,
        'box_orientation': box_orientation,
        'emotion': emotion,
        'orientation': orientation,
        'total_blinks': TOTAL,
        'count_blinks_consecutives': COUNTER
    }
    return output

