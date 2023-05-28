# activenessnet.py

import random 
import cv2
import imutils
import f_liveness_detection
import questions
import streamlit as st
import time

def activenessnet():
    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0 # Inisasi variabel untuk jumlah perintah yang berhasil dijalankan
    counter_ok_consecutives = 0
    limit_consecutives = 1
    limit_questions = 6  # Jumlah maksimal perintah
    counter_try = 0
    limit_try = 40  

    def show_image(cam,text,color = (0,0,255)):
        ret, im = cam.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        im = imutils.resize(im, width=720)
        cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
        return im

    num = list(range(0,limit_questions))
    random.shuffle(num)

    liveness = {}

    result_placeholder = st.empty()

    stframe = st.empty()

    # cv2.destroyAllWindows()

    stframe.empty()
    cam = cv2.VideoCapture(0)
    for i_questions in num:
        question = questions.question_bank(i_questions)
        # st.write("Question: ", question)

        for i_try in range(limit_try):
            ret, im = cam.read()
            im = imutils.resize(im, width=720)
            im = cv2.flip(im, 1)
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL_0)
            TOTAL = out_model['total_blinks']
            COUNTER = out_model['count_blinks_consecutives']
            dif_blink = TOTAL-TOTAL_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0

            challenge_res = questions.challenge_result(question, out_model,blinks_up)

            im = show_image(cam,question)
            stframe.image(im)

            if challenge_res == "pass":
                counter_ok_consecutives += 1
                if counter_ok_consecutives == limit_consecutives:
                    counter_ok_questions += 1
                    counter_try = 0
                    counter_ok_consecutives = 0
                    break
                else:
                    continue

            elif challenge_res == "fail":
                counter_try += 1
            elif i_try == limit_try-1:
                break

        if counter_ok_questions ==  limit_questions:
            im = show_image(cam,"Real",color = (0,255,0))
            stframe.image(im)
            time.sleep(2)
            # cv2.destroyAllWindows()

            result_placeholder.success("Anda dinyatakan real")
            liveness['label'] = "Real"
            break 

        elif i_try == limit_try-1:
            im = show_image(cam,"Fake", color = (255,0,0))
            stframe.image(im)
            time.sleep(2)
            # cv2.destroyAllWindows()

            result_placeholder.error("Anda tidak mengikuti semua perintah dengan benar. Silakan coba lagi")
            liveness['label'] = "Fake"
            break 

        else:
            continue

    cam.release()
    # cv2.destroyAllWindows()
    stframe.empty()