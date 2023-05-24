import random 
import cv2
import imutils
import f_liveness_detection
import questions

def capture_frame():
    cv2.namedWindow('liveness_detection')
    cam = cv2.VideoCapture(0)
    return cam

def activenessnet(cam): 
    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0 # Inisasi variabel untuk jumlah perintah yang berhasil dijalankan
    counter_ok_consecutives = 0
    limit_consecutives = 3
    limit_questions = 6 # Jumlah maksimal perintah
    counter_try = 0
    limit_try = 30 


    def show_image(cam,text,color = (0,0,255)):
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_DUPLEX,0.75,color,2)
        return im

    num = list(range(0,limit_questions))
    random.shuffle(num)
    
    liveness = {}

    for i_questions in num:
        question = questions.question_bank(i_questions)
        print(question)

        im = show_image(cam,question,color = (0,255,0))
        cv2.imshow('liveness_detection',im)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

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

            im = show_image(cam,question,color = (0,255,0))
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break 

            if challenge_res == "pass":
                # im = show_image(cam,question+" : ok")
                cv2.imshow('liveness_detection',im)
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break

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
                show_image(cam,question+" : fail")
            elif i_try == limit_try-1:
                break
                
        if counter_ok_questions ==  limit_questions:
            while True:
                im = show_image(cam,"Anda dinyatakan real",color = (0,255,0))
                cv2.imshow('liveness_detection',im)
                if cv2.waitKey(3000):
                    break
            cv2.destroyAllWindows()
                
            liveness['label'] = "Real"
            
        elif i_try == limit_try-1:
            while True:
                im = show_image(cam,"Silakan coba lagi")
                cv2.imshow('liveness_detection',im)
                if cv2.waitKey(3000):
                    break
            cv2.destroyAllWindows()
                
            liveness['label'] = "Fake"
            break 

        else:
            continue

    return liveness

# ini berhasil tapi gaada video streamnya
""" import random 
import cv2
import imutils
import f_liveness_detection
import questions

def activenessnet(cam): 
    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0 # Inisasi variabel untuk jumlah perintah yang berhasil dijalankan
    counter_ok_consecutives = 0
    limit_consecutives = 3
    limit_questions = 6 # Jumlah maksimal perintah
    counter_try = 0
    limit_try = 30 

    num = list(range(0,limit_questions))
    random.shuffle(num)
    
    liveness = {}

    for i_questions in num:
        question = questions.question_bank(i_questions)
        print(question)

        for i_try in range(limit_try):
            ret, im = cam.read()
            if im is None:
                continue
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
            liveness['label'] = "Real"
            break 

        elif i_try == limit_try-1:
            liveness['label'] = "Fake"
            break 

        else:
            continue

    return liveness  """

# aktif.py socket.io
""" import random 
import cv2
import imutils
import f_liveness_detection
import questions

def activenessnet(cam): 
    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0 # Inisasi variabel untuk jumlah perintah yang berhasil dijalankan
    counter_ok_consecutives = 0
    limit_consecutives = 3
    limit_questions = 6 # Jumlah maksimal perintah
    counter_try = 0
    limit_try = 30 

    num = list(range(0,limit_questions))
    random.shuffle(num)
    
    liveness = {}

    for i_questions in num:
        question = questions.question_bank(i_questions)
        print(question)

        for i_try in range(limit_try):
            ret, im = cam.read()
            if im is None:
                continue
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
            liveness['label'] = "Real"
            break 

        elif i_try == limit_try-1:
            liveness['label'] = "Fake"
            break 

        else:
            continue

    return liveness
 """