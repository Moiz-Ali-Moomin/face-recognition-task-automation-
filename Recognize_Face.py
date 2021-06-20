#STEP 3  RUN FACIAL RECOGNITION
from TrainModel import face1_model, face2_model 
import cv2
import numpy as np
import os
def confidence(results, image):
    if results[1] < 500 :
        confidence = int( 100 * (1 - (results[1])/400) )
        display_string = str(confidence) + '% Confident it is User'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
    return confidence

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is () :
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        a_results = face1_model.predict(face)
        f_results = face2_model.predict(face)
        
        c_face1= confidence(a_results, image)
        c_face2 = confidence(f_results, image)

        if c_face1 > 90:
            cv2.putText(image, "face1", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            # If it is face1 send mail
            try:
                from sendmail import mail
                mail("youmail", "Hey Boss, your face detected in camera!")
                from whatsappmsg import send_whatsApp_msg
                send_whatsApp_msg("yourno", "Hello face2, How are you?")
                break
            except: 
                print("Allow less secure apps in Email settings! ")
                break
         
        elif c_face2 > 90:
            cv2.putText(image, "face2", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            # If it is face2 it it go terraform & do task
            try:
                os.system("terraform init ")
                os.system("terraform apply --auto-approve")
                break
            except:
                print("Error in terraform file")
                break

        else:    
            cv2.putText(image, "Rcognizing Face...", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
        
    except:
        cv2.putText(image, "No Face Found!", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Looking For Face...", (220, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      