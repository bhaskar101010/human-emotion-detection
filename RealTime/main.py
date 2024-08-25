#Importing required packages and dependencies
import numpy as np
import cv2
from face_recognition import face_landmarks

#Importing prediction pipeline
from req import predict

#Capture the frame from device camera
capture = cv2.VideoCapture(0)

#Run Until manually terminated/inturrupted
while True:
    #Reading the frame captured and preprocessing
    _, frame = capture.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Exracting landmarks
    landmarks = []
    landmarks.extend(face_landmarks(gray_image))

    if len(landmarks)>0:
        landmark = landmarks[0]
        for i, v in landmark.items():
            for j in v:
                #Projecting landmarks on the frame
                cv2.circle(frame, j, 2, (255,0,0))

    #Emotion prediction
    emotion = predict(landmarks)
    #Displaying prediction on the frame
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow('Emotion Detector', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break        

capture.release()