import cv2
import numpy as np
from pynput import mouse
num_down = 2
num_bilateral = 7
video = cv2.VideoCapture(1)

classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

while True: 
  conectado, frame = video.read()
  #print(frame)
  
  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100,100))
  edges = cv2.Canny(frame,100,200)
  for (x,y,l,a) in facesDetectadas:
    cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
    regiao = frame[y:y + a, x:x + l]
    dst = cv2.stylization(regiao, sigma_s=60, sigma_r=0.07)
    frame[y:y + a, x:x + l] = dst


  cv2.imshow('Video', frame)
  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows();