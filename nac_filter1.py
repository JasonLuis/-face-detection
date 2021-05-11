import cv2

video = cv2.VideoCapture(1)
classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

while True: 
  conectado, frame = video.read()
  #print(frame)

  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100,100))
  for (x,y,l,a) in facesDetectadas:
    cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
    regiao = frame[y:y + a, x:x + l]
    frame_gaussian = cv2.GaussianBlur(regiao,(15,15),cv2.BORDER_DEFAULT)
    frame[y:y + a, x:x + l] = frame_gaussian

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows();
