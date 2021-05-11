import cv2
import numpy as np

num_down = 2
num_bilateral = 7
video = cv2.VideoCapture(1)
count = 0

classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')


def mouse_click(event, x, y, flags, param):
  # se foi click do botao direito 
  global count
  if event == cv2.EVENT_RBUTTONDOWN:
    count = 0 
    print(count)
  elif event == cv2.EVENT_LBUTTONDOWN:  
    if count == 3:
      count = 1
    else: 
      count += 1 
    print(count)

def filtro1(frame):
  glausian = cv2.GaussianBlur(frame,(15,15),cv2.BORDER_DEFAULT)
  return glausian

def filtro2(frame):
    for _ in range(num_down):
        frame = cv2.pyrDown(frame)
    
    for _ in range(num_bilateral):
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=9, sigmaSpace=7)
    
    for _ in range(num_down):
        frame = cv2.pyrUp(frame)
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    img_blur = cv2.medianBlur(img_gray, 7)
    
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    
    
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    
    img_cartoon = cv2.bitwise_and(frame, img_edge)
    return img_cartoon

def filtro3(frame):
  frame = cv2.stylization(frame, sigma_s=60, sigma_r=0.07);
  return frame


while True:
  conect1, frame = video.read()
  conect2,filtro = video.read()
  frameCinza = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100,100))
  
  for (x,y,l,a) in facesDetectadas:
    if count == 1:
      filtro[y:y + a, x:x + l] = filtro1(filtro[y:y + a, x:x + l])
    elif count == 2:
      regiao = filtro2(filtro)
      filtro[y:y + a, x:x + l] = regiao[y:y + a, x:x + l]
    elif count == 3:
      filtro[y:y + a, x:x + l] = filtro3(filtro[y:y + a, x:x + l])
    cv2.rectangle(filtro, (x,y), (x + l, y + a), (0,255,0),2)
  
  cv2.setMouseCallback('Video Filtro', mouse_click)
  cv2.imshow('Original', frame)
  cv2.imshow('Video Filtro', filtro)
  if cv2.waitKey(1) == ord('q'):
    break

cv2.destroyAllWindows();