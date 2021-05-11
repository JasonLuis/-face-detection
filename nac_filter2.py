import cv2
import numpy as np
num_down = 2
num_bilateral = 7
video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

def cartoon_img(img_rgb):
    # Downsampling da imagem usando Gaussian Pyramid
    for _ in range(num_down):
        img_rgb = cv2.pyrDown(img_rgb)
    # Aplicacao do filtro bilateral 
    for _ in range(num_bilateral):
        img_rgb = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=9, sigmaSpace=7)
    # Upsampling da imagem usando Gaussian Pyramid
    for _ in range(num_down):
        img_rgb = cv2.pyrUp(img_rgb)
    # Conversao da imagem em tons de cinza
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Filtro mediano para realce
    img_blur = cv2.medianBlur(img_gray, 7)
    # Deteccao de bordas
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    
    # Conversao da imagem para RGB
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # Combinando a imagem colorida com a imagem com bordas destacadas
    img_cartoon = cv2.bitwise_and(img_rgb, img_edge)
    return img_cartoon


while True: 
  conectado, frame = video.read()
  #print(frame)

  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100,100))
  carton = cartoon_img(frame)
  for (x,y,l,a) in facesDetectadas:
    cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
    regiao = carton[y:y + a, x:x + l]
    frame[y:y + a, x:x + l] = regiao


  cv2.imshow('Video', frame)

  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows();