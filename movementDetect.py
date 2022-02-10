import cv2
import numpy as np


captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, frame = captura.read()
    if ret == False:
        break 
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    color = (0,255,0)
    texto_estado = 'Estado: no se ha detectado movimiento'

    area_pts = np.array([[240,320],[480,320],[620,frame.shape[0]],[60,frame.shape[0]]])

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts],-1,(255),-1)
    image_area = cv2.bitwise_and(gris,gris, mask=imAux)

    fgmask = mog.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt)>500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            texto_estado = "Estado: Alerta de Movimiento!"
            color = (0,0,255)
    cv2.drawContours(frame,[area_pts],-1,color,2)
    cv2.putText(frame, texto_estado,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,1)

    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)

    k = cv2.waitKey(30) & 0xFF
    if k == 27: break

captura.release()
cv2.destroyAllWindows()
