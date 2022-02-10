import cv2
import numpy as np


captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, frame = captura.read()
    if ret == False:
        break 