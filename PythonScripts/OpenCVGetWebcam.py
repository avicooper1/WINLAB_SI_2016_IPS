import numpy as np
import cv2

cap = cv2.VideoCapture('/home/avi/Desktop/NegativeVid.webm')
counter = 0
while(True):
    ret, frame = cap.read()
    if ret:
    	cv2.imwrite("/home/avi/Desktop/NegativeImages/{}.jpg".format(counter), frame)
    else:
    	break
    print counter
    counter += 1
cap.release()
cv2.destroyAllWindows()

