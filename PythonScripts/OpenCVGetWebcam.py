import numpy as np
import cv2

cap = cv2.VideoCapture('http://192.168.206.175/mjpg/video.mjpg')
counter = 0
while(True):
    ret, frame = cap.read()
    if ret:
    	cv2.imshow('frame', frame)
    	#cv2.imwrite("/home/avi/opencv-haar-classifier-training/negative_images/{}.jpg".format(counter), frame)
    else:
    	break
    print counter
    counter += 1
cap.release()
cv2.destroyAllWindows()

