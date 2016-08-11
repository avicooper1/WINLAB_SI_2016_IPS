# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime
import thread
import time

classifier = cv2.CascadeClassifier('/home/avi/opencv-haar-classifier-training/classifier3/cascade.xml')

cap = cv2.VideoCapture('/home/avi/Desktop/OUTPUT1.avi')
#cap = cv2.VideoCapture('http://192.168.206.175/mjpg/video.mjpg')
#cap = cv2.VideoCapture(0)
#cv2.namedWindow('Stream', cv2.WINDOW_AUTOSIZE)

def processImage(frame):
	#frame = imutils.resize(frame, width = min(600, frame.shape[1]))

	# detect people in the image
	rects = classifier.detectMultiScale(frame, 1.1, 200)# winStride=(4, 4), padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:	
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxe
	print("[INFO]: {} are currently detected in stream".format(len(pick)))

	# show the output images
	try:
		cv2.imshow('Stream', frame)
	except:
		pass
	cv2.waitKey(1)

while(True):

	# Capture frame-by-frame
	ret, currentFrame = cap.read()
	
	processImage(currentFrame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()