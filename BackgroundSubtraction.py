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

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i]-cnt2[j])
			if abs(dist) < 50 :
				return True
			elif i==row1-1 and j==row2-1:
				return False

cap = cv2.VideoCapture('/home/avi/Desktop/allStuff/IMG_1460.MOV')

cv2.namedWindow('Stream', cv2.WINDOW_AUTOSIZE)

prevTime = datetime.datetime.now()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

counter = 0
while(True):
	ret, frame = cap.read()
	if counter < 130 or counter % 10 == 0:

		fgmask = fgbg.apply(frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

		ret,thresh = cv2.threshold(fgmask,127,255,0)
		#a,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		a, contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
		if len(contours) > 0:
			LENGTH = len(contours)
			status = np.zeros((LENGTH,1))

			for i,cnt1 in enumerate(contours):
				x = i    
				if i != LENGTH-1:
					for j,cnt2 in enumerate(contours[i+1:]):
						x = x+1
						dist = find_if_close(cnt1,cnt2)
						if dist == True:
							val = min(status[i],status[x])
							status[x] = status[i] = val
						else:
							if status[x]==status[i]:
								status[x] = i+1

			unified = []
			maximum = int(status.max())+1
			for i in xrange(maximum):
				pos = np.where(status==i)[0]
				if pos.size != 0:
					cont = np.vstack(contours[i] for i in pos)
					hull = cv2.convexHull(cont)
					x,y,w,h = cv2.boundingRect(hull)
					crop = frame[y:y+h,x:x+w]
					# detect people in the image
					width, height = height, width = crop.shape[:2]
					if width > 50 and height > 50:
						cv2.imwrite("/home/avi/Desktop/pics/{}.jpg".format(counter), crop)
						# (rects, weights) = hog.detectMultiScale(crop, winStride=(8, 8), padding=(8, 8), scale=1.3)

						# # apply non-maxima suppression to the bounding boxes using a
						# # fairly large overlap threshold to try to maintain overlapping
						# # boxes that are still people
						# print("counted {} rects".format(len(rects)))
						# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
						# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

						# # draw the final bounding boxes
						# # for (xA, yA, xB, yB) in pick:
						# # 	cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

						# # show some information on the number of bounding boxes
						# print("[INFO]: {} are currently detected in stream".format(len(pick)))
						# print(weights)

		# show the output images
		cv2.imshow('Stream', frame)

	counter += 1
	print(counter)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 

cap.release()
cv2.destroyAllWindows()