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
import CreateThreads as ct
import random

def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i]-cnt2[j])
			if abs(dist) < 1000 :
				return True
			elif i==row1-1 and j==row2-1:
				return False

cap = cv2.VideoCapture('/home/phalpha/Videos/Webcam/walking.webm')

kernel = np.ones((5,5),np.uint8)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

counter = 0

startTime = datetime.datetime.now()

rslt=[]

def processImage(frame, conts, count):
	LENGTH = len(conts)
	status = np.zeros((LENGTH,1))

	for i,cnt1 in enumerate(conts):
		x = i    
		if i != LENGTH-1:
			for j,cnt2 in enumerate(conts[i+1:]):
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
	subcounter = 0
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(conts[i] for i in pos)
			hull = cv2.convexHull(cont)
			x,y,w,h = cv2.boundingRect(hull)
			print("(",x+w/2,",",y+h/2,")")
			temp = (x+w/2,y+h/2)
			while(len(rslt)>32):
				del rslt[0]
			rslt.append(temp)
			subcounter += 1
			#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)


while(True):
	ret, frame = cap.read()
	if not ret:
		break
	else:
		fgmask = fgbg.apply(frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		a,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if counter > 0 and len(contours) > 0:
			#processImage(frame, contours, counter)
		 	newThread = ct.MyThread(processImage, frame, contours, counter)
			newThread.start()
		#newimg=frame.copy()
		#for x in range(1024):
		#	for y in range(1280):
		#		newimg[x,y]=[0,0,0]

		size=len(rslt)
		for x in range(size-1):
			#print (x)
			#cv2.circle(frame, x, 5, (255, 255, 255), -1,5,0)
			cv2.line(frame, rslt[x], rslt[x+1], (255,255,255), 4, 4, 0)
			#if counter!=0:
			#	cv2.line(newimg, rslt[x],rslt[x+1], (255,255,255), 1, 8, 0)
		cv2.imwrite("/home/phalpha/Desktop/pathtest/img-{}.jpg".format(counter), frame)
	counter += 1

	
	#print(counter)
cap.release()
cv2.destroyAllWindows()

endTime = datetime.datetime.now()
runTime = endTime - startTime
print("Total run time was {} seconds".format(runTime))
print("done")