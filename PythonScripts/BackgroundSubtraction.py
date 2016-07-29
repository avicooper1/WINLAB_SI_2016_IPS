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
import pdb
import sys

cap = cv2.VideoCapture('/home/avi/Desktop/PositiveVid.webm')
WRITE_LOCATION = '/home/avi/Desktop/A/'

kernel = np.ones((5,5),np.uint8)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

counter = 0

startTime = datetime.datetime.now()

GROUPING_DIST = 500
MIN_RECT_SIZE = 20
RECT_PADDING = 15


# Function: -isClose
# 	Purpose:
# 		-determine whether one countour is a given distance from another contours
# 	Params:
# 		-cnt1: the first contour
# 		-cnt2: the second contour
# 		-dist: the distance
# 	Returns:
# 		-bool: whether the contours are within the given distance
# def isClose(cnt1, cnt2, dist):
# 	row1,row2 = cnt1.shape[0],cnt2.shape[0]
# 	for i in xrange(row1):
# 		for j in xrange(row2):
# 			dist = np.linalg.norm(cnt1[i]-cnt2[j])
# 			if abs(dist) < dist :
# 				return True
# 			elif i==row1-1 and j==row2-1:
# 				return False

def ptDist(pt1, pt2):
	return (((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2)) ** 0.5

def interiorDist(w, h, slope):
	if slope == 1:
		return (((0.5 * w) ** 2) + ((0.5 * h) ** 2)) ** 0.5
	if slope < 1:
		return (((slope * w) ** 2) + (w ** 2)) ** 0.5
	if slope > 1:
		return ((((slope ** -1) * h) ** 2) + (h ** 2)) ** 0.5

def isClose(rect1, rect2, dist):
	
	deltaX = rect1[0] - rect2[0]
	deltaY = rect1[1] - rect2[1]

	if deltaX == 0 or deltaY == 0:
		if deltaY == 0 and deltaX > 0:
			return (deltaX - (0.5 * rect1[2]) - (0.5 * rect2[2])) < dist
		if deltaX == 0 and deltaY > 0:
			return (deltaY - (0.5 * rect1[3]) - (0.5 * rect2[3])) < dist
		else:
			return True
	slope = (deltaY) / (deltaX)
	absSlope = abs(slope)
	midPntDistance = (deltaX ** 2 + deltaY ** 2) ** 0.5

	return (midPntDistance - interiorDist(rect1[2], rect1[3], absSlope) - interiorDist(rect2[2], rect2[3], absSlope)) < dist

def generatePts(rect):
	return ((rect[0], rect[1]), (rect[0] + rect[2], rect[1]), (rect[0], rect[1] + rect[3]), (rect[0] + rect[2], rect[1] + rect[3]))

def bindRects(pt1, pt2):
	xMin = sys.maxint
	xMax = 0
	yMin = sys.maxint
	yMax = 0

	pts = generatePts(pt1) + generatePts(pt2)

	for pt in pts:
		if pt[0] < xMin:
			xMin = pt[0]
		elif pt[0] > xMax:
			xMax = pt[0]
		if pt[1] < yMin:
			yMin = pt[1]
		elif pt[1] > yMax:
			yMax = pt[1]
	return (xMin, yMin, xMax - xMin, yMax - yMin)

def groupRects(rects):
	for x in range (0, len(rects)):
		for y in range (x + 1, len(rects)):
			if isClose(rects[x], rects[y], GROUPING_DIST):
				rects.append(bindRects(rects[x], rects[y]))
				del rects[y]
				del rects[x]
				return (True, rects)
				break
	return (False, rects)


# Function: -processImage
# 	Purpose:
# 		-group contours close to one another and return a list of rectangles bounding these grouped contours
# 	Params:
# 		-frame: the image to be processed
# 		-conts: the generated contours for this image
# 		-count: the identification for this frame. will be used as the filename
# 	Returns:
# 		-[(x, y, w, h)]: a list of tuples of the rectangles dimenstions containing grouped contours
def processImage(frame, conts, count):
	LENGTH = len(conts)
	status = np.zeros((LENGTH,1))

	for i,cnt1 in enumerate(conts):
		x = i    
		if i != LENGTH-1:
			for j,cnt2 in enumerate(conts[i+1:]):
				x += 1
				# if isClose(cnt1,cnt2, GROUPING_DIST):	
				# 	val = min(status[i],status[x])
				# 	status[x] = status[i] = val
				# else:
				if status[x]==status[i]:
					status[x] = i+1

	rects = []

	maximum = int(status.max())+1
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(conts[i] for i in pos)
			hull = cv2.convexHull(cont)
			x,y,w,h = cv2.boundingRect(hull)
			if w > MIN_RECT_SIZE and h > MIN_RECT_SIZE:
				rects.append((x, y, w, h))
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

	while True:
		grouped, rects = groupRects(rects)
		if not grouped:
			break
	
	returnRects = []
	for (x, y, w, h) in rects:
		returnRects.append((x - RECT_PADDING, y - RECT_PADDING, w + (2 * RECT_PADDING), h + (2 * RECT_PADDING)))
		cv2.rectangle(frame, (x - RECT_PADDING, y - RECT_PADDING), (x+w + (2 * RECT_PADDING), y+h+ (2 * RECT_PADDING)), (0,0,255), 2)
	cv2.imshow('frame', frame)
	cv2.waitKey(1)
	return returnRects

writeCounter = 0
def writeImage(frame):
	global writeCounter
	cv2.imwrite(WRITE_LOCATION + str(writeCounter) + '.jpg', frame)
	writeCounter += 1

while(True):
	ret, frame = cap.read()
	if not ret:
		break
	else:
		fgmask = fgbg.apply(frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		a,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if counter > 120 and len(contours) > 0:
			rects = processImage(frame, contours, counter)
			for rect in rects:
				writeImage(frame[rect[1]:rect[1] + rect[3], rect[0]: rect[0] + rect[2]])
			# newThread = ct.MyThread(processImage, frame, contours, counter)
			# newThread.start()
	counter += 1
	print(counter)
cap.release()
cv2.destroyAllWindows()

endTime = datetime.datetime.now()
runTime = endTime - startTime
print("Total run time was {} seconds".format(runTime))
print("done")