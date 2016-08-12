# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys
import SI2016IPS_Required_Imports as reqi

class BS:
	def __init__(self, negativeVid):
		self.kernel = np.ones((10,10),np.uint8)
		self.fgbg = cv2.createBackgroundSubtractorMOG2()
		self.frameCounter = 0
		self.cap = cv2.VideoCapture(negativeVid)
		ret, frame = self.cap.read()
		while self.frameCounter <= 120:
			if not ret:
				break
			else:
				self.frameCounter += 1

	def setupPos(self, positiveVid):
		self.cap = cv2.VideoCapture(positiveVid)

	# def: 
	# 	run: accepts videos to run background subtraction and returns an array of the cropped images with transparent backgrounds
	# 	(limitations: currently only supports one user at a time)
	# params:
	# 	-negativeVid: the file path of the video of the empty room
	# 	-positiveVid: the file path of the video of the occupied room
	#	-doBoundingRect: an option to return or yield the image as well as the bounding rect in a tuple
	# returns:
	# 	-images: an array of the images with users croppped out. (perfect crops are not gauranteed. there may be parts of the image of the user missing or background added on.)
	# 		the image returned is surrounded by a bounding rect where areas non included in the crop are transparent 
	def get(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			else:
				fgmask = self.fgbg.apply(frame)
				fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
				fgmask = cv2.dilate(fgmask, self.kernel, iterations = 2)
				a, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
				mask = np.zeros(frame.shape, dtype=np.uint8)
				channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
				ignore_mask_color = (255,)*channel_count
				cv2.fillPoly(mask, contours, ignore_mask_color)
				# apply the mask
				masked_image = cv2.bitwise_and(frame, mask)
				lastLargestArea = 0
				currentLargestCnt = None
				for cnt in contours:
					if cv2.contourArea(cnt) > lastLargestArea:
						lastLargestArea = cv2.contourArea(cnt)
						currentLargestCnt = cnt
				if currentLargestCnt is not None:
					x,y,w,h = cv2.boundingRect(currentLargestCnt)
					masked_image = masked_image[y:y+h, x:x+w]
					if lastLargestArea > 15000:
						yield (masked_image, (x,y,w,h), frame)