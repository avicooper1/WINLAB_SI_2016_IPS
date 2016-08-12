# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys
import SI2016IPS_Required_Imports as reqi

def getStereo(bss):
	returns = []
	for bs in bss:
		returnVal = next(bs.get())
		if not returnVal:
			return None
		elif not returns:
			returns.append(returnVal)
			break
		elif len(returnVal) == len(returns[0]):
			returns.append(returnVal)

	return returns

	if img1cnts and img2cnts:
		return (img1cnts, img2cnts)

class BS:
	def __init__(self, negativeVid, kernelSize=10, minContourArea=5000):
		self.kernel = np.ones((kernelSize,kernelSize),np.uint8)
		self.fgbg = cv2.createBackgroundSubtractorMOG2()
		self.minContourArea = minContourArea
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
				if len(contours != 0):
					returnContours = []
					for cnt in contours:
						if cv2.contourArea(cnt) > self.minContourArea
						x,y,w,h = cv2.boundingRect(currentLargestCnt)
						masked_image = masked_image[y:y+h, x:x+w]
						returnContours.append((masked_image, (x,y,w,h), frame))
					yield returnContours