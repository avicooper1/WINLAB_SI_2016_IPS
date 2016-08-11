# import the necessary packages
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
from os import listdir
from os.path import isfile, join

cap = cv2.VideoCapture('/home/avi/Desktop/OUTPUT1.avi')
WRITE_LOCATION = '/home/avi/Desktop/PositiveImages/ForTraining/'

kernel = np.ones((2,2),np.uint8)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

counter = 0

GROUPING_DIST = 500
MIN_RECT_SIZE = 20
RECT_PADDING = 30
NUMBER_OF_CLUSTERS = 10


def printProgress (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = u'\u2588' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\r%s |%s| Complete\n' % (prefix, bar, percents, '%', suffix)),
        sys.stdout.flush()

class Cluster:
	def __init__ (self, center, values):
		self.center = center
		self.values = values

class Image:
	def __init__ (self, image, roi):
		try:
			self.image = image
			self.roi = roi
			heightpre, widthpre, channelspre = image.shape
			ratiopre = heightpre / float(widthpre)
			self.height = heightpre
			self.width = widthpre
			self.ratio = ratiopre
		except:
			pass
class ROI:
	def __init__ (self, roi):
		self.x = roi[0]
		self.y = roi[1]
		self.w = roi[2]
		self.h = roi[3]
		self.r = float(self.h) / self.w

class Pt:
	def __init__ (self, pt):
		self.x = pt[0]
		self.y = pt[1]

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
	return [Pt((rect[0], rect[1])), Pt((rect[0] + rect[2], rect[1])), Pt((rect[0], rect[1] + rect[3])), Pt((rect[0] + rect[2], rect[1] + rect[3]))]

def bindRects(rect1, rect2):
	xMin = sys.maxint
	xMax = 0
	yMin = sys.maxint
	yMax = 0

	pts = generatePts(rect1) + generatePts(rect2)

	pts.sort(key=lambda pt: pt.x)
	xMin = pts[0].x
	xMax = pts[-1].x

	pts.sort(key=lambda pt: pt.y)
	yMin = pts[0].y
	yMax = pts[-1].y
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

	while True:
		grouped, rects = groupRects(rects)
		if not grouped:
			break
	roi = []
	for (x, y, w, h) in rects:
		roi.append(ROI((x - RECT_PADDING, y - RECT_PADDING, w + (2 * RECT_PADDING), h + (2 * RECT_PADDING))))
	return Image(frame, roi)

images = []
while(True):
	global images 
	ret, frame = cap.read()
	if not ret:
		break
	else:
		fgmask = fgbg.apply(frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		a,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if counter > 120 and len(contours) > 0:
			images.append(processImage(frame, contours, counter))
			# newThread = ct.MyThread(processImage, frame, contours, counter)
			# newThread.start()
	counter += 1
	sys.stdout.write("\r {} frames processed".format(counter))
	sys.stdout.flush()

NUM_OF_ROI = 0
for currentPic in images:
	NUM_OF_ROI += len(currentPic.roi)

dimensions = np.empty(NUM_OF_ROI)
counter = 0
for currentPic in images:
	for roi in currentPic.roi:
		dimensions[counter] = roi.r
		counter += 1

dimensions = np.float32(dimensions)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(dimensions,NUMBER_OF_CLUSTERS,None,criteria,10,flags)

clusters = []

for x in range(0, NUMBER_OF_CLUSTERS):
	clusters.append(Cluster(centers[x], dimensions[labels.ravel()==x]))

clusters.sort(key=lambda x: x.center)

for x in range (0,NUMBER_OF_CLUSTERS):
	cluster = clusters[x]
	print str(x) + ": ",
	print cluster.center,
	print":",
	print len(cluster.values),
	if len(cluster.values) != 0:
		listMax = max(cluster.values)
		listMin = min(cluster.values)
		print "		",
		print "range: " + str (listMax - listMin) + ", min: " + str(listMin) + ", max: " + str(listMax)
	else:
		print

arraysChosen = raw_input("Input the array number(s) to be saved: (',' seperated) ")
arrays = []
currentString = ""
for x in arraysChosen:
	if x != ',':
		currentString += x
	else:
		arrays.append(int(currentString))
		currentString = ""
arrays.append(int(currentString))
imagesToSave = []

for x in range (0,len(labels)):
	print(labels[x])
	if labels[x] in arrays:
		imagesToSave.append(images[x])

#dataFile = open(WRITE_LOCATION + 'info.dat', 'w')
SPACING = '  '
NUM_OF_IMGS_TO_SAVE = len(imagesToSave)
for x in range(0, NUM_OF_IMGS_TO_SAVE):
	imageFileName = 'pha.{}.jpg'.format(x)
	numOfROI = str(len(imagesToSave[x].roi))
	descriptionOfRoi = ""
	for roi in imagesToSave[x].roi:
		descriptionOfRoi += "{}{} {} {} {}".format(SPACING, roi.x, roi.y, roi.w, roi.h)
		cv2.imwrite(WRITE_LOCATION + imageFileName, imagesToSave[x].image[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w])
	# try:
	# 	cv2.imwrite(WRITE_LOCATION + imageFileName, imagesToSave[x].image[image.roi.y:image.roi.y+image.roi.h, image.roi.x:image.roi.x+image.roi.w])
	# 	#dataFile.write(imageFileName + SPACING + numOfROI + descriptionOfRoi + '\n')
	# except:
	# 	pass
	#printProgress(x, NUM_OF_IMGS_TO_SAVE - 1)
#dataFile.close()


cap.release()
cv2.destroyAllWindows()
print("Complete.")

