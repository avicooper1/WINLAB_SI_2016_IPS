import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

READ_PATH = '/home/avi/Desktop/PositiveImages/All/'
imageFiles = onlyfiles = [f for f in listdir(READ_PATH) if isfile(join(READ_PATH, f))]

print "Reading in positive images"
images = []
for imageFile in imageFiles:
	try:
		images.append(cv2.imread(READ_PATH + imageFile))
	except:
		pass
print "Finished reading"

sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(images[0], None)

numberOfClusters = 300

dimensions = np.float32(kps)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(dimensions,numberOfClusters,None,criteria,10,flags)

print labels
print centers


