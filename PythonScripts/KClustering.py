import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

class Cluster:
	def __init__ (self, center, values):
		self.center = center
		self.values = values

class Image:
	def __init__ (self, image):
		try:
			self.image = image
			heightpre, widthpre, channelspre = image.shape
			ratiopre = heightpre / float(widthpre)
			self.height = heightpre
			self.width = widthpre
			self.ratio = ratiopre
		except:
			pass

imagesPath = '/home/avi/Desktop/PositiveImages/All/'
numberOfClusters = 10
savePath = '/home/avi/Desktop/PositiveImages/Clustered/'

images = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]

dimensions = np.empty(len(images))

for x in range(0, len(images)):
	currentPic = cv2.imread(imagesPath + images[x])
	try:
		height, width, channels = currentPic.shape
		ratio = height / float(width)
		dimensions[x - 121] = ratio
	except:
		pass

dimensions = np.float32(dimensions)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,label,centers = cv2.kmeans(dimensions,numberOfClusters,None,criteria,10,flags)

clusters = []

for x in range(0, numberOfClusters):
	clusters.append(Cluster(centers[x], dimensions[label.ravel()==x]))

clusters.sort(key=lambda x: x.center)

for x in range (0,numberOfClusters):
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
for array in arrays:
	chosenMin = min(clusters[array].values)
	chosenMax = max(clusters[array].values)
	for image in images:
		currentPic = Image(cv2.imread(imagesPath + image))
		try:
			if currentPic.ratio >= chosenMin and currentPic.ratio <= chosenMax:
				imagesToSave.append(currentPic)
		except:
			pass

imagesToSave.sort(key=lambda x: x.ratio)
for x in range(0, len(imagesToSave)):
	try:
		cv2.imwrite(savePath + '{}.jpg'.format(x), imagesToSave[x].image)
	except:
		pass


