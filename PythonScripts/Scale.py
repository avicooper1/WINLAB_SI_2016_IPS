import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb

imagesPath = '/home/avi/opencv-haar-classifier-training/positive_images/'

images = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]

for imageFile in images:
	image = cv2.imread(imagesPath + imageFile)
	try:
		res = cv2.resize(image,None,fx=.1, fy=.1, interpolation = cv2.INTER_CUBIC)
		cv2.imwrite('/home/avi/Desktop/ASDF/' + imageFile, res)
	except:
		print "Coulnd't scale image"