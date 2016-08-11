import cv2
from os import listdir
from os.path import isfile, join

READ_PATH = '/home/avi/Desktop/PositiveImages/ForTraining/'
imageFiles = onlyfiles = [f for f in listdir(READ_PATH) if isfile(join(READ_PATH, f))]

images = []

def showImage(image, FDType, kps):
	img2 = cv2.drawKeypoints(image, kps, outImage = None, color=(0,255,0), flags=0)
	cv2.namedWindow(FDType, 0);
	cv2.resizeWindow(FDType, 200,400);
	cv2.imshow(FDType, img2)
	print("For: {} -- # kps: {}, descriptors: {}".format(FDType,len(kps), descs.shape))

for imageFile in imageFiles:
	try:
		images.append(cv2.imread(READ_PATH + imageFile))
	except:
		pass
print "done reading"

for image in images:
	kaze = cv2.KAZE_create()
	(kps, descs) = kaze.detectAndCompute(image, None)
	showImage(image, "KAZE", kps)

	akaze = cv2.AKAZE_create()
	(kps, descs) = akaze.detectAndCompute(image, None)
	showImage(image, "AKAZE", kps)

	brisk = cv2.BRISK_create()
	(kps, descs) = brisk.detectAndCompute(image, None)
	showImage(image, "BRISK", kps)

	sift = cv2.xfeatures2d.SIFT_create()
	(kps, descs) = sift.detectAndCompute(image, None)
	showImage(image, "SIFT", kps)

	surf = cv2.xfeatures2d.SURF_create()
	(kps, descs) = surf.detectAndCompute(image, None)
	showImage(image, "SURF", kps)

	print
	print "----------------------------------------------"
	print

	response = cv2.waitKey(0)
	print response
	if response == 113:
		break