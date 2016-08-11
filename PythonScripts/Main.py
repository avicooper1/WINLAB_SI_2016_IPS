import SI2016IPS_Required_Imports as reqi

reqi.bs.setup("/home/avi/Desktop/NegativeVid.webm")

#Will only run if there are no images in folder to store prepped images for training. If some were deleted, the folder needs to be fully emptied
if len(reqi.os.listdir("/home/avi/Desktop/positivesTest")) == 0:
	reqi.bs.setupPos("/home/avi/Desktop/OUTPUT1.avi")
	images1 = [i[0] for i in images1]
	reqi.mf.writeImagesToFile(images1, "/home/avi/Desktop/positivesTest/poj.")
	
	reqi.bs.setupPos("/home/avi/Desktop/OUTPUT2.avi")
	images2 = list(reqi.bs.get())
	images2 = [i[0] for i in images2]
	reqi.mf.writeImagesToFile(images2, "/home/avi/Desktop/positivesTest/avi.")


#cpp file loc "/home/avi/workspace/DisplayImage/Debug/DisplayImage"
reqi.nn.setTrainingFiles("/home/avi/Desktop/positivesTest", "/home/avi/Desktop")
reqi.nn.train()
#reqi.bs.setupPos("/home/avi/Desktop/OUTPUT2.avi")
# while True:
# 	try:
# 		result = next(reqi.bs.get())
# 		predictionResult = reqi.nn.predict(result[0])
# 		print predictionResult
# 		predictedName = ""
# 		if predictionResult == -1:
# 			predictionResult = "NA"
# 		elif predictionResult == 0:
# 			predictedName = "Poojit"
# 		elif predictionResult == 1:
# 			predictedName = "Avi"
# 		reqi.cv2.rectangle(result[2],(result[1][0],result[1][1]),(result[1][0] + result[1][2],result[1][1] + result[1][3]), (0,255,0), 3)
# 		reqi.cv2.putText(result[2], predictedName, (0,50), reqi.cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2, reqi.cv2.LINE_AA)
# 		reqi.mf.showImage(result[2])
# 	except Exception,e:
# 		print "The video stream may have run out of frames. Refer to the following exception: "
# 		print str(e)
# 		break
a = reqi.cv2.imread("/home/avi/Desktop/positivesTest/avi.17.jpg")
reqi.nn.predict(a)
