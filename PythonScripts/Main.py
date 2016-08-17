import SI2016IPS_Required_Imports as reqi

MAIN_PATH = "/home/avi/Desktop/IPS2016Workspace"
reqi.mf.ensureDir(MAIN_PATH)

bs1 = reqi.bs.BS("~/Desktop/NegativeVideo.webm")
bs1.setupPos("/home/avi/Desktop/OUTPUT2.avi")
nn = reqi.nn.NN(MAIN_PATH, "/home/avi/workspace/DisplayImage/Debug/DisplayImage", 100, 500)	
#nn.createTrainingData(bs1, "avi")
#bs1.setupPos("/home/avi/Desktop/OUTPUT1.avi")
#nn.createTrainingData(bs1, "poj")
#nn.createTrainingFiles()
nn.train()

result = None
while True:
	try:
		result = next(bs1.get())
	except Exception,e:
		print "The video stream may have run out of frames. Refer to the following exception: "
		print str(e)
		break
	result = result[0]
	predictedName = nn.predict(result[0])
	reqi.cv2.rectangle(result[2],(result[1][0],result[1][1]),(result[1][0] + result[1][2],result[1][1] + result[1][3]), (0,255,0), 3)
	reqi.cv2.putText(result[2], predictedName, (0,50), reqi.cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2, reqi.cv2.LINE_AA)
	reqi.mf.showImage(result[2])
