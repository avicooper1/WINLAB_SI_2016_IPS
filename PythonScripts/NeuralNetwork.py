import cv2
import numpy as np
import yaml
import os
import SI2016IPS_Required_Imports as reqi
import collections
import time
import sys

class NN:
	def __init__ (self, workspaceDir, cppFileLoc, numOfInputNeurons, numPosFindings):
		self.workspaceDir = workspaceDir
		self.trainingDataDir = workspaceDir + "/TrainingDataHoldingFolder"
		reqi.mf.ensureDir(self.trainingDataDir)
		self.inputFile = workspaceDir + "/InputNuerons"
		self.inputKey = "Input"
		self.outputFile = workspaceDir + "/OutputNeurons"
		self.outputKey = "Output"
		self.descriptorsFile = workspaceDir + "/Vocabulary"
		self.descriptorsKey = "vocabulary"
		self.classesFile = workspaceDir + "/classes.txt"
		self.cppFileLoc = cppFileLoc
		self.numOfInputNeurons = numOfInputNeurons
		self.numPosFindings = numPosFindings

	@staticmethod
	def readYAMLFile(filePath, header):
		f = open(filePath)
		s = f.read()
		f.close()
		s = s[10:-1]
		toRemove = ['[', ']']
		s = s.translate(None, ''.join(toRemove))

		def opencv_matrix(loader, node):
		    mapping = loader.construct_mapping(node)
		    mat = np.float32(map(float, mapping['data'].split(",")))
		    mat.resize(mapping["rows"], mapping["cols"])
		    return mat

		yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
		y = yaml.load(s)
		return y[header]

	@staticmethod
	def readADLFile(filePath):

		returnDict = {}

		f = open(filePath)
		data = f.read()[:-1]
		f.close()

		data = data.split(",")

		for pair in data:
			returnDict[pair[0]] = pair[2:]
		return returnDict

	def createTrainingData(self, bs, classId):
		counter = 5
		while counter > 0:
			sys.stdout.write("\rData collection will begin in " + str(counter) + "...")
			sys.stdout.flush()
			time.sleep(1)
			counter -= 1
		sys.stdout.write("\rData collection will begin in now")
		sys.stdout.flush()
		print "\nCollecting identification data. Please continue to move within the frame."
		result = None
		while counter < self.numPosFindings:
			try:
				result = next(bs.get())
				cv2.imwrite(self.trainingDataDir + "/" + classId + "." + str(counter) + ".jpg", result[0][0])
				reqi.mf.printProgress(counter, self.numPosFindings - 1)
			except Exception,e:
				if e:
					print "BS crashed. Refer to the following exception: ",
					print str(e)
				else:
					print "The video stream probably ran out of frames."
				break
			counter += 1
		print "Finished collecting data. Normal operations will now resume."

	def createTrainingFiles(self, imagesLoc=None):
		if imagesLoc is None:
			imagesLoc = self.trainingDataDir
		print "Creating training files"
		os.system("{} {} {} {} {} {} {} {} {} {}".format(self.cppFileLoc, imagesLoc, self.numOfInputNeurons, self.inputFile, self.inputKey, self.outputFile, self.outputKey, self.descriptorsFile, self.descriptorsKey, self.classesFile))
		print "done"

	def train(self):

		print "Training network"

		inputNeurons = self.readYAMLFile(self.inputFile, self.inputKey)
		outputNeurons = self.readYAMLFile(self.outputFile, self.outputKey)
		self.descriptors = self.readYAMLFile(self.descriptorsFile, self.descriptorsKey)
		self.classes = self.readADLFile(self.classesFile)

		self.network = cv2.ml.ANN_MLP_create()
		self.network.setLayerSizes(np.array([inputNeurons.shape[1], 100, outputNeurons.shape[1]]))	
		self.network.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
		self.network.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
		self.network.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))
		self.network.train(inputNeurons, cv2.ml.ROW_SAMPLE, outputNeurons)

		print "done"

	def predict(self, image):
		IMAGE_LOC = self.workspaceDir + "/ImageHoldingFolder"
		reqi.mf.ensureDir(IMAGE_LOC)
		IMAGE_ADD_ON = "/aaa.000.jpg"
		INDIVIDUAL_DESCRIPTORS_FILE = self.workspaceDir + "/IndividualDescriptors"
		INDIVIDUAL_DESCRIPTORS_KEY = "descriptors"
		cv2.imwrite(IMAGE_LOC + IMAGE_ADD_ON, image)
		describeSuccess = os.system("{} {} {} {} {} {} {}".format(self.cppFileLoc, IMAGE_LOC, self.numOfInputNeurons, INDIVIDUAL_DESCRIPTORS_FILE, INDIVIDUAL_DESCRIPTORS_KEY, self.descriptorsFile, self.descriptorsKey))
		individualDescriptors = self.readYAMLFile(INDIVIDUAL_DESCRIPTORS_FILE, INDIVIDUAL_DESCRIPTORS_KEY)
		os.system("rm {}".format(IMAGE_LOC + IMAGE_ADD_ON))
		return self.classes[str(int(self.network.predict(individualDescriptors)[0]))]