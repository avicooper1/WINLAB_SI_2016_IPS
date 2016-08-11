import cv2
import numpy as np
import yaml
import os

network = cv2.ml.ANN_MLP_create()

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

def trainNetwork():
	INPUT_DIR = "/home/avi/Desktop/TrainInput"
	INPUT_KEYWORD = "Input"
	OUTPUT_DIR = "/home/avi/Desktop/TrainOutput"
	OUTPUT_KEYWORD = "Output"
	os.system("/home/avi/workspace/DisplayImage/Debug/DisplayImage /home/avi/Desktop/PositiveImages/ForTraining 50 {} {} 1 {} {}".format(INPUT_DIR, INPUT_KEYWORD, OUTPUT_DIR, OUTPUT_KEYWORD))
	inputNeurons = readYAMLFile(INPUT_DIR, INPUT_KEYWORD)
	outputNeurons = readYAMLFile(OUTPUT_DIR, OUTPUT_KEYWORD)

	network.setLayerSizes(np.array([inputNeurons.shape[1], 100, outputNeurons.shape[1]]))	
	network.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
	network.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
	network.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))
	network.train(inputNeurons, cv2.ml.ROW_SAMPLE, outputNeurons)

def networkPredict():
	INPUT_DIR = "/home/avi/Desktop/PredictInput"
	INPUT_KEYWORD = "Input"
	OUTPUT_DIR = "~/Desktop/OUTOUT"
	OUTPUT_KEYWORD = "TESTTESTTEST"
	os.system("/home/avi/workspace/DisplayImage/Debug/DisplayImage /home/avi/Desktop/PositiveImages/B 50 {} {} 0 {} {}".format(INPUT_DIR, INPUT_KEYWORD, OUTPUT_DIR, OUTPUT_KEYWORD))
	inputNeurons = readYAMLFile(INPUT_DIR, INPUT_KEYWORD)
	print network.predict(inputNeurons)[0]

trainNetwork()
networkPredict()
