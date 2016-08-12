import cv2
import numpy as np
import yaml
import os
import SI2016IPS_Required_Imports as reqi
import collections
import pdb

NUM_INPUT_NEURONS = 300

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

def setTrainingFiles(imagesLoc, inputOutputDirLoc):
	global INPUT_DIR
	global INPUT_KEY
	global OUTPUT_DIR
	global OUTPUT_KEY
	global DESCRIPTORS_DIR
	global DESCRIPTORS_KEY

	INPUT_DIR = inputOutputDirLoc + "/InputNuerons"
	INPUT_KEY = "Input"
	OUTPUT_DIR = inputOutputDirLoc + "/OutputNeurons"
	OUTPUT_KEY = "Output"
	DESCRIPTORS_DIR = inputOutputDirLoc + "/Vocabulary"
	DESCRIPTORS_KEY = "asdf"

def createTrainingFiles(cppFileLoc, imagesLoc):
	print "Creating training files"
	os.system("{} {} {} {} {} {} {}".format(cppFileLoc, imagesLoc, NUM_INPUT_NEURONS, INPUT_DIR, INPUT_KEY, OUTPUT_DIR, OUTPUT_KEY))
	print "done"

def train():

	global network
	global flann
	global descriptors

	print "Training network"

	inputNeurons = readYAMLFile(INPUT_DIR, INPUT_KEY)
	outputNeurons = readYAMLFile(OUTPUT_DIR, OUTPUT_KEY)
	descriptors = readYAMLFile(DESCRIPTORS_DIR, DESCRIPTORS_KEY)

	network = cv2.ml.ANN_MLP_create()
	network.setLayerSizes(np.array([inputNeurons.shape[1], 100, outputNeurons.shape[1]]))	
	network.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
	network.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
	network.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))
	network.train(inputNeurons, cv2.ml.ROW_SAMPLE, outputNeurons)

	print "done"

def predict(image):
	IMAGE_LOC = "/home/avi/Desktop/ImageHoldingFolder"
	IMAGE_ADD_ON = "/aaa.000.jpg"
	INDIVIDUAL_DESCRIPTORS_DIR = "/home/avi/Desktop/ASDF"
	INDIVIDUAL_DESCRIPTORS_KEY = "asdf"
	cv2.imwrite(IMAGE_LOC + IMAGE_ADD_ON, image)
	describeSuccess = os.system("/home/avi/workspace/DisplayImage/Debug/DisplayImage {} {} {} {} {} {} a".format(IMAGE_LOC, NUM_INPUT_NEURONS, INDIVIDUAL_DESCRIPTORS_DIR, INDIVIDUAL_DESCRIPTORS_KEY, DESCRIPTORS_DIR, DESCRIPTORS_KEY))
	individualDescriptors = readYAMLFile(INDIVIDUAL_DESCRIPTORS_DIR, INDIVIDUAL_DESCRIPTORS_KEY)
	os.system("rm {}".format(IMAGE_LOC + IMAGE_ADD_ON))
	return network.predict(individualDescriptors)[0]