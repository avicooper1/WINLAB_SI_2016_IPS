# import the necessary packages
from __future__ import print_function
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
import random
import sys
import math
#import matplotlib.pyplot as plt
#from scipy.optimize import fsolve

#takes in x,y,w,h of bounding box, returns distances and coordinates

rslt = []
rslt2= []
coordlist= []
angllist=[]
distllist=[]
angrlist=[]
distrlist=[]
count=[]

sep = 2400
dl = 10
dr = 10
AL=10
AR=10
wa=0

fxl=976.75544679
fyl=971.22275106
cxl=628.09908652
cyl=324.93190823


fxr=969.37677045
fyr=966.71172291
cxr=630.11580587
cyr=327.64621795

camera_matrixr=np.array([[fxr, 0, cxr],[0, fyr, cyr],[0, 0, 1]])
dist_coeffr = np.array([-0.51007507,  0.35115921,  0.00163526,  0.00418143, -0.12703123])

camera_matrixl=np.array([[fxl, 0, cxl],[0, fyl, cyl],[0, 0, 1]])
dist_coeffl = np.array([-0.49090347,  0.24224545,  0.00203769,  0.00450548,  0.06122994])



def angle(center, camera_matrix, dist_coeff):
	xpix=center[0]
	ypix=center[1]
	horizontalangleofview=84.0
	resolutionx=1280
	resolutiony=720
	degperpixel=float(horizontalangleofview/resolutionx)
	test2 = np.zeros((1,1,2), dtype=np.float32)
	test2[0]=[xpix,ypix]
	newpoint2=cv2.undistortPoints(test2,camera_matrix, dist_coeff, P=camera_matrix)
	x = newpoint2[0][0][0]
	y = newpoint2[0][0][1]
	distance=x-640
	angle=distance*degperpixel
	return angle


def triangle(anglel, angler, separ,wallangle,disp):
	angle3=180-anglel-angler
	global dl
	global dr
	dl=separ/math.sin(math.radians(angle3))*math.sin(math.radians(angler))
	dr=separ/math.sin(math.radians(angle3))*math.sin(math.radians(anglel))
	if (disp==1):
		print (dl, dr, separ)
		file4.write(str(dl) + " " + str(dr) + " ")
	tarea=area(separ,dr,angler)
	alt=tarea/separ
	coord1=coordinate(dl,dr,separ)
	if (disp==1):
		print (coord1)
	xfin=feettomm(coord1[0])
	yfin=math.cos(math.radians(wallangle))*feettomm(coord1[1])
	return (xfin, yfin)


def area(s1,s2,a1):
	return s1*s2*math.sin(math.radians(a1))/2


def actualangle(angle, cam):
	global AL
	global AR
	if (cam=="camr"):
		AR= 90+angle
	if (cam=="caml"):
		AL= 90-angle


def mmtofeet(x):
	x=x*0.00328084
	return x


def feettomm(x):
	x=x/0.00328084
	return x


def coordinate(distance1,distance2,separation):
	try:
		theta = math.acos((math.pow(distance1,2)+math.pow(separation,2)-math.pow(distance2,2))/(2*distance1*separation))
		ydist = math.sin(theta)*distance1
		botdist = math.cos(theta)*distance1
		xdist = botdist - separation/2
		xdist=mmtofeet(xdist)
		ydist=mmtofeet(ydist)
		return (xdist,ydist)
	except:
		return "error"


current=0

def getcoord(boxr,boxl,location):

	global current
	current+=1

	xr=boxr[0]
	yr=boxr[1]
	wr=boxr[2]
	hr=boxr[3]
	xl=boxl[0]
	yl=boxl[1]
	wl=boxl[2]
	hl=boxl[3]

	if (location=="center"):
		centerr=(xr+wr/2,yr+hr/2)
		centerl=(xl+wl/2,yl+hl/2)

	elif (location=="top"):
		centerr=(xr+wr/2,yr)
		centerl=(xl+wl/2,yl)
		
	angr=angle(centerr, camera_matrixr, dist_coeffr)
	angl=angle(centerl, camera_matrixl, dist_coeffl)

	angr=90+angr
	angl=90-angl

	global coordlist
	global anglist
	global angrlist
	global distllist
	global distrlist

	coordtemp=triangle(angl,angr,mmtofeet(sep),wa,0)

	if ((len(coordlist)>=1) and ((abs(coordlist[len(coordlist)-1][0]-coordtemp[0])>5) or (abs(coordlist[len(coordlist)-1][1]-coordtemp[1])>5))):
		return -1

	coordlist.append(coordtemp)
	angllist.append(angl)
	angrlist.append(angr)
	distllist.append(dl)
	distrlist.append(dr)
	count.append(current)

	return coordtemp


#firstbox=(200,800,100,100)
#secondbox=(200,800,800,400)

#print (getcoord(firstbox,secondbox))
#print(angllist,angrlist)
#print(coordlist)




