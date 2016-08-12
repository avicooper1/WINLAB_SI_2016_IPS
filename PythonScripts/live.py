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
from scipy.optimize import fsolve


rslt = []
rslt2= []
coordlist= []

sep = 2400
D1 = 10
D2 = 10
AL=10
AR=10
templ=0
tempr=0

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





def angle(xpix,ypix, camera_matrix, dist_coeff):
	horizontalangleofview=84.0
	resolutionx=1280
	resolutiony=800
	degperpixel=float(horizontalangleofview/resolutionx)
	test2 = np.zeros((1,1,2), dtype=np.float32)
	test2[0]=[xpix,ypix]
	newpoint2=cv2.undistortPoints(test2,camera_matrixr, dist_coeffr, P=camera_matrixr)
	x = newpoint2[0][0][0]
	y = newpoint2[0][0][1]
	distance=x-640
	angle=distance*degperpixel
	return angle


def triangle(anglel, angler, separ,wallangle,disp):
	angle3=180-anglel-angler
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


def distance(height,sensor,focal,hoc):
	distanceA = height*focal/sensor
	distanceB = math.sqrt(math.pow(distanceA,2) - math.pow(hoc-height/2,2))
	return distanceB


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


def sensor(focal, fx, fy, pixel):
	mx=fx/focal
	my=fy/focal
	m=(mx+my)/2
	sensorsize=pixel/m
	sensorsize=sensorsize*720/800
	return sensorsize


def height(boxsize):
	ss=sensor(focallength, fx, fy, boxsize)
	func = lambda h :  camrtodoor**2- h**2*focallength**2/ss**2+(cameraheight-h)**2/4
	guess=1700
	solution=fsolve(func,guess)
	return solution
	

def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i]-cnt2[j])
			if abs(dist) < 1000 :
				return True
			elif i==row1-1 and j==row2-1:
				return False


cap2 = cv2.VideoCapture('http://192.168.206.175/mjpg/video.mjpg')
cap = cv2.VideoCapture('http://192.168.206.241/mjpg/video.mjpg')

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()

counter = -1
startTime = datetime.datetime.now()


def processImage(frame, conts, count, cam):
	LENGTH = len(conts)
	status = np.zeros((LENGTH,1))
	i=0
	for i,cnt1 in enumerate(conts):
		x = i    
		if i != LENGTH-1:
			for j,cnt2 in enumerate(conts[i+1:]):
				x = x+1
				dist = find_if_close(cnt1,cnt2)
				if dist == True:
					val = min(status[i],status[x])
					status[x] = status[i] = val
				else:
					if status[x]==status[i]:
						status[x] = i+1
	unified = []
	maximum = int(status.max())+1



	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(conts[i] for i in pos)
			hull = cv2.convexHull(cont)
			x,y,w,h = cv2.boundingRect(hull)
	
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)


			if (cam=="caml"):
				ang=angle(x+w/2,y+h/2, camera_matrixl, dist_coeffl)
			elif (cam=="camr"):
				ang=angle(x+w/2,y+h/2, camera_matrixr, dist_coeffr)

			actualangle(ang,cam)
			
			global tempr
			global templ
			if (cam=="camr"):			
				tempr = (x+w/2,y+h/2)
			elif (cam=="caml"):
				templ=(x+w/2,y+h/2)



file = open("/home/phalpha/Desktop/stereo/datz.txt", "w")
file4 = open("/home/phalpha/Desktop/stereo/DATTlive.txt", "w")

subcounter=0
while(True):
	ret, frame = cap.read()
	ret2, frame2 = cap2.read()

	if not ret or not ret2:
		break
	else:
		counter+=1
		#cv2.imwrite("/home/phalpha/Desktop/stereo/saveimagesr/{}.jpg".format(counter), frame)
		#cv2.imwrite("/home/phalpha/Desktop/stereo/saveimagesl/{}.jpg".format(counter), frame2)
		fgmask = fgbg.apply(frame)
		fgmask2 = fgbg2.apply(frame2)
		fgmask = cv2.morphologyEx(fgmask, 	cv2.MORPH_OPEN, kernel)
		fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel2)
		a,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		b, contours2, heirarchy2 = cv2.findContours(fgmask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		if len(contours2 and contours) > 0:
			subcounter+=1
			processImage(frame, contours, counter, "camr")
			processImage(frame2, contours2, counter, "caml")
			
			coordtemp=triangle(AL,AR,mmtofeet(sep),0,0)

			#if ((len(coordlist)>=1) and ((abs(coordlist[len(coordlist)-1][0]-coordtemp[0])>5) or (abs(coordlist[len(coordlist)-1][1]-coordtemp[1])>5))):
			#	continue
			#coordlist.append(coordtemp)
			#if (len(coordlist)>1):
			#	print (abs(coordlist[len(coordlist)-2][0]-coordtemp[0]))

			coordlist.append(coordtemp)
			#print (abs(coordlist[len(coordlist)-2][1]-coordtemp[1]))
			print(AL, AR)
			coord2=triangle(AL,AR,mmtofeet(sep),0,1)
			file.write(str(coord2) + " " + str(counter))
			file4.write(str(coord2[0])+ " " + str(coord2[1]) + " " + str(counter))
			file.write("\n")
			file4.write("\n")
			print(coord2)
			print(counter)
			print("\n")

			
			while(len(rslt)>32):
				del rslt[0]
			rslt.append(tempr)

			size=len(rslt)
			for x in range(size-1):
				cv2.line(frame, rslt[x], rslt[x+1], (255,255,255), 4, 4, 0)

			
			while(len(rslt2)>32):
				del rslt2[0]
			rslt2.append(templ)

			size=len(rslt2)
			for x in range(size-1):
				cv2.line(frame2, rslt2[x], rslt2[x+1], (255,255,255), 4, 4, 0)

			#frameline=frame
			#frameline2=frame2
			#cv2.line(frameline,(tempr[0],0), (tempr[0], 799), (255,255,0), 4, 4, 0)
			#cv2.line(frameline2,(templ[0],0), (templ[0], 799), (255,255,0), 4, 4, 0)

			#cv2.imwrite("/home/phalpha/Desktop/stereo/camrline/{}.jpg".format(counter), frameline)
			#cv2.imwrite("/home/phalpha/Desktop/stereo/camlline/{}.jpg".format(counter), frameline2)

			coord3 = ("%.2f" % coord2[0], "%.2f" % coord2[1])
			font=cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame,str(coord3), (tempr), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
			cv2.putText(frame2,str(coord3), (templ), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
			#cv2.imwrite("/home/phalpha/Desktop/stereo/camrv2/{}.jpg".format(counter), frame)
			#cv2.imwrite("/home/phalpha/Desktop/stereo/camlv2/{}.jpg".format(counter), frame2)
		cv2.imshow("frame", frame)
		cv2.waitKey(1)

file.close()
file4.close()
cap.release()
cap2.release()
cv2.destroyAllWindows()


endTime = datetime.datetime.now()
runTime = endTime - startTime
print("Total run time was {} seconds".format(runTime))
print("done")