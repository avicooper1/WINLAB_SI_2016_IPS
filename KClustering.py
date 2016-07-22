import numpy as np
import cv2
from matplotlib import pyplot as plt

dimensions = np.empty(740 - 121)

for x in range(121, 740):
	currentPic = cv2.imread('/home/avi/Desktop/positivesTest/' + str(x) + '-1.jpg')
	try:
		height, width, channels = currentPic.shape
		ratio = height / float(width)
		#if ratio > 0.5 or ratio < 5:
		dimensions[x - 121] = ratio
		print ratio
	except:
		pass
	print(x)

dimensions = np.float32(dimensions)

numberOfClusters = 5

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,centers=cv2.kmeans(dimensions,numberOfClusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

clusters = [[]]

for x in range(0, numberOfClusters):
	clusters.append(dimensions[label.ravel()==x])

for x in range (0,numberOfClusters):
	print centers[x],
	print":",
	print len(clusters[x])

# Plot the data
plt.hist(clusters[0],color = 'r')
plt.hist(clusters[1],color = 'k')
plt.hist(clusters[3],color = 'y')
plt.hist(clusters[4],color = 'g')
plt.hist(clusters[5],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.xlim([0, 5])
plt.xlabel('Ratio'),plt.ylabel('Freq.')
plt.savefig('/home/avi/Desktop/foo.png')
cv2.imshow('plot', cv2.imread('/home/avi/Desktop/foo.png'))
cv2.waitKey()