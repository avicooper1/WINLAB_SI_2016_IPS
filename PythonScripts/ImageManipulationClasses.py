import cv2

class Image:
	def __init__ (self, image, roi):
		try:
			self.image = image
			self.roi = roi
			for singleRoi in roi:
				
		except:
			pass
class ROI:
	def __init__ (self, roi):
		self.x = roi[0]
		self.y = roi[1]
		self.w = roi[2]
		self.h = roi[3]
		self.r = float(self.h) / self.w

class Pt:
	def __init__ (self, pt):
		self.x = pt[0]
		self.y = pt[1]