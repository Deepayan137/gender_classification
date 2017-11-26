import cv2
import os
import json
import pdb
import sys
from skimage import exposure
from skimage.exposure import rescale_intensity
import numpy as np
class Preprocess:
	def __init__(self, *args, **kwargs):
		#pdb.set_trace()

		if 'haar_path' in kwargs:
			self.haar_path = kwargs['haar_path']
	def detect_faces(self, image):
		minisize = (image.shape[1],image.shape[0])
		miniframe = cv2.resize(image, minisize)
		#min_size = (30, 30)
		face_cascade = cv2.CascadeClassifier(self.haar_path)
		faces = face_cascade.detectMultiScale(miniframe, 1.3, 5)
		x, y, w, h = [v for f in faces for v in f] 
		return image[y:y+h, x:x+w]
	def pad_image(self, image, bordersize):
		border=cv2.copyMakeBorder(im, 
			top=bordersize, bottom=bordersize, 
			left=bordersize, right=bordersize, 
			borderType= BORDER_REPLICATE, 
			)
	def normalize(self, image):
		#image = image/255

		image = self.detect_faces(image)

		image = cv2.equalizeHist(image)
		image = image.astype('float64')
		try:
			image = cv2.resize(image, (48, 48))
		except Exceptions as e:
			pad_image(image, (48-image.shape[0], 48-image.shape[1]))
		#mean, std = np.mean(image), np.std(image)
		#image -= mean
		#image /= std
		cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
		# print(image.shape)
		I = np.reshape(image,(1,-1))
		# print(I.shape)
		return I

	def normalize_frame(self, image):
		#image = image/255

		# image = self.detect_faces(image)

		image = cv2.equalizeHist(image)
		image = image.astype('float64')
		try:
			image = cv2.resize(image, (48, 48))
		except Exceptions as e:
			pad_image(image, (48-image.shape[0], 48-image.shape[1]))
		#mean, std = np.mean(image), np.std(image)
		#image -= mean
		#image /= std
		cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
		# print(image.shape)
		I = np.reshape(image,(1,-1))
		# print(I.shape)
		return I

	def intensity_rescale(self, image):
		return rescale_intensity(image, out_range=(0, 255))




# if __name__ == '__main__':

# 	config = json.load(open(sys.argv[1]))
# 	image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
	
# 	images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY) for image in image_locs[:2]]
# 	pr = Preprocess(haar_path=config["haar_path"])
	
# 	pdb.set_trace()
# 	norm = np.array([pr.normalize(image) for image in images]) 
	
# 	norm*=std
# 	norm+=mean
	
# 	cv2.imshow('face', norm.astype('uint8'))
	
# 	cv2.waitKey(0)
