import os 
import sys
import time 
import warnings
import h5py
import argparse
import logging
import pickle

import numpy as np
import pandas as pd

import scipy.io as sio

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split as splitter
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


#%matplotlib inline


class DataLoader(object):

	def __init__(self):
		
		self.image_dir = os.sep+'images'+os.sep
		self.labels = os.sep+'labels'+os.sep+'category_GT.mat'
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		self.train_data = None
		self.train_labels = None

	def labelLoader(self):

		mat_contents = sio.loadmat(os.getcwd() + self.labels)['GT']
		self.train_labels=np.array(mat_contents)

		
		file_list = [f for f in os.listdir(os.getcwd() 
			+ self.image_dir) if os.path.isfile(os.path.join(
				os.getcwd() + self.image_dir, f))]
		file_list.sort()
		inputShape = (224, 224)
		img_list =[]

		for file in file_list:
			temp = os.getcwd() + self.image_dir + "/" + file
			#print filename
			
			#print("[INFO] loading and pre-processing image...")
			image = load_img(temp, target_size=inputShape)
			#print (image.size)
			image = img_to_array(image)
			pos = int(file.split(".")[0])
			img_list.insert(pos -1 , image)
			
		self.train_data = np.array(img_list)
		self.train_data /= 255 # normalizing pixel intensities
		
	
	def dropData(self,inputs,labels):
		self.train_data = inputs[np.isfinite(np.ravel(labels))]
		self.train_labels = labels[np.isfinite(labels)]
		
	def calls(self):
		
		self.labelLoader()
		self.dropData(self.train_data,self.train_labels)
		self.x_train,self.x_test,self.y_train,self.y_test = splitter(self.train_data,self.train_labels,
											test_size = 0.33,
											random_state = 14,#to get reproducible splits
											shuffle = True)
		print('Training Data:{}\
			Validation Data:{}'.format(self.x_train.shape,self.x_test.shape))
	
	def plot_first_n_images(self,img_list=None,n=9):

		if not img_list:
			img_list = self.train_data
		plt.figure(figsize = (10,10))

		for i in range(0, n):
			plt.subplot(330 + 1 + i)
			plt.imshow(img_list[i])
		plt.show()
