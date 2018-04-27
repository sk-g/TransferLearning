import keras
import numpy as np
import pandas as pd
import os
import sys
import time
import warnings


from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import applications
from keras.optimizers import Adam
from keras.datasets import mnist

from transfer_learning import DataLoader

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Augmentation(object):

	def __init__(self,data_object = None,model = 'ResNet50'):
		
		if not data_object:
			print('Calling data class form augmentation class')
			self.data_object = DataLoader()
			self.data_object.calls()
		
		self.model = str(model)
	
	def save_bottleneck_features(self, 
				filename = None,
				train = True):
		model = self.model
		if train:
			train_data_filter = self.data_object.x_train
			train_labels_filter = self.data_object.y_train
			filename = str('bottleneck_features_train'+str(model)+'.npy')
		else:
			train_data_filter = self.data_object.x_test
			train_labels_filter = self.data_object.y_test
			filename = str('bottleneck_features_test'+str(model)+'.npy')

		train_data_aug=[]
		train_labels_aug=[]
		batch_size = 128
		datagen = ImageDataGenerator(featurewise_center=True,
									 featurewise_std_normalization=True,
									 horizontal_flip=True,
									 fill_mode='nearest')
		
		if model == 'InceptionResNetV2':
			model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,
										 weights='imagenet')
		
		elif model == 'ResNet50':
			model = keras.applications.resnet50.ResNet50(include_top=False, 
														 weights='imagenet')
		

		elif model == 'Xception':
			keras.applications.xception.Xception(include_top=False,
										 weights='imagenet')
		
		elif model == 'VGG16':
			model = keras.applications.vgg16.VGG16(include_top=False,
										 weights='imagenet')

		elif model == 'VGG19':
			model = keras.applications.vgg19.VGG19(include_top=False,
										 weights='imagenet')
		
		elif model == 'InceptionV3':
			model = keras.applications.inception_v3.InceptionV3(include_top=False,
										 weights='imagenet')
		
		elif model == 'DenseNet':
			model = keras.applications.densenet.DenseNet201(include_top=False,
										 weights='imagenet')
		else:
			print('{} not implemented yet!'.format(model))

		print("loading gen on training data")
		
		datagen.fit(train_data_filter)
		
		print("generating augmentations of data")
		bottleneck_features_train =[]
		
		i = 0
		print("Total iterations = {}".format((len(train_data_filter) * 10)//batch_size))
		if os.path.isfile('models'+os.sep+'_'+filename):
			print('Skipping {}, features already saved'.format(self.model))
		
		for X_batch, y_batch in datagen.flow(train_data_filter, 
											 train_labels_filter, 
											 batch_size=batch_size, 
											 shuffle=False):
			
			train_data_aug.extend(X_batch)
			train_labels_aug.extend(y_batch)
			
			#print("in iter ", i)
			
			#print("generating bottleneck features")
			bottleneck_features_train_batch = model.predict(
			X_batch,  verbose = 0)
			
			#print('\nBottleneck feature shape:{}'.format(bottleneck_features_train_batch.shape))
			
			
			bottleneck_features_train.extend(bottleneck_features_train_batch)
			i += 1
			
			if i > (len(train_data_filter) * 10)//batch_size:
				break
				
		bottleneck_features_train = np.array(bottleneck_features_train)
		train_data_aug = np.array(train_data_aug)
		train_labels_aug = np.array(train_labels_aug)
		print('\n_________________\nData Shapes\n_________________\n')
		print('\nBottleneck features:{}'.format(bottleneck_features_train.shape))
		print('\nAugmented Data:{}'.format(train_data_aug.shape))
		print('\nTrain labels:{}'.format(train_labels_aug.shape ))
		print("\nsaving bottleneck features to a file\n")
		if not os.path.isfile('models'+os.sep+'_'+filename):
			np.save(open(str('models'+os.sep+'_'+filename), 'wb'),
				bottleneck_features_train)
		
		return train_data_aug, train_labels_aug, bottleneck_features_train

		
