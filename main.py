import argparse
import numpy as np
import time

from transfer_learning import DataLoader
from predictors import predictors
from augmenter import Augmentation

def main():
	
	start = time.time()

	parser = argparse.ArgumentParser()
	

	parser.add_argument('--model', type = str, default = 'ResNet50',
						help = 'model to use for transfer learning\n Available:\
						ResNet50 \
						Xception \
						VGG16/19 \
						DenseNet \
						InceptionV3 etc.\
						look for models available in keras official documentation.')
	
	args = parser.parse_args()
	
	augmenter = Augmentation(model = args.model)
	#augmenter.data_object.plot_first_n_images()
	train_data_aug, train_labels_aug, bottleneck_features_train = augmenter.save_bottleneck_features(
		train = True)
	
	test_data_aug, test_labels_aug, bottleneck_features_test = augmenter.save_bottleneck_features(
		train = False)
	reduced_dims = 2000
	
	train_data_flat_pca = predictors.pca(predictors.pca(np.reshape(
        bottleneck_features_train,
        (array.shape[0],-1)),reduced_dims))
	test_data_flat_pca = predictors.pca(predictors.pca(np.reshape(
       bottleneck_features_test,
       (array.shape[0],-1)),reduced_dims))
	
	predictors.calls(train_data_flat_pca,train_labels_aug,
		x_eval = test_data_flat_pca,
		y_eval = test_labels_aug)
	
	

if __name__ == '__main__':
	main()