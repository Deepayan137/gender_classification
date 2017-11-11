import os
import sys
from aux.data_loader import images_and_truths, pair_to_unit
from pre.preproc import Preprocess
from pre.dim_reduce import pca_decmp
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import json
import cv2
import pdb
from sklearn import metrics
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
acc_score = 0
if __name__ == '__main__':
	config = json.load(open(sys.argv[1]))
	image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
	pairs = images_and_truths(image_locs)
	images, truths = pair_to_unit(pairs)
	pr = Preprocess(haar_path=config["haar_path"])
	norm = np.array([pr.normalize(image) for image in images])
	norm = norm.reshape(100, -1)
	X_train, X_test, y_train, y_test = train_test_split(norm, truths, test_size=0.4, random_state=42, stratify=truths)
	X_train_pca, X_test_pca = pca_decmp(X_train, X_test)
	for i in range(10):
            if i == 0:
                count = 1
	    else:
		metric = ['minkowski', 'euclidean']
	        model = KNeighborsClassifier(n_neighbors=i, 
	                weights='uniform', algorithm='auto', leaf_size=30, p=2, metric=metric[1], metric_params=None, n_jobs=1)
	        model.fit(X_train_pca, y_train)
	        y_pred = model.predict(X_test_pca)
	        acc_score = accuracy_score(y_test, y_pred)*100
	        print(" Accuracy for KNN with NN: %d and metric %s is %f" % (i, metric[1], acc_score))
 	   


