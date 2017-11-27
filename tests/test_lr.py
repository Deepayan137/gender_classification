import os
import sys
from aux.data_loader import images_and_truths, pair_to_unit
from pre.preproc import Preprocess
from pre.dim_reduce import pca_decmp
#from sklearn.model_selection import train_test_split
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
	penalty = ['l1','l2']
	param_grid = {'C': [0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000]}
	clf = GridSearchCV(linear_model.LogisticRegression(penalty=penalty[1], class_weight='balanced'), param_grid)
	clf = clf.fit(X_train_pca,y_train)
	y_pred = clf.predict(X_test_pca) 
	print(metrics.classification_report(y_test, y_pred))
	print(metrics.confusion_matrix(y_test, y_pred))
	acc_score = accuracy_score(y_test,y_pred)
	print("The accuracy %f with Logistic Regression having penalty %s",(acc_score,penalty[1]))
 	       
