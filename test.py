import os
import sys
from aux.data_loader import images_and_truths, pair_to_unit
from pre.preproc import Preprocess
from pre.dim_reduce import pca_decmp, pca
from sklearn.model_selection import train_test_split
import numpy as np
import json
import cv2
import pdb
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from aux.visualize import plot, plot_stats
from random import shuffle
import pandas as pd
if __name__ == '__main__':
	config = json.load(open(sys.argv[1]))
	image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
	pairs = images_and_truths(image_locs)
	images, truths = pair_to_unit(pairs)
	pr = Preprocess(haar_path=config["haar_path"])
	norm = np.array([pr.normalize(image) for image in images])
	norm = norm.reshape(100, -1)
	n_comp = [20, 25, 35, 45, 60]
	row = []
	for comp in n_comp:
		X_train, X_test, y_train, y_test = train_test_split(norm, truths, test_size=0.20, random_state=42, stratify=truths)
		X_train_pca, X_test_pca, eigenfaces = pca_decmp(X_train, X_test, comp)

		# plot(eigenfaces, output="eigen.png")
		print("Fitting the classifier to the training set")
		t0 = time()
		clf = 	SVC(kernel='rbf', class_weight='balanced')
		clf = clf.fit(X_train_pca, y_train)
		print("done in %0.3fs" % (time() - t0))
		print("Best estimator found by grid search:")
		
		print("Predicting people's names on the test set")
		t0 = time()
		y_pred = clf.predict(X_test_pca)
		sh = list(range(len(X_test)))
		shuffle(sh)
		X_test = [np.reshape(X_test[sh[i]],(48,48)) for i in range(len(sh))]
		y_ = [y_pred[sh[i]] for i in range(len(sh))]
		# plot(X_test, output="predictions.png", predictions=y_)
		acc = (accuracy_score(y_test, y_pred))
		pr = precision_score(y_test, y_pred, average='weighted')
		re = recall_score(y_test, y_pred, average='weighted')
		f1 = 2*(pr*re)/float((pr+re))
		print('f1-score: %.2f'%f1)
		print("done in %0.3fs" % (time() - t0))
		row.append([comp, acc, f1])
	df = pd.DataFrame(row, columns=['n_components', 'Accuracy', 'F1-score'])
	plot_stats(df)