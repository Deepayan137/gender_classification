import os
import sys
from aux.data_loader import images_and_truths, pair_to_unit, reorder_dict
from pre.preproc import Preprocess
from pre.dim_reduce import pca_decmp, pca
from sklearn.model_selection import train_test_split
import numpy as np
import json
import cv2
import pdb
import pickle

from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from aux.visualize import plot, plot_stats
from random import shuffle
import pandas as pd
from argparse import ArgumentParser
from aux.opts import base_opts
if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config = json.load(open(args.config))
	if args.data == 'facescrub':
		with open('FaceScrub_Data.pickle' , 'rb') as handle:
			data = pickle.load(handle,encoding='latin1')
		pairs=reorder_dict(data)
		n_comp = [20,50, 100, 200, 300, 400]
		images, truths = pair_to_unit(pairs)
	if args.data == 'old':
		image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
		pairs = images_and_truths(image_locs)
		images, truths = pair_to_unit(pairs)
		pr = Preprocess(haar_path=config["haar_path"])
		images = np.array([pr.normalize(image) for image in images])
		images = images.reshape(100, -1)
		n_comp = [5, 15, 20, 25, 35, 45]
	# 
	row = []
	acc = 0
	for comp in n_comp:
		X_train, X_test, y_train, y_test = train_test_split(images, truths, test_size=0.20, random_state=42, stratify=truths)
		X_train, X_test = np.array(X_train), np.array(X_test)
		# pdb.set_trace()
		X_train_pca, X_test_pca, eigenfaces = pca_decmp(X_train, X_test, comp)
		
		# plot(eigenfaces, output="eigen_%d.png"%comp)
		print("Fitting the classifier to the training set")
		t0 = time()
		clf = 	SVC(kernel='rbf', class_weight='balanced')
		clf = clf.fit(X_train_pca, y_train)
		print("done in %0.3fs" % (time() - t0))
		#print("Predicting people's names on the test set")
		t0 = time()
		y_pred = clf.predict(X_test_pca)
		sh = list(range(len(X_test)))
		shuffle(sh)
		X_test = [np.reshape(X_test[sh[i]],(48,48)) for i in range(len(sh))]
		y_ = [y_pred[sh[i]] for i in range(len(sh))]
		# pdb.set_trace()
		outpath = os.path.join(args.output,"predictions_fscrub_%d.png"%comp)
		# plot(X_test, output= outpath, predictions=y_)
		if accuracy_score(y_test, y_pred) > acc:
			acc = accuracy_score(y_test, y_pred)
			with open('svm.pkl', 'wb') as f:
				pickle.dump(clf, f, protocol=2)
		pr = precision_score(y_test, y_pred, average='weighted')
		re = recall_score(y_test, y_pred, average='weighted')
		f1 = 2*(pr*re)/float((pr+re))
		print('f1-score: %.2f \n accuracy: %.2f'%(f1, acc))
		#print("done in %0.3fs" % (time() - t0))
		row.append([comp, acc, f1])
	df = pd.DataFrame(row, columns=['n_components', 'Accuracy', 'F1-score'])
	plot_stats(df, args.output)
