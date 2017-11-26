import cv2
import os
import numpy as np
from pre.dim_reduce import pca
from sklearn.svm import SVC
from time import time
import pickle
from sklearn.metrics import accuracy_score
def build_SVC(X_train, y_train, X_test, y_test, n_comp):
	comp = n_comp
	pc = pca(X_train, comp)
	X_train_pca = pc.transform(X_train)
	X_test_pca = pc.transform(X_test)
	print("Fitting the classifier to the training set")
	t0 = time()
	clf = 	SVC(kernel='rbf',class_weight='balanced', probability=True)
	clf = clf.fit(X_train_pca, y_train)
	print("done in %0.3fs" % (time() - t0))
	y_pred = clf.predict(X_test_pca)
	prob = clf.predict_proba(X_test_pca)
	with open("svm.pickle", 'wb') as f:
		pickle.dump(clf, f, protocol=2) 
	
	# for item in list(zip(y_pred,y_test, prob)):
	# 	print(item)
	# acc = accuracy_score(y_test, y_pred)
	# print("accuracy: %.2f"%acc)
	return pc, clf

def predict(clf, pca, image):

	# img = image.ravel()
	principle_components = pca.transform(image)
	with open("pca.pickle", 'wb') as f:
		pickle.dump(principle_components, f, protocol=2)
	gender = clf.predict(principle_components)
	prob = clf.predict_proba(principle_components)
	
	return gender, prob