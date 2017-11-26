from sklearn.decomposition import PCA 
from time import time
from aux.math import Math
import pdb
import pickle
def pca_decmp(X_train, X_test, n_comp):
	n_components = n_comp
	
	h,w = 48, 48
	pca = PCA(n_components=n_components, whiten=True).fit(X_train)
	
	eigenfaces = pca.components_.reshape((n_components, h, w))
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#pdb.set_trace()
	return(X_train_pca, X_test_pca, eigenfaces)

# def pca(X, num_components):
# 	m = Math()
# 	eig_value, eig_vector, mu = m.eig(X, num_components)
# 	return eig_vector
def pca(X_train, n_comp):
	
	pca = PCA(n_components= n_comp, whiten=True).fit(X_train)
		
	return pca