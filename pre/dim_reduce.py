from sklearn.decomposition import PCA 
from time import time
from aux.math import Math
import pdb
<<<<<<< Updated upstream
def pca_decmp(X_train, X_test):
	n_components = 20
	

	
	#pca = PCA(whiten=True).fit(X_train)
	#pca = PCA(n_components=n_components, svd_solver='randomized',
        #whiten=True).fit(X_train)
	
	pca = PCA(n_components=n_components, copy=True, 
		whiten=True, svd_solver='auto', 
		tol=0.0, iterated_power='auto', random_state=None).fit  	(X_train)	
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#pdb.set_trace()
	#h,w = X_train[1].shape
        #eigenfaces = pca.components_.reshape((n_components, h, w))
	return(X_train_pca, X_test_pca)
=======
def pca_decmp(X_train, X_test, n_comp):
	n_components = n_comp
	
	h,w = 290, 290
	pca = PCA(n_components=n_components, whiten=True).fit(X_train)
	#pca = PCA(whiten=True).fit(X_train)
	
	eigenfaces = pca.components_.reshape((n_components, h, w))
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#pdb.set_trace()
	return(X_train_pca, X_test_pca, eigenfaces)

def pca(X, num_components):
	m = Math()
	eig_value, eig_vector, mu = m.eig(X, num_components)
	return eig_vector
>>>>>>> Stashed changes
