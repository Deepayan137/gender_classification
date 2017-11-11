from sklearn.decomposition import PCA 
from time import time
import pdb
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
