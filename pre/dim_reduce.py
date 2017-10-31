from sklearn.decomposition import PCA
from time import time
import pdb
def pca_decmp(X_train, X_test):
	n_components = 200
	

	
	pca = PCA(whiten=True).fit(X_train)
	
	
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#pdb.set_trace()
	return(X_train_pca, X_test_pca)