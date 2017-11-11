import numpy as np
import pdb
class Math:
	def __init__(self):
		print("hi")
	def covariance(self, A):
		A = A.reshape((290, 290))
		ones_vector = np.ones((1, A.shape[0]), dtype=int)
		mu = np.dot(ones_vector, A)/float(A.shape[0])
		A_mean = A - np.repeat(mu, A.shape[0], axis=0)
		covA = np.dot(A_mean.T, A_mean) / (A.shape[0] -1)
		
		return covA.reshape(1, -1)

	def eig(self, X, num_components):
		n, d = X.shape
		if (num_components <= 0) or (num_components >n):
			num_components = n
		mu = X.mean(axis =0)
		X = X - mu
		C = np.dot(X.T,X)
		# C = []
		# for x in X:
		# 	C.append(self.covariance(x))
		# C = np.array(C)
		pdb.set_trace()
		[eigenvalues ,eigenvectors] = np.linalg.eigh(C)
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:,idx]
		idx = [i[0] for i in sorted(enumerate(eigenvalues), key=lambda x:-x[1])]
		return (eigenvalues[:num_components], eigenvectors[:,:num_components], mu)


	def project(self, W, X, mu=None):
		if mu is None:
			return np.dot(X, W)
		
		return np.dot((X-mu), W)

	def reconstruct(self, W, Y, mu=None):
		if mu is None:
			return np.dot(Y, W.T)
		
		return np.dot(Y, W.T)+ mu
# c = Math()
# A = np.array([[4, 2, 0.6], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [4.3, 2.1, 0.62],[4.1, 2.2, 0.63]])
# cov = c.covariance(A)

# [eigenvalues ,eigenvectors] = np.linalg.eigh(cov)
# idx = [i[0] for i in sorted(enumerate(eigenvalues), key=lambda x:-x[1])]
# print(eigenvalues)
# print(eigenvalues[idx])
# print(eigenvectors[idx,:])