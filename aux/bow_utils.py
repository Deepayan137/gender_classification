import cv2
import numpy as np 
from glob import glob
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from time import time
import sys
from sklearn.feature_extraction.image import extract_patches_2d
from math import floor
from matplotlib import pyplot as plt

class FeatureGetter:
    def __init__(self, type):
        if type=='sift':
            self.feat_obj = cv2.xfeatures2d.SIFT_create()

    def get_features(self,image):
        if type=='sift':
            kp,desc = self.features_sift(image)
        else:
            desc = self.features_patches(image)

        return desc

    def features_sift(self, image):
        keypoints, descriptors = self.feat_obj.detectAndCompute(image.astype('uint8'), None)
        return [keypoints, descriptors]

    def features_patches(self, image):
        sz = 8
        patches = extract_patches_2d(image,(sz,sz))
        patches = patches.reshape(-1,sz**2)
        return patches


class BOWHelpers:
    def __init__(self):
        self.vocab_size = None
        self.kmeans_obj = None
        self.kmeans_ret = None
        self.descriptor_all = None
        self.vocab_hist_train = None
        self.vocab_scaler = None
        self.clf = SVC()   


    def format_descriptors(self, desc_init):
        # desc_all = np.array([]).reshape(0,128)
        # print(desc_init[0].shape[1])
        desc_all = np.array([]).reshape(0,desc_init[0].shape[1])

        for desc in desc_init:
            desc_all = np.concatenate((desc_all, desc), axis=0)
        
        self.descriptor_all = desc_all.copy()
        print("final_desc_shape:",self.descriptor_all.shape)
        
        self.vocab_size = int(self.descriptor_all.shape[0]/100)
        # self.kmeans_obj = KMeans(n_clusters=self.vocab_size)
        self.kmeans_obj = MiniBatchKMeans(n_clusters=self.vocab_size, 
            batch_size=int(self.vocab_size/10), 
            init_size=int(1.1*self.vocab_size))

        return self.vocab_size    


    def cluster_descriptors(self):
        print("Clustering Descriptors...")
        print("vocab_size:",self.vocab_size)
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_all)
        # print(self.kmeans_ret)
        print("Done clustering")


    def generateVocabulary(self,images_count, descriptor_list):
        
        print("Generating Vocabulary Histogram...")
        
        self.vocab_hist_train = np.array([np.zeros(self.vocab_size) for i in range(images_count)])
        desc_count_total = 0

        for i in range(images_count):
            desc_count_image = len(descriptor_list[i])                  #no. of desc for i-th image
            sys.stdout.write("\r- Building histograms: %d%%" %(floor((i+1)/images_count *100)))
            sys.stdout.flush()
            # print(desc_count)
            for desc_id in range(desc_count_image):
                word = self.kmeans_ret[desc_count_total + desc_id]      #word for a desc of i-th image
                self.vocab_hist_train[i][word] += 1                           #incrementing the hist entry related to word
            desc_count_total += desc_count_image

        print ("\nDone Generating Vocabulary Histogram")



    def normalizeVocabulary(self, std=None):
        # print("before standardize:",[row[0] for row in self.vocab_hist_train])
        self.vocab_scaler = StandardScaler().fit(self.vocab_hist_train)
        # print("mean:",self.vocab_scaler.mean_[0],"var:",self.vocab_scaler.var_[0])
        self.vocab_hist_train = self.vocab_scaler.transform(self.vocab_hist_train)
        # print("after standardize:",[row[0] for row in self.vocab_hist_train])
        # self.plotHist()


    def train(self, y_train):
        print ("Training SVM")
        print (self.clf)
        # print ("Train labels", y_train)
        self.clf.fit(self.vocab_hist_train, y_train)
        print ("Training completed")


    def predict(self, vocab_hist):
        y_pred = self.clf.predict(vocab_hist)
        return y_pred


    def plotHist(self, vocabulary=None):
        print ("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.vocab_hist_train

        x_scalar = np.arange(self.vocab_size)
        y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.vocab_size)])

        print (y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

