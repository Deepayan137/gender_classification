import cv2
import numpy as np 
from math import log10
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from skimage.util import view_as_windows
from matplotlib import pyplot as plt


class Preprocessor:
    def __init__(self, size):
        self.max_size = size

    def give_X_y(self,a):
        y=[]
        X=[]
        for label,images in a.items():
            # print(label)
            for img in images:
                img = np.array(img)
                y.append(label)
                X.append(img)
        y = np.array(y)
        X = np.array(X)
        return X,y

    def resize(self, img):
        img = np.array(img)
        scale_x = self.max_size/img.shape[1]
        scale_y = self.max_size/img.shape[0]
        scale = min(scale_x,scale_y)
        final = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # print('initial:',img.shape,'final:',final.shape)
        return final

    def normalize(self, img):
        img = cv2.equalizeHist(img)
        img = img.astype('float64')
        mean, std = np.mean(img), np.std(img)
        img -= mean
        # print(mean,std)
        if std !=0 :
            img /= std
        # if std==0:
        #     cv2.imshow('zero std',img)
        #     cv2.waitKey(-1)

        return img

class FeatureGetter:
    def __init__(self, type):
        self.count =0
        if type=='sift':
            self.feat_obj = cv2.xfeatures2d.SIFT_create()

    def get_features(self,img):
        if type=='sift':
            kp,desc = self.features_sift(img)
        else:
            desc = self.features_patches(img)

        return desc

    def features_sift(self, img):
        keypoints, descriptors = self.feat_obj.detectAndCompute(img.astype('uint8'), None)
        return [keypoints, descriptors]

    def features_patches(self, img):
        sz = 10
        # patches = extract_patches_2d(img,(sz,sz))
        patches = view_as_windows(img,sz,sz)
        patches = patches.reshape(-1,sz**2)
        self.relevent_patches(patches)
        return patches

    def relevent_patches(self, patches):
        # count = 0
        for patch in patches:
            sharpness = cv2.Laplacian(patch, cv2.CV_64F).var()
            # print(sharpness, end='\t')
            if sharpness>0.5:
                self.count+=1



class BOWHelpers:
    def __init__(self):
        self.vocab_size = None
        self.kmeans_obj = None
        self.kmeans_ret = None
        self.descriptor_all = None
        self.vocab_hist_train = None
        self.vocab_scaler = None
        self.tf_vocab = None
        self.tfidf_scores = None
        self.clf = SVC()   


    def format_descriptors(self, desc_init, vocab_sz=None):
        # desc_all = np.array([]).reshape(0,128)
        # print(desc_init[0].shape[1])
        desc_all = np.array([]).reshape(0,desc_init[0].shape[1])

        for desc in desc_init:
            desc_all = np.concatenate((desc_all, desc), axis=0)
        
        self.descriptor_all = desc_all.copy()
        # print("final_desc_shape:",self.descriptor_all.shape)
        
        if vocab_sz is None:
            self.vocab_size = int(self.descriptor_all.shape[0]/100)
        else:
            self.vocab_size = vocab_sz
        # self.kmeans_obj = KMeans(n_clusters=self.vocab_size)
        self.kmeans_obj = MiniBatchKMeans(n_clusters=self.vocab_size, 
            batch_size=int(self.vocab_size/10), 
            init_size=int(1.1*self.vocab_size))

        return self.vocab_size    


    def cluster_descriptors(self):
        # print("Clustering Descriptors...")
        # print("vocab_size:",self.vocab_size)
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_all)
        # print(self.kmeans_ret)
        # print("Done clustering")


    def generateVocabulary(self,images_count, desc_list):
        
        # print("Generating Vocabulary Histogram...")
        
        self.vocab_hist_train = np.array([np.zeros(self.vocab_size) for i in range(images_count)])
        desc_count_total = 0

        for i in range(images_count):
            desc_count_image = len(desc_list[i])                  #no. of desc for i-th image
            # sys.stdout.write("\r- Building histograms: %d%%" %(floor((i+1)/images_count *100)))
            # sys.stdout.flush()
            # print(desc_count)
            for desc_id in range(desc_count_image):
                word = self.kmeans_ret[desc_count_total + desc_id]      #word for a desc of i-th image
                self.vocab_hist_train[i][word] += 1                           #incrementing the hist entry related to word
            desc_count_total += desc_count_image
        
        self.getTFvocab()
        # print ("\nDone Generating Vocabulary Histogram")



    def normalizeVocabulary(self, std=None):
        # print("before standardize:",[row[0] for row in self.vocab_hist_train])
        self.vocab_scaler = StandardScaler().fit(self.vocab_hist_train)
        # print("mean:",self.vocab_scaler.mean_[0],"var:",self.vocab_scaler.var_[0])
        self.vocab_hist_train = self.vocab_scaler.transform(self.vocab_hist_train)
        # print("after standardize:",[row[0] for row in self.vocab_hist_train])
        # self.plotHist()
        # self.getTFvocab()

    def getTFvocab(self):
        self.tf_vocab = normalize(self.vocab_hist_train,axis=0,norm='l1')
        print(self.tf_vocab.shape)


    def predict_tfidf(self, vocab_hist_retrieved, N, y_train):
        self.tfidf_scores = [0]*N
        sz = np.count_nonzero(self.tf_vocab,axis=1)     #num of images that each word is present in  
        # print(vocab_hist_retrieved.shape)
        for word_id_retrieved in range(vocab_hist_retrieved.shape[1]):
            word_count = vocab_hist_retrieved[0,word_id_retrieved]
            # print(word_id_retrieved, word_count)
            if word_count==0:
                continue
            idf = log10(N/sz[word_id_retrieved])
            for img_id in range(N):
                self.tfidf_scores[img_id] += self.tf_vocab[img_id,word_id_retrieved] * idf

        class_id = np.argmax(self.tfidf_scores)
        y_pred = y_train[class_id]
        return y_pred




    def train(self, y_train):
        # print ("Training SVM")
        # print (self.clf)
        # print ("Train labels", y_train)
        self.clf.fit(self.vocab_hist_train, y_train)
        # print ("Training completed")


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

