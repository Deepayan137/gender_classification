from aux.data_loader import images_and_truths, pair_to_unit
from pre.preproc import Preprocess
from aux.bow_utils import *
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sys
import json
import os
import pickle
from math import floor
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def normalize(image, size=100):
    image = cv2.equalizeHist(image)
    image = image.astype('float64')
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    mean, std = np.mean(image), np.std(image)
    image -= mean
    image /= std

    return image


class BOW:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_getter = None
        self.feature_type = None
        self.descriptor_list = []
        self.bow_helper = None
        self.trainImageCount = 0
        self.name_dict = {}

    def trainModel(self):

        self.trainImageCount = X_train.shape[0]
        
        self.feature_getter = FeatureGetter(self.feature_type)

        image_count = 0

        for image in self.X_train:
            count = floor((image_count+1)/self.trainImageCount *100)
            sys.stdout.write("\r- Obtaining descriptors: %d%%" % count)
            sys.stdout.flush()

            descriptors = self.feature_getter.get_features(image)
            self.descriptor_list.append(descriptors)
            image_count +=1

        with open('desc.pickle','wb') as write_desc:
            pickle.dump(self.descriptor_list, write_desc)
            
        print("\n")

        self.bow_helper = BOWHelpers()
        self.bow_helper.format_descriptors(self.descriptor_list)
        # print("image_shape:",image.shape,"descriptors count: ",len(self.descriptor_list))
        
        self.bow_helper.cluster_descriptors()

        self.bow_helper.generateVocabulary(images_count=self.trainImageCount, descriptor_list=self.descriptor_list)
        
        self.bow_helper.normalizeVocabulary()
        self.bow_helper.train(self.y_train)


    def recognize(self,test_img):
        des = self.feature_getter.get_features(test_img)

        vocab_hist = np.array( [ 0 for i in range(self.bow_helper.vocab_size)])
        
        words = self.bow_helper.kmeans_obj.predict(des)  #we get word_id for each descriptor

        for each_word in words:
            vocab_hist[each_word] += 1

        # Scale the features
        vocab_hist = self.bow_helper.vocab_scaler.transform([vocab_hist])

        # predict the class of the image
        y_pred = self.bow_helper.clf.predict(vocab_hist)
        # print("Image belongs to class :",y_pred) 
        return y_pred


    def testModel(self):
        predictions = []
        image_count = 0

        y_pred = []
        gender = lambda x: "female" if x=='1' else "male"
        for image in self.X_val:
            pred = self.recognize(image)
            # print('pred: ', gender(pred[0]), 'y_val', gender(self.y_val[image_count]))
            predictions.append({
                'image':self.X_val[image_count],
                'y_pred':gender(pred[0]),
                'y_val':gender(self.y_val[image_count])})
            y_pred.append(pred)
            image_count += 1

        # print(predictions)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='macro')
        print("accuracy:",accuracy,"precision:",precision,"recall:",recall,"f1:",f1)

        # for each_pred in predictions:
        #     cv2.imshow(each_pred['y_pred'], each_pred['image'])
        #     cv2.waitKey()



if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))

    image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
    pairs = images_and_truths(image_locs)
    images, truths = pair_to_unit(pairs)
    
    norm = np.array([normalize(image,size=100) for image in images])
    
    X_train, X_val, y_train, y_val = train_test_split(norm, truths, test_size=0.20, random_state=42)
    
    print("Fitting the classifier to the training set")

    bow = BOW()
    bow.feature_type = 'patches'

    bow.X_train = X_train
    bow.y_train = y_train

    bow.X_val = X_val
    bow.y_val = y_val
 
    # train the model
    bow.trainModel()
    # test model
    bow.testModel()
