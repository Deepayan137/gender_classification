from aux.bow_utils import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold




class BOW:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.desc_available = False
        self.feature_getter = None
        self.feature_type = None
        self.desc_list = []
        # self.desc_count = []
        self.bow_helper = None
        self.vocab_size = None
        self.trainImageCount = 0
        self.prev_f1 = 0
        self.best_model = False

    def trainModel(self):

        self.trainImageCount = X_train.shape[0]
        
        self.feature_getter = FeatureGetter(self.feature_type)

        if self.desc_available:
            with open('desc_train.pickle','rb') as read_desc:
                self.desc_list = pickle.load(read_desc)
        else:
            for image in self.X_train:
                # count = floor((image_count+1)/self.trainImageCount *100)
                # sys.stdout.write("\r- Obtaining descriptors: %d%%" % count)
                # sys.stdout.flush()

                descriptors = self.feature_getter.get_features(image)
                self.desc_list.append(descriptors)
                # self.desc_count.append(descriptors.shape[0])

            with open('desc_train.pickle','wb') as write_desc:
                pickle.dump(self.desc_list, write_desc)
            self.desc_available = True

            print('\nTotal Train features:',str(self.feature_getter.count))

                
        # print("\n")

        self.bow_helper = BOWHelpers()
        self.bow_helper.format_descriptors(self.desc_list, vocab_sz=self.vocab_size)
        # print("image_shape:",image.shape,"descriptors count: ",len(self.desc_list))
        
        self.bow_helper.cluster_descriptors()

        self.bow_helper.generateVocabulary(images_count=self.trainImageCount, desc_list=self.desc_list)
        
        self.bow_helper.normalizeVocabulary()
        print(self.bow_helper.vocab_hist_train.shape)
        self.bow_helper.train(self.y_train)

        self.validateModel()

        if self.best_model:
            joblib.dump((self.bow_helper.clf, self.y_train, self.bow_helper.vocab_scaler, self.bow_helper.kmeans_obj,
            self.vocab_size, self.bow_helper.vocab_hist_train), "bow_best.model", compress=3)   


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
        # y_pred = self.bow_helper.predict_tfidf(vocab_hist,self.trainImageCount,self.y_train)

        # print("Image belongs to class :",y_pred) 
        return y_pred


    def validateModel(self):
        predictions = []
        image_count = 0

        y_pred = []
        gender = lambda x: "female" if x=='f' else "male"
        for image in self.X_val:
            pred = self.recognize(image)
            # print('pred: ', gender(pred[0]), 'y_val', gender(self.y_val[image_count]))
            # print('pred: ', pred[0], 'y_val', self.y_val[image_count])
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
        print("vocab_size:",self.vocab_size,
         "accuracy:",accuracy,"precision:",precision,
         "recall:",recall,"f1:",f1)


        sv_metrics = open('bow_val_results.txt','a')
        sv_metrics.write("vocab_size:" + str(self.vocab_size) + "\taccuracy:" + str(accuracy) +
            "\tprecision:" + str(precision) + "\trecall:" + str(recall) + "\tf1:" + str(f1)+"\n")
        sv_metrics.close()

        if f1 > self.prev_f1:
            self.best_model = True
        else:
            self.best_model = False



    def testModel(self):
        predictions = []
        image_count = 0

        self.feature_getter = FeatureGetter(self.feature_type)
        self.bow_helper = BOWHelpers()

        clf, classes_names, scaler, kmeans_obj, vocab_size, vocab = joblib.load("bow_best.model")
        self.bow_helper.clf = clf
        self.bow_helper.vocab_scaler = scaler
        self.bow_helper.kmeans_obj = kmeans_obj
        self.bow_helper.vocab_hist_train = vocab
        self.bow_helper.vocab_size = vocab_size

        y_pred = []
        gender = lambda x: "female" if x=='f' else "male"
        for image in self.X_test:
            pred = self.recognize(image)
            # print('pred: ', gender(pred[0]), 'y_test', gender(self.y_test[image_count]))
            predictions.append({
                'image':self.X_test[image_count],
                'y_pred':gender(pred[0]),
                'y_test':gender(self.y_test[image_count])})
            y_pred.append(pred)
            image_count += 1

        # print(predictions)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='macro')
        print("vocab_size:",self.bow_helper.vocab_size,"accuracy:",accuracy,
            "precision:",precision,"recall:",recall,"f1:",f1)

        sv_metrics = open('bow_test_results.txt','a')
        sv_metrics.write("vocab_size:" + str(self.bow_helper.vocab_size) + "\taccuracy:" + str(accuracy) +
            "\tprecision:" + str(precision) + "\trecall:" + str(recall) + "\tf1:" + str(f1)+"\n")
        sv_metrics.close()
        # for each_pred in predictions:
        #     cv2.imshow(each_pred['y_pred'], each_pred['image'])
        #     cv2.waitKey()



if __name__ == '__main__':
    
    pkl_file = open('FaceScrub_Data_raw.pickle', 'rb')
    a = pickle.load(pkl_file, encoding='latin1')
    print(a.keys())
    am = a['m']
    af = a['f']

    # print(len(am))
    # print(len(af))
    # print(af[0][0])
    # cv2.imshow('normal',a['m'][0])
    # cv2.waitKey(0)

    max_img_size = 100
    preprocessor = Preprocessor(size=max_img_size)
    X,y = preprocessor.give_X_y(a)
    print('input:',X.shape)

    X_rsz = np.array([preprocessor.resize(img) for img in X])

    X_norm = np.array([preprocessor.normalize(img) for img in X_rsz])
    # X_norm = X_rsz

    X_t, X_test, y_t, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=42)

    num_fold = 5

    bow = BOW()
    bow.feature_type = 'patches'

    print('Training the classifier')

    for vocab_size in range(500, 20000, 500):
        curr_val = 0
        X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.20, random_state=42)

       
        bow.X_train = X_train
        bow.y_train = y_train

        bow.X_val = X_val
        bow.y_val = y_val
        
        bow.vocab_size = vocab_size
        
        bow.trainModel()
            
            
    bow.X_test = X_test
    bow.y_test = y_test
    print('Testing the classifier')
    bow.testModel()
