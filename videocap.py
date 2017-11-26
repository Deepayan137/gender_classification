import cv2
import json
from pre.preproc import Preprocess
from argparse import ArgumentParser
from aux.opts import base_opts
from aux.build_svm import build_SVC, predict
from pre.dim_reduce import pca
from sklearn.model_selection import train_test_split
from aux.data_loader import images_and_truths, pair_to_unit, reorder_dict
import pickle
import numpy as np
import warnings
import os
import pdb
from random import shuffle
warnings.filterwarnings('ignore')
def videocap(haar_path, clf, pca):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #WIDTH
    cap.set(4, 480) #HEIGHT
    pr = Preprocess(haar_path = haar_path)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        key = cv2.waitKey(1)
        if key in [27, ord('Q'), ord('q')]: 
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Display the resulting frame
        for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
             roi_gray = gray[y:y+h, x:x+w]
             roi_color = frame[y:y+h, x:x+w]
             roi_gray = pr.normalize_frame(roi_gray) 
             face_to_predict = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)
             gender, prob = predict(clf, pca, roi_gray)
             # print(prob)
             if gender[0] == '0' or gender[0] == 'm':
                gender = 'male'
             if gender[0] == '1' or gender[0] == 'f':
                gender = 'female'
             # print(gender)
             cv2.putText(frame, gender, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
             # print(gender)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def imagecap(image_path, haar_path, clf, pca,c,outpath):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pr = Preprocess(haar_path = haar_path)
    # pdb.set_trace()
    gray = pr.normalize_frame(gray)    
    face_to_predict = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
    # pdb.set_trace()
    gender, prob = predict(clf, pca, gray)

    # if prob[0][0] > prob[0][1]:
    #     gender = 'female'
    # if prob[0][0] < prob[0][1]:
    #     gender = 'male'

    if gender[0] == '0' or gender[0] == 'm':
        gender = 'male'
    if gender[0] == '1' or gender[0] == 'f':
        gender = 'female'
    
    cv2.imwrite('%s/cartoon_%s_%d.jpg'%(outpath,gender,c), face_to_predict)
    # cv2.imshow('face',image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.data == 'facescrub':
        with open('FaceScrub_Data.pickle' , 'rb') as handle:
            data = pickle.load(handle, encoding='latin1')
            pairs = reorder_dict(data)
            print('loading facescrub')
            images, truths = pair_to_unit(pairs)
            comp = 150
    if args.data == 'old':
        image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
        pairs = images_and_truths(image_locs)
        images, truths = pair_to_unit(pairs)
        
        pr = Preprocess(haar_path=config["haar_path"])
        images = np.array([pr.normalize(image) for image in images])
        images = images.reshape(100, -1)
        comp =20
    X_train, X_test, y_train, y_test = train_test_split(images, truths, test_size=0.20, random_state=42, stratify=truths)
    X_train, X_test = np.array(X_train), np.array(X_test)
    
    pca, clf = build_SVC(X_train, y_train, X_test, y_test, comp)
    haar_path = config['haar_path'] 
    face_cascade = cv2.CascadeClassifier(haar_path)
    videocap(haar_path, clf, pca)
    # image_path = args.image
    # im = os.listdir(image_path)
    # sh = list(range(len(im)))
    # shuffle(sh)
    # c =0
    # for i in range(len(sh))[:100]:
    #     c+=1
    #     print(os.path.join(image_path,im[sh[i]]))
    #     imagecap(os.path.join(image_path,im[sh[i]]), haar_path, clf, pca,c, args.output)