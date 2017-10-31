#!/usr/env/python
import cv2
import os

count = 0

facedata = "/home/ashutosh/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

if (os.path.isdir("faces") == True):
    print("The cropped faces directory is present")
else:
    print("Creating the cropped faces directory")
    folder_path = "faces_dir"
    os.makedirs("faces")
    for data_file in sorted(os.listdir(folder_path)):
        img = cv2.imread(folder_path + "/" + data_file)
	minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)
        faces = cascade.detectMultiScale(miniframe,1.3,5)
	for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0),2)
            sub_face = img[y:y+h, x:x+w]
            face_file_name = "faces/" + str(data_file)
            sub_face = cv.normalize(sub_face, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(face_file_name, sub_face) 
             


new_folder_path = "faces"
os.makedirs("hist_eq_faces")
for new_file in sorted(os.listdir(new_folder_path)):
    img1 = cv2.imread(new_folder_path + "/" + str(new_file))
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    print(img2.shape)
    gray = cv2.equalizeHist(img2)
    hist_file_name = "hist_eq_faces/" + str(new_file)
    cv2.imwrite(hist_file_name,gray)
    

