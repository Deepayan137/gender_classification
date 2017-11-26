import json

# data = {
# 		"dir":"/OCRData/minesh.mathew/cnn_rnn/crnn_minesh/tool/mohit_data/urdu_ocr/rawFEatures/",
# 		"Files":["urduTrain7kFeat.txt",
# 				"urduTest16kFeat_1_2.txt",
# 				"urduTest16kFeat_1_1.txt",
# 				"urduTest16kFeat_2.txt"
# 		]	
# 		}

data = {
	"haar_path": "/home/deepayan/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
	"dir": "/home/deepayan/git/gender_classification/faces_dir"
	"data":{
	
	}
}
with open('faces.json','w') as outfile:
	json.dump(data, outfile, indent=4)