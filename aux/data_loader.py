import os
import sys
import cv2
import json
import pdb
from sklearn.model_selection import train_test_split
def read_image(path):
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
def get_label(image_name):
		f = lambda x: 1 if 'f' in x else 0
		return(f(image_name))
def images_and_truths(image_locs):
	results =[]
	for loc in image_locs:
		image_name = loc.split('/')[-1]
		label = get_label(image_name)
		image = read_image(loc)
		results.append((image, label))
	return results
def pair_to_unit(pairwise):
    images, truths = [], []

    for im, tr in pairwise:
    	
    	images.append(im)
    	truths.extend(str(tr))
    #units = tuple(zip(images, truths)) 
    return (images,truths)


# if __name__ == "__main__":
# 	config = json.load(open(sys.argv[1]))
# 	image_locs = list(map(lambda x: config["dir"] + x  , os.listdir(config["dir"])))
# 	pairs = images_and_truths(image_locs)
# 	im, tr = pair_to_unit(pairs)
# 	pdb.set_trace()
# 	X_train, X_test, y_train, y_test = train_test_split(im,tr, test_size=0.20, random_state=42)
	


	
	
