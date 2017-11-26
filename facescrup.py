import os
import sys
import pickle
import pdb
if __name__ == '__main__':
	with open('FaceScrub_Data.pickle', 'rb') as f:
		x = pickle.load(f)
	pdb.set_trace()