from data_loader import *
import sys

def test(path):
	pairwise = images_and_truths(path)
	return(pairwise)


if __name__ == "__main__":
	path = sys.argv[1]
	pairs = test(path)
