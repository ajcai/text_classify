from __future__ import print_function
import os
print("preparing pre-processed data...")
os.system('python preproc.py')
print("training doc2vec model...")
os.system('python traindoc2vec.py')
print("doing cross validation test")
os.system('python doc2vec_classification.py')
