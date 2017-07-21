# coding=utf-8
from __future__ import print_function
import os
print("preparing pre-processed data...")
os.system('python preproc.py')
print("extracting tf-idf LDA features...")
os.system('python feature_extract_LDA.py')
print("doing cross validation test...")
os.system('python LDA_classification.py')
