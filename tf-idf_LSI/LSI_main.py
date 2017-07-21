# coding=utf-8
from __future__ import print_function
import os
print("preparing pre-processed data...")
os.system('python preproc.py')
print("training tf-idf LSI model...")
os.system('python LSI_vector.py')
print("doing cross validation test...")
os.system('python LSI_classification.py')