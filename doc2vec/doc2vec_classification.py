# coding=utf-8
from __future__ import print_function
import numpy as np
import time,types,os
from sklearn.model_selection import cross_val_score
import gensim
from gensim.models.doc2vec import Doc2Vec
# meta-estimator
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

##读取向量
def getVecs(model, tags):
    vecs = [np.array(model.docvecs[z]).reshape(1,-1) for z in tags]
    return np.concatenate(vecs)

classifiers = {
    #'KN': KNeighborsClassifier(3),
    'SVM':svm.LinearSVC(C=1),
    #'RBFSVM':svm.SVC(),
    #'POLYSVM':svm.SVC(kernel='poly', degree=3),
    'DT': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(n_estimators=10, max_depth=5, max_features=1),  # clf.feature_importances_
    'ET': ExtraTreesClassifier(n_estimators=10, max_depth=None),  # clf.feature_importances_
    #'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), # clf.feature_importances_
    'NB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QD': QuadraticDiscriminantAnalysis(),
}

if __name__=="__main__":
    # load data    
    label=np.zeros(17910) #creat label
    base0=np.zeros(1990)
    base1=np.ones(1990)
    label[:1990]=base0
    label[1990:3980]=base1
    label[3980:5970]=2*base1
    label[5970:7960]=3*base1
    label[7960:9950]=4*base1
    label[9950:11940]=5*base1
    label[11940:13930]=6*base1
    label[13930:15920]=7*base1
    label[15920:]=8*base1   
    
    model_dm=Doc2Vec.load('../data/model/model_doc_dm.dat')
    model_dbow=Doc2Vec.load('../data/model/model_doc_dbow.dat')
    tags=open('../data/label').readlines()
    tags=[l.strip() for l in tags]
    vecs_dm=getVecs(model_dm,tags)
    vecs_dbow=getVecs(model_dbow,tags)
    features=np.hstack((vecs_dm,vecs_dbow))
    print("vecs_dm")
    for name, clf in classifiers.items():
        print('test with %s...'%name)
        scores = cross_val_score(clf, vecs_dm,label)
        print(name,'\t--> ',scores.mean())
    print("vecs_dm+vecs_dbow")
    for name, clf in classifiers.items():
        print('test with %s...'%name)
        scores = cross_val_score(clf, features,label,cv=5)
        print(name,'\t--> ',scores.mean())
    print("vecs_dbow")
    for name, clf in classifiers.items():
        print('test with %s...'%name)
        scores = cross_val_score(clf, vecs_dbow,label,cv=5)
        print(name,'\t--> ',scores.mean())

