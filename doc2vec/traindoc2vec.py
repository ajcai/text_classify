# -*- coding:utf-8 -*-
import numpy as np
import gensim
import random
import multiprocessing
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
def load_data():
    seg_name="../data/seg"
    label_name="../data/label"
    docs=open(seg_name).readlines()
    tags=open(label_name).readlines()
    return docs,tags

#Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识.
#我们使用Gensim自带的LabeledSentence方法. 标识的格式为"C0000XX_i"，其中,C0000XX为类别，i为序号
def tag_docs(docs,tags):
    labelized=[]
    for i,s in enumerate(docs):
        tag=tags[i].strip()
        v=s.strip().split()
        labelized.append(LabeledSentence(v, [tag]))
    return labelized

##对数据进行训练
def train(train_doc,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=multiprocessing.cpu_count())
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=multiprocessing.cpu_count())

    #使用所有的数据建立词典
    model_dm.build_vocab(train_doc)
    model_dbow.build_vocab(train_doc)

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    for epoch in range(epoch_num):
        random.shuffle(train_doc)  
        model_dm.train(train_doc,\
                        total_examples=model_dm.corpus_count,\
                        epochs=model_dm.iter)
        model_dbow.train(train_doc,\
                total_examples=model_dbow.corpus_count,\
                epochs=model_dbow.iter)

    return model_dm,model_dbow
if __name__ == "__main__":
    #设置向量维度和训练次数
    size,epoch_num = 400,10
    docs,tags=load_data()
    train_doc=tag_docs(docs,tags)
    #对数据进行训练，获得模型
    model_dm,model_dbow = train(train_doc,size,epoch_num)
    model_dm.save('../data/model/model_doc_dm.dat')
    model_dbow.save('../data/model/model_doc_dbow.dat')

