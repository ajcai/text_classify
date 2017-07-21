# coding=utf-8
'''
Author: Weihao
Email: weihao2016@ia.ac.cn
'''
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix
import numpy as np
import time,os
# MyCorus类加载分词完成的文件
class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/seg').readlines():
            yield line.split()
# 生成词典
Corp = MyCorpus()
dictionary = corpora.Dictionary(Corp)
dictionary.filter_extremes(no_below=5) #去除词典中频率小于5的词
# tf-idf方法生成文本向量
corpus = [dictionary.doc2bow(text) for text in Corp] #基于词典生成一个词袋模型
tfidf = models.TfidfModel(corpus) #使用tf-idf 得到文本的向量模型
corpus_tfidf = tfidf[corpus] #基于这个TF-IDF模型，将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量

for dimensions in [50,100,200,500]:
	begin=time.time()#记录生成LSI特征所需的时间
	# 生成LSI特征向量
	lsi = models.LsiModel(corpus = corpus_tfidf,id2word = dictionary,num_topics=dimensions)
	corus_LSI = lsi[corpus]
	# 将gensim的文件向量转化成sklearn能够识别的格式，并保存
	data,rows,cols = [],[],[]
	line_count = 0
	for line in corus_LSI: 
		for elem in line:
			rows.append(line_count)
			cols.append(elem[0])
			data.append(elem[1])
		line_count += 1
	lsi_sparse_matrix = csr_matrix((data,(rows,cols))) # 稀疏向量
	lsi_matrix = lsi_sparse_matrix.toarray()  # 密集向量
	data_name=('train_LSI_vector_'+str(dimensions)+'.npy')
	np.save(data_name,lsi_matrix)
	#保存LDA文本向量
