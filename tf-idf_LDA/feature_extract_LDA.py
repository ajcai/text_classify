# coding=utf-8
from __future__ import print_function
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix
import numpy as np
import time,os

class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/seg').readlines():
            yield line.split()

# gensim to tf-idf
Corp = MyCorpus()

dictionary = corpora.Dictionary(Corp)
dictionary.filter_extremes(no_below=5)

corpus = [dictionary.doc2bow(text) for text in Corp] #bag of words

tfidf = models.TfidfModel(corpus) #使用tf-idf 模型得出该评论集的tf-idf 模型
corpus_tfidf = tfidf[corpus] 


#extrack_time=open('feature_extrack_LDA.txt','a')
#extrack_time.truncate()

for dimensions in [50,100,200,500,600]:
	print('extracting features with %s dimensions...'%dimensions)
	begin=time.time()
	lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=dimensions)
	corpus_LDA=lda[corpus]
	data,rows,cols = [],[],[]
	line_count = 0
	for line in corpus_LDA:  # gensim lda vec
		for elem in line:
			rows.append(line_count)
			cols.append(elem[0])
			data.append(elem[1])
		line_count += 1
	#print line_count
	lda_sparse_matrix = csr_matrix((data,(rows,cols))) # sparse vec
	lda_matrix = lda_sparse_matrix.toarray()  # dense vec
	#print lda_matrix.shape
	print('features with %s dimensions finished'%dimensions)
	print('used %s seconds'%(time.time()-begin))
	#extrack_time.write('\n')
	data_name='train_LDA_vector_%s.npy'%dimensions
	np.save(data_name,lda_matrix)
