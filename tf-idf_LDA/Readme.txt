环境：
    Linux，python2.7
依赖包：
    gensim，sci-kitlearn工具包，numpy
文件功能：
    preproc.py: 对源文件进行预处理，包括分词，停用词清洗，将处理后文本放到seg文件中
    feature_extrack_LDA.py：读取预处理后的seg文件，利用tf-idf算法向量化处理，输出LDA降维后的文本向量（并保存到txt文件中），分别指定了50，100，200，500，600维
    LDA_classification.py：读取生成的LDA文本向量后利用多种分类器进行分类，输出各个分类器分类所需时间和精确度，并记录在文件中（tf-idf_LDA_classification_record.txt）
    LDA_main.py：依次执行上述文件，完成基于tf-idf算法的文本分类过程

运行：
    python LDA_main.py
