此程序采用DOC2VEC的中文文本分类实验。数据为搜狗实验室内容分类数据集，这是搜狗实验室在2006年公布的一批文本分类的数据集。其中包含汽车、财经、IT、健康、体育、旅游、教育、招聘、文化、军事等九个领域。
程序组成：
    preproc.py: 预处理程序。去除文本中的特殊符号和停用词，将文本分词。
    traindoc2vec.py: 训练doc2vec模型。
    doc2vec_classification.py: 采用多个分类器对模型进行交叉验证。
    doc2vec_main.py: 主程序，一次执行上面三个程序。
文件夹：
    data/： 数据文件夹
    data/sougou/： 搜狗原始数据文件夹，里面包含9类文本数据。
    data/model/：保存训练好的模型的文件夹。
    data/stopwords.dat：停用词文件
    data/seg：预处理完的文本，一个文本一行保存在这个文件中。
    data/label：原始文本的标记，与seg文件每行对应。
环境：
    系统：Linux ubuntu
    语言：python2.7
    依赖包：gensim，numpy，sklearn
运行：
    cd $text_classify/doc2vec
    python doc2vec_main.py
