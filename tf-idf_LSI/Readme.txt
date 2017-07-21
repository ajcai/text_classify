文件功能：
    preproc.py：对源文件进行预处理，包括分词，停用词清洗，将处理后文本放到seg文件中
    LSI.py：读取预处理后的seg文件，利用tf-idf算法向量化处理，输出LSI降维后的文本向量（并保存到txt文件中），分别指定了50，100，200，500维
    LSI_classification.py：读取生成的LSI文本向量后利用多种算法进行分类，输出各个分类器分类所需时间和精确度（tf-idf_LSI_classification_record.txt）
    LSI_main.py：执行上述文件，完成基于tf-idf算法的文本分类过程

运行：
    python LSI_main.py
运行结果：文本分类结果会打印到窗口同时保存在tf-idf_LSI_classification_record.txt文件中

提示：
    当运行sklearn 中的GB分类时，会出现RuntimeWarning: overflow encountered in double_scalars,但该警告不会影响分类器的分类结果
