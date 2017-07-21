#coding=utf-8
import os
import re
import jieba
import glob
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化                  
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring

def zhprint(obj):
    print re.sub(r"\\u([a-f0-9]{4})", lambda mg: unichr(int(mg.group(1), 16)), obj.__repr__())

def segment(in_file_name,out_file_name,log_file_name):  # 对原始数据集进行分词处理，标签和分词结果分别存在label、seg文件中
    stop_word_file=open('../data/stopwords.dat') #open stop word file
    stop_word_list=[]
    for wd in stop_word_file:
        stop_word_list.append(wd.strip().decode('utf-8')) #停用词表
    log_file=open(log_file_name,'a')
    punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．﹒：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻●☆★%/~
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')  # 中文标点符号
    num_pattern=re.compile(r'\d+') # 数字的正则表达式
    # 对str/unicode
    filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
    with open(out_file_name, 'a') as out_file,\
        open(in_file_name,'r') as in_file:
        for line in in_file:
            ms=''
            try:
                ms=strQ2B(line.decode('gbk'))
                ms=ms.replace('&nbsp','').strip()
                ms = filterpunt(ms)
                ms = num_pattern.sub('num',ms) #将数字全部替换成 'num'
            except:
                log_file.write('%s :%s\n'%(in_file_name,line)) #错误日志
            #print ms
            #zhprint(list(jieba.cut(ms)))
            words_list=[]
            for w in jieba.cut(ms):
                if w not in stop_word_list:
                    words_list.append(w)
            words = ' '.join(map(lambda _:_.strip(), words_list))
            out_file.write('%s ' % (words.encode('utf-8')))
        out_file.write('\n')
    #return words.encode('utf-8')
def prepare_segmented_data():
    log_file_name='error_preprocess.log'
    seg_file_name="../data/seg"
    label_file_name="../data/label"
    if os.path.exists(seg_file_name) and os.path.exists(label_file_name):
        return
    with open(log_file_name,'w') as fw1,\
        open(seg_file_name,'w') as fw2,\
        open(label_file_name,'w') as fw3:
        fw1.truncate()
        fw2.truncate()
        fw3.truncate()
    
    for class_dir in glob.glob(r'../data/sougou/C*'):
        class_name=class_dir.split('/')[-1]
        for in_file_name in glob.glob("%s/*.txt"%class_dir):
            doc_name=in_file_name.split('/')[-1].split('.')[0]
            label="%s_%s"%(class_name,doc_name)
            segment(in_file_name,seg_file_name,log_file_name)
            with open(label_file_name,'a') as fw:
                fw.write("%s\n"%label)

if __name__ =='__main__':
    prepare_segmented_data()
