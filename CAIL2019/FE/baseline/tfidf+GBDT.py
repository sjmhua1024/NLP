#!/usr/bin/env python
# coding: utf-8
import json

import jieba
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.ensemble import GradientBoostingClassifier

dim = 5000


# 函数功能：对列表进行分词
# 返回值：分词后的列表
def cut_text(alltext):
    count = 0
    cut = jieba
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(' '.join(cut.cut(text)))
    return train_text


# 函数：计算TF-IDF
# 返回值：训练好的TF-IDF类
def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,		# 至少在5个文本中出现过
        max_features=dim,	# 选取前5000个词频最高的词
        ngram_range=(1, 3),	# 选取1-3元语言模型进行计算
        use_idf=1,		# 使用idf
        smooth_idf=1		# 增加平滑
    )
    tfidf.fit(train_data)	

    return tfidf


# 函数功能：读取训练集
# 返回值：
## 列表alltext: 句子
## 列表tag_label: 标签
def read_trainData(path, tag_path):
    fin = open(path, 'r', encoding='utf8')
    tag_dic, tagname_dic = init(tag_path)

    alltext = []
    tag_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        for sent in d:
            alltext.append(sent['sentence'])
            tag_label.append(getlabel(sent, tag_dic))
        line = fin.readline()
    fin.close()

    return alltext, tag_label


# 函数功能：训练SVM分类器
# 返回值：SVM分类器
def train_GBDT(vec, label):
    GBDT = GradientBoostingClassifier()
    GBDT.fit(vec, label)
    return GBDT


# 函数功能：构建标签字典
# 返回值：
## 字典tag_dic: key:标签名称; name:序号
## 字典tagname_dic: key:序号; name:标签名称
def init(tags_path):
    f = open(tags_path, 'r', encoding='utf8')
    tag_dic = {}
    tagname_dic = {}
    line = f.readline()
    while line:
        tagname_dic[len(tag_dic)] = line.strip()
        tag_dic[line.strip()] = len(tag_dic)
        line = f.readline()
    f.close()
    return tag_dic, tagname_dic


# 函数功能：做单标签（返回多个类的第一个）
# 返回值：如果有标签则返回标签对应的序号；如果没有标签则返回空串
def getlabel(d, tag_dic):
    if len(d['labels']) > 0:
        print(d['labels'])
        return tag_dic[d['labels'][0]]
    return ''


# 函数功能：训练对应的模型
# 参数：　type: 案件类型，分为labor,divorce,loan
def train(type):
    print('train_' + type + '_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/'+type+'/data_small_selected.json', '../data/'+type+'/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)

    print('tag GBDT')
    tag = train_GBDT(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_'+type+'//tfidf.model')
    joblib.dump(tag, 'predictor/model_'+type+'/tag_gbdt.model')


# 主函数
if __name__ == '__main__':
    # 训练劳动争议模型
    train('labor')

    # 训练离婚模型
    train('divorce')

    # 训练借款纠纷模型
    train('loan')
