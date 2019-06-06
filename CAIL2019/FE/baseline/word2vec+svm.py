#!/usr/bin/env python
# coding: utf-8
import json

import jieba
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from gensim.models import FastText

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


# 函数： 训练fasttext词向量
# 返回值：训练好的fasttext词向量模型
def train_fasttext(train_data):
    # 参数待调整
    model = FastText(
        train_data,
        size=300,		    # 词向量维度
        window=8,	        # 窗口大小
        min_count=5,	    # 最小词频数量
        min_n=3,		    # 最小的n元语言模型
        max_n=6,		    # 最大的n元语言模型
        word_ngrams=1       # 
    )

    return model


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
def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


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
    print('train fasttext...')
    fasttext_model = train_fasttext(train_data)

    vec = fasttext_model.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(fasttext_model, 'predictor/model_'+type+'/fasttext.model')
    joblib.dump(tag, 'predictor/model_'+type+'/tag_svc.model')

# 主函数
if __name__ == '__main__':
    # 训练劳动争议模型


    # 训练离婚模型
    print('train_divorce_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/divorce/data_small_selected.json', '../data/divorce/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train fasttext...')
    fasttext_model = train_fasttext(train_data)

    vec = tfidf.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_divorce/tfidf.model')
    joblib.dump(tag, 'predictor/model_divorce/tag.model')

    # 训练借款纠纷模型
    print('train_loan_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/loan/data_small_selected.json', '../data/loan/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    fasttext_model = train_fasttext(train_data)

    vec = tfidf.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_loan/tfidf.model')
    joblib.dump(tag, 'predictor/model_loan/tag.model')
    
