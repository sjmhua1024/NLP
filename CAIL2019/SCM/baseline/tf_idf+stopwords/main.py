import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 定义停用词列表
stop_words_list = []
sf = open('stop_words.txt','r',encoding='utf8')
for line in sf:
    stop_words_list.append(line)


# 分词
def trans(x):
    x = list(jieba.cut(x))
    for word in x:
        word.replace('\n', '')
    return " ".join(x)


# 本地环境读入输入数据
se = set()
f = open("input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])
f.close()

# 实际上传环境读入输入数据
f = open("/input/input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])
f.close()

# 将输入数据转换为分词后的字符串
data = list(se)
for a in range(0, len(data)):
    data[a] = trans(data[a])

# 构建TF-IDF模型，得到输入数据的编码（稀疏矩阵形式）
tfidf_model = TfidfVectorizer(stop_words=stop_words_list)
tfidf_model.fit(data)


# 实际上传环境读入输入数据
f = open("/input/input.txt", "r", encoding="utf8")

# 实际上传环境写入输出数据
ouf = open("/output/output.txt", "w", encoding="utf8")

for line in f:
    x = json.loads(line)
    y = [
        trans(x["A"]),
        trans(x["B"]),
        trans(x["C"])
    ]

    y = tfidf_model.transform(y)
    y = y.todense()

    v1 = np.sum(np.dot(y[0], np.transpose(y[1])))
    v2 = np.sum(np.dot(y[0], np.transpose(y[2])))
    if v1 > v2:
        print("B", file=ouf)
    else:
        print("C", file=ouf)

f.close()
ouf.close()
