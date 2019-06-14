import json
import jieba
import numpy as np
from gensim.models import Doc2Vec, doc2vec
import codecs


# 函数： 训练doc2vec模型
# 返回值：训练好的doc2vec模型
def train_doc2vec(train_data):
    # 参数待调整
    model = Doc2Vec(
        train_data, 
        dm=0, 
        dbow_words=1, 
        size=300, 
        window=8, 
        min_count=5, 
        iter=5)

    return model


# 定义停用词列表
stop_words_list = []
sf = open('stop_words.txt','r',encoding='utf8')
for line in sf:
    stop_words_list.append(line)


# 分词
def trans(x):
    x = list(jieba.cut(x))
    lis = []
    for word in x:
        word.replace('\n', '')
        if word not in stop_words_list:
            lis.append(word)
    return lis

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
'''
f = open("/input/input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])
'''

# 将输入数据转换为分词后的字符串
data = list(se)
for a in range(0, len(data)):
    data[a] = trans(data[a])

# 构建doc2vec的训练集
train = []
for i , v in enumerate(data):
    train.append(doc2vec.TaggedDocument(v,[i]))

model = train_doc2vec(train)

model.save('test.doc2vec')

# 实际上传环境读入输入数据
# f = open("/input/input.txt", "r", encoding="utf8")

# 本地环境读入输入数据
f = open("input.txt", "r", encoding="utf8")

# 本地测试环境写入输出数据
ouf = open('output.txt','w',encoding='utf8')

# 实际上传环境写入输出数据
# ouf = open("/output/output.txt", "w", encoding="utf8")

# 定义实际结果输出文件
true_ouf = open('true_output.txt', "w", encoding="utf8")

# 定义训练参数
start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192


def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity

for line in f:
    x = json.loads(line)
    y = [
        trans(x["A"]),
        trans(x["B"]),
        trans(x["C"])
    ]

    y0_vec = model.infer_vector(y[0], alpha=start_alpha, steps=infer_epoch)
    y1_vec = model.infer_vector(y[1], alpha=start_alpha, steps=infer_epoch)
    y2_vec = model.infer_vector(y[2], alpha=start_alpha, steps=infer_epoch)

    v1 = simlarityCalu(y0_vec,y1_vec)
    v2 = simlarityCalu(y0_vec,y2_vec)
    if v1 > v2:
        print("B", file=ouf)
    else:
        print("C", file=ouf)
    print('B',file=true_ouf)

true_ouf.close()
f.close()
ouf.close()
