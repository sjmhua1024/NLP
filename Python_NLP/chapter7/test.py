#coding=utf-8
import gensim
def my_function():

    model = gensim.models.Word2Vec.load('./data/zhiwiki_news.word2vec')
    print(model.wv.similarity('大数据','人工智能'))  #相似度为0.63
    print(model.wv.similarity('滴滴','共享单车'))  #相似度为0.44

    word = '起诉书'
    if word in model.wv.index2word:
        print(model.wv.most_similar(word))


if __name__ == '__main__':
    my_function()
