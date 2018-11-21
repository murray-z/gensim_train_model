# coding: utf-8

import codecs
import jieba
from gensim.models import LdaModel, LsiModel, TfidfModel
from gensim.corpora import Dictionary
from gensim import corpora
import os
import argparse


def train(corpus_path, stopwords_path='./stop_words.txt', model='lda', model_path='./model', num_topics=10):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]

    train = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            train.append([word for word in jieba.lcut(line.strip()) if word not in stopwords])

    dictionary = corpora.Dictionary(train)

    corpus = [dictionary.doc2bow(text) for text in train]

    if not os.path.exists(os.path.join(model_path, model)):
        os.mkdir(os.path.join(model_path, model))

    if model == 'lda':
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        lda.save(os.path.join(model_path, 'lda', 'lda.model'))
        dictionary.save(os.path.join(model_path, 'lda', 'dictionary.dic'))

    if model == 'lsi':
        lsi = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        lsi.save(os.path.join(model_path, 'lsi', 'lsi.model'))
        dictionary.save(os.path.join(model_path, 'lsi', 'dictionary.dic'))

    if model == 'tfidf':
        tfidf = TfidfModel(corpus=corpus, id2word=dictionary)
        tfidf.save(os.path.join(model_path, 'tfidf', 'tfidf.model'))
        dictionary.save(os.path.join(model_path, 'tfidf', 'dictionary.dic'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--corpus_path', help='语料路径，一行存放一篇文章')
    args.add_argument('--stopwords_path', default='./stop_words.txt', help='停用词路径')
    args.add_argument('--model', default='lda', help='要训练模型, "lda|lsi|tfidf"')
    args.add_argument('--num_topics', type=int, default=10, help='主题类别数目')

    args = args.parse_args()

    corpus_path = args.corpus_path
    stopwords_path = args.stopwords_path
    model = args.model
    num_topics = args.num_topics

    train(corpus_path=corpus_path, model=model, stopwords_path=stopwords_path, num_topics=num_topics)
