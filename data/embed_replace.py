#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
import os
from gensim import matutils
from itertools import islice
import numpy as np


class EmbedReplace():
    def __init__(self, sample_path, wv_path):
        self.samples = read_samples(sample_path)
        self.titles = [sample.split('<sep>')[0].split() for sample in self.samples]
        self.refs = [sample.split('<sep>')[1].split() for sample in self.samples]
        self.wv = KeyedVectors.load_word2vec_format(
            wv_path,
            binary=False)

        if os.path.exists('saved/tfidf.model'):
            self.tfidf_model = TfidfModel.load('saved/tfidf.model')
            self.dct = Dictionary.load('saved/tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
        else:
            self.dct = Dictionary(self.refs)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('saved/tfidf.dict')
            self.tfidf_model.save('saved/tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        '''
        docs :: iterable of iterable of (int, number)
        '''
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):

        """find high TFIDF socore keywords

        Args:
            dct (Dictionary): gensim.corpora Dictionary  a reference Dictionary
            tfidf (list of tfidf):  model[doc]  [(int, number)]
            threshold (float) : high TFIDF socore must be greater than the threshold
            topk(int): num of highest TFIDF socore 
        Returns:
            (list): A list of keywords
        """

        ###########################################
        #          TODO: module 1 task 1          #
        ###########################################
        tfidf.sort(key=lambda x: x[1], reverse=True)
        tfidf = tfidf[:topk]
        tfidf = [item for item in tfidf if item > threshold]
        return tfidf



    def replace(self, token_list, doc):
        """replace token by another token which is similar in wordvector 

        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """
        
        ###########################################
        #          TODO: module 1 task 2          #
        ###########################################
        token_bow = doc.doc2bow(token_list)
        token_tfidf = self.tfidf_model[token_bow]
        keyword_k = self.extract_keywords(doc, token_tfidf)
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()
        while num == int(len(token_list) * 0.3):
            indexes = np.random.choice(len(token_list), num)
            for index in indexes:
                token = token_list[index]
                if isChinese(token) and token not in keyword_k and token in self.mv:
                    new_tokens[index] = self.wv.most_similar(positive=token, topn=1)[0][0]
            num -= 1
        return ' '.join(new_tokens)


    def generate_samples(self, write_path):
        """generate new samples file
        Args:
            write_path (str):  new samples file path

        """
        ###########################################
        #          TODO: module 1 task 3          #
        ###########################################
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            count += 1
            if count % 100 == 0:
                write_samples(replaced, write_path, 'a')
                replaced = []
            replaced.append(sample.split('<sep>')[0] + ' <sep> ' + self.replace(token_list, doc))

sample_path = 'output/train.txt'
wv_path = 'word_vectors/merge_sgns_bigram_char300.txt'
replacer = EmbedReplace(sample_path, wv_path)
replacer.generate_samples('output/replaced.txt')
