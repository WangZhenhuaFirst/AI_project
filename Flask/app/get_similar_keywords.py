# -*- coding: utf-8 -*-

import jieba
import numpy as np
import pandas as pd
import time
from functools import wraps
import gc
import jieba.analyse
from collections import Counter
from flask import Flask, jsonify
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import json
from collections import OrderedDict
from annoy import AnnoyIndex
from app.models.config import Config
import os


class Keywords(object):
    '''基于textrank提取关键字'''

    @staticmethod
    def init():
        pass

    @staticmethod
    def clean():
        pass

    @staticmethod
    def get_keywords(sentence, topn=20, weight=0.5):
        ''' 基于textrank算法提取关键字'''
        # 词性限制集合为["ns", "n", "vn", "v", "nr"]，表示只能从词性为地名、名词、动名词、动词、人名这些词性的词中抽取关键词。
        allow_pos = ("ns", "n", "vn", "v", "nr")
        keywords = jieba.analyse.textrank(
            sentence, topK=topn, withWeight=True, allowPOS=allow_pos)
        # print(keywords)
        keywords_item = {}
        for item in keywords:
            # 取权重大于定义的weight
            if item[1] > weight:
                keywords_item[item[0]] = item[1]

        return keywords_item


class Similarwords(object):
    '''基于word2vec提取相似词'''
    model = None
    # 加载模型文件

    @staticmethod
    def init():
        if Similarwords.model is None:
            app = Flask(__name__)
            dict_path = os.path.join(app.static_folder, Config.bigram_char)
            Similarwords.model = KeyedVectors.load_word2vec_format(
                dict_path, binary=False, encoding="utf8",  unicode_errors='ignore')

    @staticmethod
    def clean():
        # 释放
        if Similarwords.model is not None:
            del Similarwords.model
            gc.collect()
            Similarwords.model = None
        pass

    @staticmethod
    def get_similar_words(text, topn=20, weight=0.5):
        ''' 基于textrank算法提取关键字'''
        sim_words = Similarwords.model.most_similar(text, topn=topn)
        # print(keywords)
        sim_words_item = {}
        for item in sim_words:
            # 取权重大于定义的weight
            if item[1] > weight:
                sim_words_item[item[0]] = item[1]

        return sim_words_item


class Annoysimilarwords():
    '''基于annoy提取相似词'''
    model = None
    word_index = {}
    reverse_word_index = {}
    # 加载模型文件

    @staticmethod
    def init():
        app = Flask(__name__)

        if len(Annoysimilarwords.word_index) == 0:
            dict_path = os.path.join(app.static_folder, Config.word_index)
            with open(dict_path, 'rb') as fo:     # 读取pkl文件
                Annoysimilarwords.word_index = pickle.load(
                    fo, encoding='bytes')

        if len(Annoysimilarwords.reverse_word_index) == 0:
            dict_path = os.path.join(
                app.static_folder, Config.reverse_word_index)
            with open(dict_path, 'rb') as fo:     # 读取pkl文件
                Annoysimilarwords.reverse_word_index = pickle.load(
                    fo, encoding='bytes')

        if Annoysimilarwords.model is None:
            dict_path = os.path.join(app.static_folder, Config.index_build200)
            Annoysimilarwords.model = AnnoyIndex(300, metric='angular')
            Annoysimilarwords.model.load(dict_path)

    @staticmethod
    def clean():
        # 释放
        if Annoysimilarwords.model is not None:
            del Annoysimilarwords.model
            del Annoysimilarwords.word_index
            del Annoysimilarwords.reverse_word_index
            gc.collect()
            Annoysimilarwords.model = None
            Annoysimilarwords.word_index = []
            Annoysimilarwords.reverse_word_index = []
        pass

    @staticmethod
    def get_similar_words(text, topn=20, weight=0.5):
        ''' 基于annoy算法提取相似词'''
        sim_words = Annoysimilarwords.model.get_nns_by_item(
            Annoysimilarwords.word_index[text], topn, include_distances=True)
        # print(keywords)
        sim_words_item = {}
        for item, j in zip(sim_words[0], sim_words[1]):
            # 余弦相似度归一化的计算公式，
            weights = 0.5*(abs(1-j))+0.5
            # 取权重大于定义的weight
            if weights > weight:
                sim_words_item[Annoysimilarwords.reverse_word_index[item]] = weights

        return sim_words_item
