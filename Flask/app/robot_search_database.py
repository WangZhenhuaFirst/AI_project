# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import json
from collections import OrderedDict
from annoy import AnnoyIndex
import time
import json
import datetime
import numpy as np
import pickle
import os
from app.robot_get_embedding import build_tokenizer, build_model, extract_emb_feature
from flask import Flask


class SearchDataBase:
    # 定义历史对话记录history全局变量
    global history

    def __init__(self):
        app = Flask(__name__)

        # 最大历史记录长度
        self.dict_name = r"model04/search_database/sentence_emb_dict.pkl"
        self.ann_save_path = r"model04/search_database/annoy.index"
        self.id2word_path = r"model04/search_database/id2word.pkl"
        self.word2id_path = r"model04/search_database/word2id.pkl"
        self.answer_path = r'model04/search_database/finance_data_answer_dict.pkl'
        self.id2word = pickle.load(
            open(os.path.join(app.static_folder, self.id2word_path), "rb"))
        self.word2id = pickle.load(
            open(os.path.join(app.static_folder, self.word2id_path), "rb"))
        self.annoy_index = AnnoyIndex(768, metric='angular')
        self.annoy_index.load(os.path.join(
            app.static_folder, self.ann_save_path))
        self.answer = pickle.load(
            open(os.path.join(app.static_folder, self.answer_path), "rb"))
        # 设置bert文件路径
        self.dict_path = r"model04/search_database/chinese_L-12_H-768_A-12/vocab.txt"
        self.config_path = r"model04/search_database/chinese_L-12_H-768_A-12/bert_config.json"
        self.checkpoint_path = r"model04/search_database/chinese_L-12_H-768_A-12/bert_model.ckpt"
        # 加载tokenizer
        self.tokenizer = build_tokenizer(
            os.path.join(app.static_folder, self.dict_path))
        # 加载bert模型
        self.model = build_model(os.path.join(app.static_folder, self.config_path), os.path.join(
            app.static_folder, self.checkpoint_path))
        # 定义阈值
        self.threshold = 0.96

    # 构建索引，只需构建一次
    def build_index(self, tree_num=500):
        # 自定义的读取word2vec的函数
        items_vec = pickle.load(open(self.dict_name, "rb"))
        # 向量维度为768
        a = AnnoyIndex(768, metric='angular')
        i = 0
        id2word, word2id = dict(), dict()
        for word in items_vec.keys():
            a.add_item(i, items_vec[word])
            id2word[i] = word
            word2id[word] = i
            i += 1
        a.build(tree_num)
        # r如果路径中已有保存的文件会报错,OSError: Unable to open: Invalid argument (22)OSError: Unable to open: Invalid argument (22)
        a.save(self.ann_save_path)
        pickle.dump(id2word, open(self.id2word_path, "wb"))
        pickle.dump(word2id, open(self.word2id_path, "wb"))

    # 近似检索，query为用户输入的问题
    def annoy_search(self, query, topn=10, print_value=False):
        # global id2word, word2id, annoy_index
        query_vec = extract_emb_feature(self.model, self.tokenizer, [
                                        query], max_len=30)[0].reshape(-1, 1)
        idxes, dists = self.annoy_index.get_nns_by_vector(
            query_vec, topn, include_distances=True)
        idxes_weight_item = {}
        for i, j, n in zip(idxes, dists, range(len(idxes))):
            # 余弦相似度归一化的计算公式，
            weights = 0.5*(abs(1-j))+0.5
            # print('weights: ', weights)
            if weights > self.threshold:
                idxes_weight_item[idxes[n]] = weights
        if print_value:
            print(idxes_weight_item)
        if idxes_weight_item != {}:
            max_idxes = max(zip(idxes_weight_item.values(),
                                idxes_weight_item.keys()))[1]
            return self.answer[max_idxes]
        else:
            return {}
