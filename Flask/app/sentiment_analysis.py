# -*- coding: utf-8 -*-

# export AUTOGRAPH_VERBOSITY=10
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import re
import jieba
from collections import Counter
import os
import argparse
import logging
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
import math
from tqdm import tqdm
import warnings
import os
import sys
from app.models.config import Config
from flask import Flask
import gc
import sys


class SentimentAnalysis(object):
    flags = None
    logger = None
    test_comment = []

    @staticmethod
    def Init():
        if SentimentAnalysis.flags is None:
            SentimentAnalysis.flags = SentimentAnalysis.initial_arguments()

        if SentimentAnalysis.logger is None:
            SentimentAnalysis.logger = SentimentAnalysis.initial_logging()

        if len(SentimentAnalysis.test_comment) == 0:
            app = Flask(__name__)
            dict_path = os.path.join(
                app.static_folder, Config.test_comment_file)
            s_csv = pd.read_csv(dict_path)
            SentimentAnalysis.test_comment = s_csv['content'].tolist()

    @staticmethod
    def clean():
        # 释放
        if SentimentAnalysis.flags is not None:
            del SentimentAnalysis.flags
            del SentimentAnalysis.logger
            del SentimentAnalysis.test_comment
            gc.collect()
            SentimentAnalysis.flags = None
            SentimentAnalysis.logger = None
            SentimentAnalysis.test_comment = []
        pass

    # 数据文件，模型路径，训练和验证的初始化参数
    @staticmethod
    def initial_arguments():
        sys.argv = ['']
        app = Flask(__name__)

        parser = argparse.ArgumentParser()

        # data参数
        parser.add_argument('--root_path', type=str,
                            default='', help='the path of main.py')

        dict_path = os.path.join(app.static_folder, Config.train)
        parser.add_argument('--raw_data', type=str,
                            default=dict_path, help='unprocessed data')

        dict_path = os.path.join(app.static_folder, Config.train)
        parser.add_argument('--processed_data', type=str, default='data/processed.csv',
                            help='data after segment and tokenize')

        dict_path = os.path.join(app.static_folder, Config.train)
        parser.add_argument('--train_data', type=str,
                            default='data/train_data.csv', help='path of training data file')

        parser.add_argument('--num_train_sample', type=int,
                            default=100000, help='num of train sample')

        dict_path = os.path.join(app.static_folder, Config.train)
        parser.add_argument('--valid_data', type=str,
                            default='data/valid_data.csv', help='path of validating data file')

        parser.add_argument('--num_valid_sample', type=int,
                            default=5000, help='num of valid sample')

        label_file = os.path.join(app.static_folder, Config.label_file)
        parser.add_argument('--label_file', type=str,
                            default=label_file, help='path of label name')

        dict_path = os.path.join(app.static_folder, Config.stopwords_file)
        parser.add_argument('--stopwords_file', type=str,
                            default=dict_path, help='path of stopwords file')

        dict_path = os.path.join(app.static_folder, Config.vocab_file)
        parser.add_argument('--vocab_file', type=str,
                            default=dict_path, help='path of vocabulary file')

        dict_path = os.path.join(app.static_folder, Config.test_comment_file)
        parser.add_argument('--test_comment_file', type=str,
                            default=dict_path, help='comments used for testing')

        # 模型路径
        dict_path = os.path.join(app.static_folder, Config.best_weight)
        parser.add_argument('--weight_save_path', type=str,
                            default=dict_path, help='path of best weights')

        # 模型参数
        parser.add_argument('--max_len', type=int,
                            default=1000, help='max length of content')
        parser.add_argument('--vocab_size', type=int,
                            default=50000, help='size of vocabulary')
        parser.add_argument('--embedding_dim', type=int,
                            default=300, help='embedding size')
        parser.add_argument('--lstm_unit', type=int,
                            default=256, help='unit num of lstm')
        parser.add_argument('--dropout_loss_rate', type=float,
                            default=0.2, help='dropout loss ratio for training')
        parser.add_argument('--label_num', type=int,
                            default=4, help='num of label')

        # train and valid
        parser.add_argument('--train_log', type=str,
                            default='model/train_log.txt', help='path of train log')
        parser.add_argument('--batch_size', type=int,
                            default=32, help='batch size')
        parser.add_argument('--shuffle_size', type=int,
                            default=128, help='the shuffle size of dataset')
        parser.add_argument('--feature_num', type=int,
                            default=20, help='num of feature')
        parser.add_argument('--lr', type=float,
                            default=1e-4, help='learning rate')
        parser.add_argument('--ckpt_params_path', type=str, default='model/ckpt/ckpt_params.json',
                            help='path of checkpoint params')

        flags, unparsed = parser.parse_known_args()
        return flags

    # 定义日志函数
    @staticmethod
    def initial_logging(logging_path='info.log'):
        logger = logging.getLogger((__name__))
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(filename=logging_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        return logger

    @staticmethod
    def predict(text=''):
        key_list = ['交通便利', '距离商圈远近', '是否容易寻找', '排队等候时间', '服务人员态度',
                    '是否容易停车', '点菜/上菜速度', '价格水平', '性价比', '折扣力度',
                    '装修情况', '嘈杂情况', '就餐空间', '卫生情况', '分量',
                    '口感', '外观', '推荐程度', '本次消费感受', '再次消费的意愿']
        SentimentAnalysis.logger.info('Initialize model')
        sa = SentimentAnalysisModel(SentimentAnalysis.flags)
        value_list = sa.predict(text)
        results = [(key_list[i], value_list[i])
                   for i in range(len(value_list))]
        return results

    # 获取测试数据
    @staticmethod
    def get_test_comment():
        return np.random.choice(SentimentAnalysis.test_comment)


def get_model(max_len, vocab_size, embedding_dim, lstm_unit, dropout_keep_rate, label_num, show_structure=False):
    '''构建model'''
    inputs = Input((max_len,), name='input')
    embedding = Embedding(vocab_size, embedding_dim, name='embedding')(inputs)
    bilstm1 = Bidirectional(
        LSTM(lstm_unit, return_sequences=True), name='bi-lstm1')(embedding)
    dropout1 = Dropout(dropout_keep_rate)(bilstm1)
    bilstm2 = Bidirectional(
        LSTM(lstm_unit, return_sequences=True), name='bi-lstm2')(dropout1)
    dropout2 = Dropout(dropout_keep_rate)(bilstm2)
    att = Attention(max_len, name='attention')(dropout2)
    d_list = [Dense(name=f'dense{i}', units=label_num, activation='softmax')(
        att) for i in range(20)]

    model = Model(inputs=inputs, outputs=d_list)
    if show_structure:
        model.summary()

    return model

# 读取stopwords


def get_stopwords(file):
    with open(file, 'r', encoding='utf-8') as f:
        stopwords = [s.strip() for s in f.readlines()]
    return stopwords


# 建立word2id和id2word映射
def read_vocab(vocab_file):
    word2id = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line.strip()
            word2id[word] = i
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word

# 用jieba分词对内容进行切割，并用空格连接分割后的词


def segmentData(contents, stopwords):
    def content2words(content, stopwords):
        content = re.sub('~+', '~', content)
        content = re.sub('\.+', '~', content)
        content = re.sub('～+', '～', content)
        content = re.sub('(\n)+', '\n', content)
        return ' '.join([word for word in jieba.cut(content) if word.strip() if word not in stopwords])

    seg_contents = [content2words(c, stopwords) for c in contents]
    return seg_contents


# 将分割后的句子转化为id
def tokenizer(content, w2i, max_token=1000):
    tokens = content.split()
    ids = []
    for t in tokens:
        if t in w2i:
            ids.append(w2i[t])
        else:
            ids.append(w2i['<UNK>'])
    ids = [w2i['<SOS>']] + ids[:max_token-2] + [w2i['<EOS>']]
    ids += (max_token - len(ids)) * [w2i['<EOS>']]
    assert len(ids) == max_token
    return ids

# 预测函数


class SentimentAnalysisModel:
    def __init__(self, flags):
        # 加载模型
        self.model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                               flags.dropout_loss_rate, flags.label_num)
        self.model.load_weights(flags.weight_save_path)
        # 预加载处理评价数据
        self.stopwords = get_stopwords(flags.stopwords_file)
        self.w2i, _ = read_vocab(flags.vocab_file)
        with open(flags.label_file, 'r') as f:
            self.labels = [l.strip() for l in f.readlines()]
        self.classify = ['Not mention', 'Bad', 'Normal', 'Good']

    # string to tokens
    def process_data(self, comment):
        seg_comment = segmentData([comment], self.stopwords)[0]
        tokens = tokenizer(seg_comment, self.w2i)
        return tokens

    # string to labels
    def predict(self, comment):
        tokens = self.process_data(comment)
        pred = self.model.predict(np.array(tokens).reshape((1, len(tokens))))
        categorys = [int(np.argmax(p)) for p in pred]
        return categorys

    # 打印结果
    def print_result(self, comment):
        categorys = self.predict(comment)
        for c, l in zip(categorys, self.labels):
            print(f'{l:-<44} {self.classify[c]}')


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name=f'{self.name}_W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name=f'{self.name}_b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                            K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)

        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
