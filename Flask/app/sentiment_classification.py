# -*- coding: utf-8 -*-

from flask import Flask
import pandas as pd
import codecs
import gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras_bert import get_custom_objects
from keras.models import load_model
import json
import warnings
import gc
import os
from app.models.config import Config


class SentimentClassification(object):
    tokenizer = None
    model = None

    @staticmethod
    def Init():

        #GPU
        if Config.GPUEnable == False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # 获取新的tokenizer
        if SentimentClassification.tokenizer is None:
            SentimentClassification.tokenizer = OurTokenizer(
                SentimentClassification.get_token_dict())

        if SentimentClassification.model is None:
            # 模型加载
            custom_objects = get_custom_objects()
            my_objects = {'acc_top2': SentimentClassification.acc_top2}
            custom_objects.update(my_objects)

            app = Flask(__name__)
            model_path = os.path.join(app.static_folder, Config.model_path)
            SentimentClassification.model = load_model(
                model_path, custom_objects=custom_objects)

    @staticmethod
    def get_token_dict():
        """
        # 将词表中的字编号转换为字典
        :return: 返回自编码字典
        """
        token_dict = {}
        app = Flask(__name__)
        dict_path = os.path.join(app.static_folder, Config.dict_path)
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict

    @staticmethod
    def acc_top2(y_true, y_pred):
        """
        :param y_true: 真实值
        :param y_pred: 训练值
        :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
        """
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    @staticmethod
    def seq_padding(X, padding=0):
        """
        :param X: 文本列表
        :param padding: 填充为0
        :return: 让每条文本的长度相同，用0填充
        """
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

    @staticmethod
    def get_sentiment(txt):
        """ 获取文本情感
        :param txt: 输入的文本
        :return: 情感分析的结果，json格式
        """
        text = str(txt)
        DATA_text = []
        DATA_text.append((text, to_categorical(0, 2)))
        text = data_generator(DATA_text, batch_size=10, shuffle=False,
                              tokenizer=SentimentClassification.tokenizer)
        test_model_pred = SentimentClassification.model.predict_generator(
            text.__iter__(), steps=len(text), verbose=0)
        if test_model_pred[0][0] > test_model_pred[0][1]:
            sentiment_label = 0
            sentiment_classification = '负面情感'
        else:
            sentiment_label = 1
            sentiment_classification = '正面情感'
        negative_prob = str(test_model_pred[0][0])
        positive_prob = str(test_model_pred[0][1])
        result = {'text': txt,
                  'sentiment_label': sentiment_label,
                  'sentiment_classification': sentiment_classification,
                  'negative_prob': negative_prob,
                  'positive_prob': positive_prob}
        return result

# 重写tokenizer


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')      # 不在列表的字符用[UNK]表示   UNK是unknown的意思
        return R


class data_generator:
    """
    data_generator只是一种为了节约内存的数据方式
    """
    Epoch = 2     # 迭代次数
    tokenizer = None

    def __init__(self, data, batch_size=32, shuffle=True, tokenizer=None):
        """
        :param data: 训练的文本列表
        :param batch_size:  每次训练的个数
        :param shuffle: 文本是否打乱
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:128]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = self.seq_padding(X1)
                    X2 = self.seq_padding(X2)
                    Y = self.seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []

    def seq_padding(self, X, padding=0):
        """
        :param X: 文本列表
        :param padding: 填充为0
        :return: 让每条文本的长度相同，用0填充
        """
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
