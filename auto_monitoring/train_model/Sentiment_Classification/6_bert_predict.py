# -*- coding: utf-8 -*-
import pandas as pd
import codecs
import gc
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras_bert import get_custom_objects
from keras.models import load_model
import json
import warnings
warnings.filterwarnings("ignore")


# 初始参数设置
maxlen = 128   # 设置序列长度为100，要保证序列长度不超过512
Batch_size = 32    # 批量运行的个数
Epoch = 2     # 迭代次数


def get_train_test_data():
    '''读取训练数据和测试数据 '''
    train_df = pd.read_csv('/train_data/bert_train.csv').astype(str)
    test_df = pd.read_csv('/train_data/bert_valid.csv').astype(str)
    # 训练数据、测试数据和标签转化为模型输入格式
    DATA_LIST = []
    for data_row in train_df.iloc[:].itertuples():
        DATA_LIST.append(
            (data_row.comment, to_categorical(data_row.rating, 2)))
    DATA_LIST = np.array(DATA_LIST)

    DATA_LIST_TEST = []
    for data_row in test_df.iloc[:].itertuples():
        DATA_LIST_TEST.append(
            (data_row.comment, to_categorical(data_row.rating, 2)))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)

    data = DATA_LIST
    data_test = DATA_LIST_TEST

    X_train, X_valid = train_test_split(data, test_size=0.05, random_state=0)
    return X_train, X_valid, data_test


def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


class OurTokenizer(Tokenizer):
    '''重写tokenizer'''

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


def seq_padding(X, padding=0):
    """
    :param X: 文本列表
    :param padding: 填充为0
    :return: 让每条文本的长度相同，用0填充
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    """
    data_generator只是一种为了节约内存的数据方式
    """

    def __init__(self, data, batch_size=Batch_size, shuffle=True):
        """
        :param data: 训练的文本列表
        :param batch_size:  每次训练的个数
        :param shuffle: 文本是否打乱
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


def acc_top2(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 训练值
    :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert(nclass):
    """
    :param nclass: 文本分类种类
    :return: 构建的bert模型
    """
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True
    # 构建模型
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


def get_sentiment(txt):
    """ 获取文本情感
    :param txt: 输入的文本
    :return: 情感分析的结果，json格式
    """
    text = str(txt)
    DATA_text = []
    DATA_text.append((text, to_categorical(0, 2)))
    # DATA_text = np.array(DATA_text)
    text = data_generator(DATA_text, batch_size=10, shuffle=False)
    test_model_pred = model.predict_generator(
        text.__iter__(), steps=len(text), verbose=0)
    # print('预测结果',test_model_pred)
    # print(np.argmax(test_model_pred))
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
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    # bert预训练模型路径设置
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
    # 获取新的tokenizer
    tokenizer = OurTokenizer(get_token_dict())

    # 模型加载
    custom_objects = get_custom_objects()
    my_objects = {'acc_top2': acc_top2}
    custom_objects.update(my_objects)

    model_path = 'model/bertkeras_model.h5'
    model = load_model(model_path, custom_objects=custom_objects)

    # 单独评估一个本来分类
    text = '''肉类不新鲜，菜品比以前少，这样子就当吃麻辣烫一样，不过比麻辣烫还要差一点点。觉得都不新鲜的'''
    predict_result = get_sentiment(text)
    print('predict_result:', predict_result)

    # del model # 删除模型减少缓存
    gc.collect()  # 清理内存
    K.clear_session()  # clear_session就是清除一个session


'''
predict_result: {"text": "肉类不新鲜，菜品比以前少，这样子就当吃麻辣烫一样，不过比麻辣烫还要差一点点。觉得都不新鲜的",
"sentiment_label": 0, "sentiment_classification": "负面情感",
"negative_prob": "0.93662393", "positive_prob": "0.06337611"}

'''
