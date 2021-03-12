# -*- coding: utf-8 -*-
import pandas as pd
import codecs
import gc
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras import Input
# from keras.layers import *
from keras.layers import Lambda, Dense
# from keras.callbacks import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
# 读取训练集和测试集
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
    train_df = pd.read_csv('train_data/bert_train.csv').astype(str)
    test_df = pd.read_csv('train_data/bert_valid.csv').astype(str)
    # 训练数据、测试数据和标签转化为模型输入格式
    DATA_LIST = []
    for data_row in train_df.iloc[:].itertuples():
        # Converts a class vector (integers) to binary class matrix.
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
    将词表中的字+ 编号/行号 转换为字典————字:行号
    :return: 返回 自编码字典
    参考：https://github.com/CyberZHG/keras-bert
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)  # 也就是从0开始给一个编号
    return token_dict


class OurTokenizer(Tokenizer):
    '''
    The Tokenizer class is used for splitting texts and generating indices.
    重写tokenizer
    参考：https://github.com/CyberZHG/keras-bert
    '''

    def _tokenize(self, text):
        '''
        本来Tokenizer有自己的_tokenize方法，这里重写了这个方法，是要保证tokenize之后的结果，跟原来的字符串等长（如果算上两个标记，就是等长再加2）。
        Tokenizer自带的_tokenize会自动去掉空格，然后有些字符会粘在一块输出，导致tokenize之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。
        为了避免这种麻烦，还是自己重写一遍好了～主要就是用[unused1]来表示空格类字符，而其余的不在列表的字符用[UNK]表示，
        其中[unused*]这些标记是未经训练的（随机初始化），是Bert预留出来用来增量添加词汇的标记，所以我们可以用它们来指代任何新字符。
        '''
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')      # 不在列表的字符用[UNK]表示,UNK是unknown的意思
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
    '''data_generator只是一种为了节约内存的数据方式'''

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
    :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别 即认为预测正确
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert(nclass):
    """
    参考：https://kexue.fm/archives/6736
    :param nclass: 文本分类种类
    :return: 构建的bert模型
    """
    # 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
    # 真正调用Bert的也就只有load_trained_model_from_checkpoint 一行代码，剩下的只是普通的Keras操作
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True
    # 构建模型
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    # “有什么原则来指导Bert后面应该要接哪些层？”。答案是：用尽可能少的层 来完成你的任务。
    # 比如上述情感分析 只是一个二分类任务，你就取出第一个向量然后加个Dense(1)就好了，
    # 不要想着多加几层Dense，更加不要想着接个LSTM再接Dense；
    # 如果你要做序列标注（比如NER），那你就接个Dense+CRF就好，也不要多加其他东西。
    # 总之，额外加的东西尽可能少。一是因为Bert本身就足够复杂，它有足够能力应对你要做的很多任务；
    # 二来你自己加的层都是随机初始化的，加太多会对Bert的预训练权重造成剧烈扰动，容易降低效果甚至造成模型不收敛

    # 这里x1_in，x2_in 作为bert_model的输入是什么意思？引入了Bert作为编码器
    x = bert_model([x1_in, x2_in])
    # Wraps arbitrary expressions as a Layer object.
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量 用来做分类
    p = Dense(nclass, activation='softmax')(x)
    # 参考：https://keras.io/api/models/model/#model-class
    # Model groups layers into an object with training and inference features.
    # 只需要将输入层和输出层作为参数，
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


def train_model():
    """
    训练模型
    :return: 验证预测集，测试预测集，训练好的模型
    """
    # 搭建模型参数
    print('正在加载模型，请耐心等待....')
    model = build_bert(2)  # 二分类模型
    print('模型加载成功，开始训练....')
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=3)  # 早停法，防止过拟合
    # Reduce learning rate when a metric has stopped improving
    # 当评价指标不再提升时，减小学习率
    plateau = ReduceLROnPlateau(
        monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=2)
    checkpoint = ModelCheckpoint('model/bertkeras_model.h5', monitor='val_accuracy', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型
    # 获取数据并文本序列化
    X_train, X_valid, data_test = get_train_test_data()
    train_D = data_generator(X_train, shuffle=True)
    valid_D = data_generator(X_valid, shuffle=True)
    test_D = data_generator(data_test, shuffle=False)

    # 模型训练
    # fit: Trains the model for a fixed number of epochs(iterations on a dataset)
    # fit_generator: Fits the model on data yielded batch-by-batch by a Python generator
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=Epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        # List of callbacks to apply during training.
        callbacks=[early_stopping, plateau, checkpoint],
    )
    # 对验证集和测试集进行预测
    valid_model_pred = model.predict_generator(
        valid_D.__iter__(), steps=len(valid_D), verbose=1)
    test_model_pred = model.predict_generator(
        test_D.__iter__(), steps=len(test_D), verbose=1)
    # 将预测概率值转化为类别值
    valid_pred = [np.argmax(x) for x in valid_model_pred]
    test_pred = [np.argmax(x) for x in test_model_pred]
    y_true = [np.argmax(x) for x in X_valid[:, 1]]

    return valid_pred, test_pred, y_true, model, data_test


def get_metrics_scores(y_true, y_pred, type='metrics'):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param type: 预测种类
    :return: 评估指标
    """
    confusion_matrix_scores = metrics.confusion_matrix(y_true, y_pred)
    accuracy_scores = metrics.accuracy_score(y_true, y_pred)
    precision_scores = metrics.precision_score(y_true, y_pred, average=None)
    # Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    precision_score_macro_average = metrics.precision_score(
        y_true, y_pred, average='macro')
    precision_score_micro_average = metrics.recall_score(
        y_true, y_pred, average='micro')
    f1_scores = metrics.f1_score(y_true, y_pred, average='weighted')

    print(type, '...')
    print('混淆矩阵：', confusion_matrix_scores)
    print('准确率：', accuracy_scores)
    print('类别精度：', precision_scores)  # 不求平均
    print('宏平均精度：', precision_score_macro_average)
    print('微平均召回率:', precision_score_micro_average)
    print('加权平均F1得分:', f1_scores)
    return confusion_matrix_scores, accuracy_scores, precision_scores, precision_score_macro_average, precision_score_micro_average, f1_scores


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
    # 训练和预测
    valid_pred, test_pred, y_true, model, data_test = train_model()
    # 评估验证集
    get_metrics_scores(valid_pred, y_true, type='valid metrics')
    # 评估测试集
    get_metrics_scores(test_pred, [np.argmax(x)
                                   for x in data_test[:, 1]], type='test metrics')
    # 模型保存
    model_path = 'model/bertkeras_model.h5'
    model.save(model_path)

    # 模型加载
    custom_objects = get_custom_objects()
    my_objects = {'acc_top2': acc_top2}
    custom_objects.update(my_objects)
    model = load_model(model_path, custom_objects=custom_objects)

    # 单独评估一个本来分类
    text = '肉类不新鲜，菜品比以前少，这样子就当吃麻辣烫一样，不过比麻辣烫还要差一点点。觉得都不新鲜的'''
    predict_result = get_sentiment(text)
    print('predict_result', predict_result)

    # del model # 删除模型减少缓存
    gc.collect()  # 清理内存
    K.clear_session()  # clear_session就是清除一个session, resets all state generated by Keras


'''
正在加载模型，请耐心等待....
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
model_1 (Functional)            (None, None, 768)    101677056   input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 768)          0           model_1[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            1538        lambda[0][0]                     
==================================================================================================
Total params: 101,678,594
Trainable params: 101,678,594
Non-trainable params: 0
__________________________________________________________________________________________________
None
模型加载成功，开始训练....
Epoch 1/2
13004/13004 [==============================] - 14152s 1s/step - loss: 0.2978 - accuracy: 0.8759 - acc_top2: 1.0000 - val_loss: 0.2555 - val_accuracy: 0.8967 - val_acc_top2: 1.0000
Epoch 2/2
13004/13004 [==============================] - 14174s 1s/step - loss: 0.2363 - accuracy: 0.9057 - acc_top2: 1.0000 - val_loss: 0.2565 - val_accuracy: 0.8958 - val_acc_top2: 1.0000
685/685 [==============================] - 220s 316ms/step
123/123 [==============================] - 39s 314ms/step
train metrics ...
混淆矩阵： [[9870 1055]
 [1223 9752]]
准确率： 0.8959817351598174
类别精度： [0.88975029 0.90237809]
宏平均精度： 0.8960641906268354
微平均召回率: 0.8959817351598174
加权平均F1得分: 0.8959774355873534
test metrics ...
混淆矩阵： [[1683  194]
 [ 249 1782]]
准确率： 0.8866427840327533
类别精度： [0.87111801 0.90182186]
宏平均精度： 0.8864699373852691
微平均召回率: 0.8866427840327533
加权平均F1得分: 0.8866832245535813
'''
