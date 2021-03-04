# -*- coding: utf-8 -*-
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
warnings.filterwarnings("ignore")

'''
-------------文本数据处理函数汇总-------------
'''


def get_stopwords(file):
    '''读取stopwords'''
    with open(file, 'r', encoding='utf-8') as f:
        stopwords = [s.strip() for s in f.readlines()]
    return stopwords


# 用jieba分词 对内容进行切割，并用空格连接分割后的词
def segmentData(contents, stopwords):
    def content2words(content, stopwords):
        content = re.sub('~+', '~', content)
        content = re.sub('\.+', '~', content)
        content = re.sub('～+', '～', content)
        content = re.sub('(\n)+', '\n', content)
        return ' '.join([word for word in jieba.cut(content) if word.strip() if word not in stopwords])

    seg_contents = [content2words(c, stopwords) for c in contents]
    return seg_contents


# 创建字典
def create_vocab(data, vocab_file, vocab_size):
    words = Counter()

    for content in data:
        words.update(content.split())

    special_tokens = ['<UNK>', '<SOS>', '<EOS>']

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in special_tokens:
            f.write(token + '\n')
        for token, _ in words.most_common(vocab_size - len(special_tokens)):
            f.write(token + '\n')


# 建立word2id和id2word映射
def read_vocab(vocab_file):
    word2id = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line.strip()
            word2id[word] = i
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


# 将分割后的句子 转化为id
def tokenizer(content, w2i, max_token=1000):
    '''content为一个 分词后的 句子'''
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


# 将评级转化为onehot形式
def onehot(label):
    onehot_label = [0, 0, 0, 0]
    onehot_label[label+2] = 1
    return onehot_label


# f1评估
def macro_f1(label_num, predicted, label):
    results = [{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for _ in range(label_num)]
    # 统计true positive, false positive, false negative, true negative
    for i, p in enumerate(predicted):
        l = label[i]
        for j in range(label_num):
            if p == j:
                if l == j:
                    results[j]['TP'] += 1
                else:
                    results[j]['FP'] += 1
            else:
                if l == j:
                    results[j]['FN'] += 1
                else:
                    results[j]['TN'] += 1

    precision = [0.0] * label_num
    recall = [0.0] * label_num
    f1 = [0.0] * label_num
    # 对每一类标签都计算precision, recall和f1, 并求平均
    for i in range(label_num):
        if results[i]['TP'] == 0:
            if results[i]['FP'] == 0 and results[i]['FN'] == 0:
                precision[i] = 1.0
                recall[i] = 1.0
                f1[i] = 1.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0
                f1[i] = 0.0
        else:
            precision[i] = results[i]['TP'] / \
                (results[i]['TP'] + results[i]['FP'])
            recall[i] = results[i]['TP'] / \
                (results[i]['TP'] + results[i]['FN'])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return sum(precision) / label_num, sum(recall) / label_num, sum(f1) / label_num


def get_test_comment(file):
    '''获取测试数据'''
    s_csv = pd.read_csv(file)
    return s_csv['content'].tolist()


'''
---------------------------------------文本数据 预处理函数--------------------------------------
'''


def processing_data(infile, labelfile, outfile, vocab_file, stopwords_file):
    '''数据预处理'''
    print('Loading stopwords...')
    stopwords = get_stopwords(stopwords_file)

    print('Loading data...')
    data = pd.read_csv(infile)

    print('Saving labels')
    with open(labelfile, 'w') as f:
        for label in data.columns[2:]:
            f.write(label + '\n')

    # 把句子分割成词
    print('Splitting content')
    contents = data['content'].tolist()
    seg_contents = segmentData(contents, stopwords)

    if not os.path.exists(vocab_file):
        print('Creating vocabulary...')
        create_vocab(seg_contents, vocab_file, 50000)

    print('Loading vocabulary...')
    w2i, _ = read_vocab(vocab_file)

    # word2id
    print('Tokenize...')
    token_contents = [tokenizer(c, w2i) for c in seg_contents]
    data['content'] = token_contents

    # 把标签转换成one hot形式
    print('One-hot label')
    for col in data.columns[2:]:
        # 这些列应该本来就是数字，这里只是转换为one-hot形式
        label = data[col].tolist()
        onehot_label = [onehot(l) for l in label]
        data[col] = onehot_label

    print('Saving...')
    data[data.columns[1:]].to_csv(outfile, index=False)


# 将数据集分割成 训练集和验证集
def split_train_valid(infile, trainfile, validfile):
    unsplit = pd.read_csv(infile)
    # 打乱数据
    unsplit = unsplit.sample(frac=1.0)

    valid = unsplit.iloc[0:5000]
    train = unsplit.iloc[5000:]
    valid.to_csv(validfile, index=False)
    train.to_csv(trainfile, index=False)


def prepare_data(file_path):
    '''将数据用dataset格式进行包装 ???'''
    csv_data = pd.read_csv(file_path)
    size = csv_data.shape[0]
    x = np.zeros((size, 1000))
    for i, c in enumerate(csv_data[csv_data.columns[0]].tolist()):
        x[i, :] = np.array(json.loads(c))
    y = np.zeros((size, 20, 4))
    for i, col in enumerate(csv_data.columns[1:]):
        y[:, i, :] = np.array([json.loads(l) for l in csv_data[col].tolist()])
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset


'''
---------------------------------------构建model函数--------------------------------------
-----------------------------两层双向LSTM+Attention作为共享层------------------------------
'''


class Attention(Layer):
    '''定义注意力层'''

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


# 定义构建model函数
def get_model(max_len, vocab_size, embedding_dim, lstm_unit, dropout_keep_rate, label_num, show_structure=False):
    inputs = Input((max_len,), name='input')
    # Embedding: Truns positive integers(indexs) into dense vectors of fixed size.
    embedding = Embedding(vocab_size, embedding_dim, name='embedding')(inputs)
    # return_sequences: 在输出序列中，返回单个hidden_state值 还是返回全部时间步的hidden
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


'''
---------------------------------------构建预测函数--------------------------------------
'''


class SentimentAnalysis:
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
        categorys = [np.argmax(p) for p in pred]
        return categorys

    # 打印结果
    def print_result(self, comment):
        categorys = self.predict(comment)
        for c, l in zip(categorys, self.labels):
            print(f'{l:-<44} {self.classify[c]}')


'''
---------------------------------------构建训练和验证模型函数--------------------------------------
'''


def train(flags, logger, train_dataset, valid_dataset, root_path=''):
    # 读取/初始化检查点参数
    logger.info('Loading checkpoint params...')
    if os.path.exists(root_path + flags.ckpt_params_path):
        with open(root_path + flags.ckpt_params_path, 'r') as f:
            params = json.loads(f.readline())
    else:
        params = {'epoch': 0, 'patience': 1, 'final_learn': 1, 'lr': 1e-3,
                  'pre_best_loss': 10000000, 'pre_best_metrics': (0.0, 0.0, 0.0),
                  'pre_best_ckpt_path': ''}

    # 加载模型
    logger.info('Initialize model...')
    model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                      flags.dropout_loss_rate, flags.label_num)
    if params['pre_best_ckpt_path']:
        model.load_weights(root_path + params['pre_best_ckpt_path'])
    # 选择优化器
    logger.info(f'Setting learning rate as {params["lr"]}')
    optimizer = tf.keras.optimizers.Adam(params['lr'])

    # 设置其他参数
    train_batch_nums = math.ceil(flags.num_train_sample / flags.batch_size)
    while True:
        params['epoch'] += 1

        # 初始化训练参数
        train_losses = 0
        valid_losses = 0
        avg_prec, avg_recall, avg_f1 = 0, 0, 0

        # 训练(train)
        with tqdm(enumerate(train_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)),
                  total=train_batch_nums) as pbar:
            for train_step, batch in pbar:
                x, y = batch
                y_true = [y[:, i, :] for i in range(y.shape[1])]
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss = [tf.keras.losses.categorical_crossentropy(y_i, l_i) for y_i, l_i
                            in zip(y_true, logits)]
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # 如果num_sample/batch_size不为整数，那么最后一个batch的size不等于batch_size
                train_losses += sum(sum(loss) / x.shape[0])

                if train_step == train_batch_nums - 1:
                    # 验证(valid)
                    logger.info(f'Validating at epoch{params["epoch"]}')
                    for _, batch in enumerate(valid_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)):
                        x, y = batch
                        y_true = [y[:, i, :] for i in range(y.shape[1])]
                        pred = model.predict(x)
                        loss = [tf.keras.losses.categorical_crossentropy(y_i, p_i) for y_i, p_i
                                in zip(y_true, pred)]
                        valid_losses += sum(sum(loss) / x.shape[0])
                        for i in range(x.shape[0]):
                            prec, recall, f1 = macro_f1(4, list(map(np.argmax, np.array(pred)[:, i, :])),
                                                        list(map(np.argmax, y[i])))
                            avg_prec += prec
                            avg_recall += recall
                            avg_f1 += f1

                    valid_losses = valid_losses / \
                        (flags.num_valid_sample / flags.batch_size)
                    avg_prec /= flags.num_valid_sample
                    avg_recall /= flags.num_valid_sample
                    avg_f1 /= flags.num_valid_sample

                pbar.set_description(f'Epoch{params["epoch"]}: train loss={train_losses / train_step + 1:.4f}, ' +
                                     f'valid loss={valid_losses:.4f}, ' +
                                     f'prec={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}')
        logger.info(
            f'At epoch{params["epoch"]}, training loss={train_losses:.4f}')

        # 检查点
        if valid_losses < params['pre_best_loss']:
            logger.info(f'Saving best checkpoint...')
            params['pre_best_loss'] = float(valid_losses)
            params['pre_best_metrics'] = (
                float(avg_prec), float(avg_recall), float(avg_f1))
            params['pre_best_ckpt_path'] = 'model/ckpt/best_ckpt'
            model.save_weights(root_path + params['pre_best_ckpt_path'])
            # 覆盖之前的最佳检查点参数
            with open(root_path + flags.ckpt_params_path, 'w') as f:
                json.dump(params, f)
            # 记录每次loss降低
            with open(root_path + flags.train_log, 'a') as f:
                f.write(
                    f'At epoch{params["epoch"]}, lr={params["lr"]}, train loss={train_losses / train_batch_nums:.4f}, valid loss{valid_losses:.4f}, precison={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}\n')

            params['patience'] = 1
        else:
            logger.info(f'Loss increased at epoch{params["epoch"]}!')
            if params['patience'] > 0:
                params['patience'] -= 1
            else:
                if params['final_learn'] > 0:
                    logger.info(f'Restore previous best checkpoint...')
                    model.load_weights(
                        root_path + params['pre_best_ckpt_path'])
                    params['final_learn'] -= 1
                    params['lr'] /= 10
                    logger.info(f'Decrease learning rate to {params["lr"]}')
                    optimizer = tf.keras.optimizers.Adam(params['lr'])
                    params['patience'] = 1
                else:
                    model.save_weights(root_path + flags.weight_save_path)
                    logger.info('End of Train.')
                    logger.info(
                        f'Best valid loss: {params["pre_best_loss"]:.4f}, precsion: {params["pre_best_metrics"][0]:.4f}, recall: {params["pre_best_metrics"][1]:.4f}, f1: {params["pre_best_metrics"][2]:.4f}')
                    break


def valid(flags, valid_dataset, root_path=''):
    model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                      flags.dropout_loss_rate, flags.label_num)

    model.load_weights(root_path+flags.weight_save_path)

    valid_losses = 0
    avg_prec, avg_recall, avg_f1 = 0, 0, 0

    for _, batch in enumerate(valid_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)):
        x, y = batch
        y_true = [y[:, i, :] for i in range(y.shape[1])]
        pred = model.predict(x)
        loss = [tf.keras.losses.categorical_crossentropy(y_i, p_i) for y_i, p_i
                in zip(y_true, pred)]
        valid_losses += sum(sum(loss) / x.shape[0])
        for i in range(x.shape[0]):
            prec, recall, f1 = macro_f1(4, list(map(np.argmax, np.array(pred)[:, i, :])),
                                        list(map(np.argmax, y[i])))
            avg_prec += prec
            avg_recall += recall
            avg_f1 += f1

    valid_losses = valid_losses / (flags.num_valid_sample / flags.batch_size)
    avg_prec /= flags.num_valid_sample
    avg_recall /= flags.num_valid_sample
    avg_f1 /= flags.num_valid_sample

    print(
        f'Valid loss={valid_losses:.4f}, precision={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}')


'''
---------------------------------------初始化参数--------------------------------------
'''


def initial_arguments():
    '''数据文件，模型路径，训练和验证的初始化参数'''
    parser = argparse.ArgumentParser()

    # data参数
    parser.add_argument('--root_path', type=str, default='',
                        help='the path of main.py')
    parser.add_argument('--raw_data', type=str,
                        default='data/train.csv', help='unprocessed data')
    parser.add_argument('--processed_data', type=str, default='data/processed.csv',
                        help='data after segment and tokenize')
    parser.add_argument('--train_data', type=str,
                        default='data/train_data.csv', help='path of training data file')
    parser.add_argument('--num_train_sample', type=int,
                        default=100000, help='num of train sample')
    parser.add_argument('--valid_data', type=str,
                        default='data/valid_data.csv', help='path of validating data file')
    parser.add_argument('--num_valid_sample', type=int,
                        default=5000, help='num of valid sample')
    parser.add_argument('--label_file', type=str,
                        default='data/label_names.txt', help='path of label name')
    parser.add_argument('--stopwords_file', type=str,
                        default='data/stopwords.txt', help='path of stopwords file')
    parser.add_argument('--vocab_file', type=str,
                        default='data/vocab.txt', help='path of vocabulary file')
    parser.add_argument('--test_comment_file', type=str,
                        default='data/test_comments.csv', help='comments used for testing')

    # 模型路径
    parser.add_argument('--weight_save_path', type=str,
                        default='model/best_weight', help='path of best weights')

    # 模型参数
    parser.add_argument('--max_len', type=int, default=1000,
                        help='max length of content')
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
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt_params_path', type=str, default='model/ckpt/ckpt_params.json',
                        help='path of checkpoint params')

    flags, unparsed = parser.parse_known_args()
    return flags


def initial_logging(logging_path='info.log'):
    '''
    日志函数
    参考：https://www.cnblogs.com/yyds/p/6901864.html
    '''
    logger = logging.getLogger((__name__))
    logger.setLevel(logging.INFO)
    # Handler 对象的作用是将消息分发到handler指定的位置(文件)
    file_handler = logging.FileHandler(filename=logging_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def train_main(train=False):
    '''模型训练主函数'''
    flags = initial_arguments()
    logger = initial_logging()
    if train:
        # 处理数据，假如已处理完成 则跳过
        if not os.path.exists(flags.train_data):
            logger.info('Processing raw data...')
            processing_data(flags.raw_data, flags.label_file,
                            flags.processed_data, flags.vocab_file, flags.stopwords_file)
            logger.info('Split data to train and valid...')
            split_train_valid(flags.processed_data,
                              flags.train_data, flags.valid_data)

        logger.info('Prepare data...')
        # 把数据转换成适合tensorflow dataset的数据格式
        train_dataset = prepare_data(flags.train_data)
        valid_dataset = prepare_data(flags.valid_data)

        # 开始训练
        logger.info('Start training...')
        train(flags, logger, train_dataset, valid_dataset, flags.root_path)

        logger.info('Finishing training...')


def predict_main(text=None):
    '''预测主函数'''
    flags = initial_arguments()
    logger = initial_logging()
    key_list = ['交通是否便利', '距离商圈远近', '是否容易寻找', '排队等候时间', '服务人员态度',
                '是否容易停车', '点菜/上菜速度', '价格水平', '性价比', '折扣力度',
                '装修情况', '嘈杂情况', '就餐空间', '卫生情况', '分量',
                '口感', '外观', '推荐程度', '本次消费感受', '再次消费的意愿']
    logger.info('Initialize model')
    sa = SentimentAnalysis(flags)
    if text is None:
        test_comment = get_test_comment(flags.test_comment_file)
        # 随便获取一个测试数据
        comment = np.random.choice(test_comment)
        value_list = sa.predict(comment)
    else:
        value_list = sa.predict(text)
    results = dict(zip(key_list, value_list))
    return results


def label2tag(num):
    '''数字标签标签转成文字标签'''
    num = int(num)
    if num == 3:
        return '正面情感'
    elif num == 2:
        return '中性情感'
    elif num == 1:
        return '负面情感'
    elif num == 0:
        return '未提及'
    else:
        return '出错了'


if __name__ == '__main__':
    # GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 默认不训练
    train_main(train=True)
    text = '第三次参加大众点评网霸王餐的活动。这家店给人整体感觉一般。首先环境只能算中等，其次霸王餐提供的菜品也不是很多，当然商家为了避免参加霸王餐吃不饱的现象，给每桌都提供了至少六份主食，我们那桌都提供了两份年糕，第一次吃火锅会在桌上有这么多的主食了。整体来说这家火锅店没有什么特别有特色的，不过每份菜品分量还是比较足的，这点要肯定！至于价格，因为没有看菜单不了解，不过我看大众有这家店的团购代金券，相当于7折，应该价位不会很高的！最后还是要感谢商家提供霸王餐，祝生意兴隆，财源广进'
    # 返回数据中
    # 0为'未提及'
    # 1为'负面情感'
    # 2为'中性情感'
    # 3为'正面情感'
    predict_result = predict_main(text)
    print('predict_result:', predict_result)

'''
训练模型日志，最好是第5轮，验证loss最低，f1_score为0.7393

Epoch1: train loss=12.8546, valid loss=9.1229, prec=0.7264, recall=0.7182, f1=0.7054: 100%|██████████| 3125/3125 [28:47<00:00,  1.81it/s]
Epoch2: train loss=9.3220, valid loss=8.3498, prec=0.7521, recall=0.7465, f1=0.7332: 100%|██████████| 3125/3125 [28:39<00:00,  1.82it/s]
Epoch3: train loss=8.3302, valid loss=8.3642, prec=0.7552, recall=0.7509, f1=0.7370: 100%|██████████| 3125/3125 [28:05<00:00,  1.85it/s]
Epoch4: train loss=7.5604, valid loss=8.7710, prec=0.7529, recall=0.7510, f1=0.7357: 100%|██████████| 3125/3125 [28:07<00:00,  1.85it/s]
Epoch5: train loss=7.9335, valid loss=8.3458, prec=0.7570, recall=0.7533, f1=0.7393: 100%|██████████| 3125/3125 [28:09<00:00,  1.85it/s]
Epoch6: train loss=7.6340, valid loss=8.4231, prec=0.7581, recall=0.7545, f1=0.7405: 100%|██████████| 3125/3125 [28:10<00:00,  1.85it/s]
Epoch7: train loss=7.3926, valid loss=8.5244, prec=0.7579, recall=0.7547, f1=0.7404: 100%|██████████| 3125/3125 [28:05<00:00,  1.85it/s]
'''
