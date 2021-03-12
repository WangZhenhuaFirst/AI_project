# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import logging
import re
import jieba
from sklearn.utils import shuffle


def getStopwords():
    '''获取中文停顿词'''
    stopwords = [line.strip() for line in open(
        './chinese_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def tokens(text):
    '''读取数据并使用tokens函数 提取中文'''
    return ''.join(re.findall('[\u4e00-\u9fff]', text))


def cut_text(text):
    '''中文分词函数，用正则 去除多余的符号'''
    text = str(text)
    # text = re.sub('\\\\n|[\n\u3000\r]', ' ', text)
    text = ''.join(re.findall('[\u4e00-\u9fff]', text))
    seg_list = jieba.cut(text)
    sentence_segment = []
    for word in seg_list:
        if word not in stopwords:
            sentence_segment.append(word.strip())
    # 把已去掉停用词的sentence_segment用 空格' ' 拼接起来
    seg_res = ' '.join(sentence_segment)
    return seg_res


def read_ratings_data():
    '''读取评论数据'''
    print('开始读取文件')
    # 用pd.read_csv方法读取语料
    pd_ratings = pd.read_csv('yf_dianping/ratings.csv')
    ratings_data = pd_ratings[(pd_ratings['rating'].notnull()) & (
        pd_ratings['comment'].notnull())][['comment', 'rating']]
    del pd_ratings
    ratings_data = ratings_data.loc[:, ['comment', 'rating']]
    # 设置标签,0为负面情感，1为正面情感
    ratings_data.loc[ratings_data['rating'] < 3, 'rating'] = 0
    ratings_data.loc[ratings_data['rating'] >= 3, 'rating'] = 1
    ratings_data['rating'] = ratings_data['rating'].astype(int)  # 转换数据类型

    data0 = ratings_data[(ratings_data['rating'] == 0) & (
        ratings_data['comment'].str.len() > 20)][:200000]
    data1 = ratings_data[(ratings_data['rating'] == 1) & (
        ratings_data['comment'].str.len() > 20)][:len(data0)]

    data = pd.concat([data0, data1])
    # 把数据打乱
    data = shuffle(data, random_state=0)
    data[['comment', 'rating']].to_csv(
        'ratings_data.txt', index=None, sep='\t')
    print('读取完毕！')
    return data


if __name__ == '__main__':
    # 返回当前运行的py文件名称
    program = os.path.basename(sys.argv[0])
    # logging.basicConfig函数中，可以指定日志的输出格式format，这个参数可以输出很多有用的信息
    # %(asctime)s: 打印日志的时间     %(levelname)s: 打印日志级别名称      %(message)s: 打印日志信息
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.getLogger(name)方法进行初始化，name可以不填。通常logger的名字我们对应模块名
    logger = logging.getLogger(program)  # logging.getLogger(logger_name)
    # logger.info打印程序运行是的正常的信息，用于替代print输出
    logger.info('running ' + program + ': segmentation of corpus by jieba')
    # 获取中文停顿词
    stopwords = getStopwords()
    ratings_data = read_ratings_data()
    print('开始分词')
    ratings_data['comment'] = ratings_data['comment'].apply(cut_text)
    print('分词完毕！保存分词文件！')
    ratings_data.to_csv('seg_ratings_data.txt', index=None, sep='\t')
