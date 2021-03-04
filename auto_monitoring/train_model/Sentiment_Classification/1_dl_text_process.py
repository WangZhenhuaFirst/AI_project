# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.utils import shuffle


def tokens(text):
    '''读取数据并使用tokens函数提取中文'''
    return ''.join(re.findall('[\u4e00-\u9fff]', text))


def read_ratings_data():
    '''读取评价数据'''
    print('开始读取文件。。。')
    pd_ratings = pd.read_csv('yf_dianping/ratings.csv')
    ratings_data = pd_ratings[(pd_ratings['rating'].notnull()) & (
        pd_ratings['comment'].notnull())][['comment', 'rating']]
    del pd_ratings
    ratings_data = ratings_data.loc[:, ['comment', 'rating']]
    # 设置标签,0为负面情感，1为正面情感
    ratings_data.loc[ratings_data['rating'] < 3, 'rating'] = 0
    ratings_data.loc[ratings_data['rating'] >= 3, 'rating'] = 1
    ratings_data['rating'] = ratings_data['rating'].astype(int)

    data0 = ratings_data[(ratings_data['rating'] == 0) & (
        ratings_data['comment'].str.len() > 20)][:250000]
    data1 = ratings_data[(ratings_data['rating'] == 1) & (
        ratings_data['comment'].str.len() > 20)][:len(data0)]

    data = pd.concat([data0, data1])
    # 把数据打乱
    data = shuffle(data, random_state=0)
    print('读取完毕。。。')
    # 使用apply对comment列去掉多余符号
    # data['comment'] = data['comment'].apply(cut_text)

    return data


if __name__ == '__main__':
    # print('开始读取文件')
    data = read_ratings_data()
    print('正在处理文本，请耐心等待')
    data['comment'] = data['comment'].apply(tokens)
    print('处理完毕，正在导出csv文件')
    # 把数据导出为txt文本，index和header为空，分隔符 为4个空格，即"\t"
    data[['comment', 'rating']][:438000].to_csv(
        'train_data/bert_train.csv', index=None)
    data[['comment', 'rating']][438000:].to_csv(
        'train_data/bert_valid.csv', index=None)
    print('导出完毕')
