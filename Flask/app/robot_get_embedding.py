# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
import pickle
# import keras.backend.tensorflow_backend as tb


def build_tokenizer(dict_path):
    '''加载tokenizer'''
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    return tokenizer


def build_model(config_path, checkpoint_path):
    '''加载 bert 模型'''
    # tb._SYMBOLIC_SCOPE.value = False
    bert_model = build_transformer_model(config_path, checkpoint_path)
    return bert_model


def seq_padding(ids, max_len):
    '''让每条文本的长度相同，用0填充'''
    if len(ids) < max_len:
        ids.extend([0]*(max_len - len(ids)))
    return ids


# def generate_mask(sen_list, max_len):
#     '''生成mask矩阵'''
#     len_list = [len(i) if len(i) <= max_len else max_len for i in sen_list]
#     array_mask = np.array(
#         [np.hstack((np.ones(j), np.zeros(max_len-j))) for j in len_list])
#     return np.expand_dims(array_mask, axis=2)


def extract_emb_feature(model, tokenizer, sentences, max_len, mask_if=False):
    '''生成句子向量特征'''
    # mask = generate_mask(sentences, max_len)
    token_ids_list = []
    segment_ids_list = []
    result = []
    # sentences = tqdm(sentences)
    for sen in sentences:
        token_ids, segment_ids = tokenizer.encode(sen, maxlen=max_len)
        token_ids = seq_padding(token_ids, max_len)
        segment_ids = seq_padding(segment_ids, max_len)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)

    # print('Generating a sentence embedding')

    result = model.predict([np.array(token_ids_list), np.array(
        segment_ids_list)], verbose=0, batch_size=32)
    # if mask_if:
    #     result = result * mask
    return np.mean(result, axis=1)


def load_data(path):
    '''加载txt数据为list'''
    result = []
    with open('{}'.format(path), 'r', encoding='utf-8') as f:
        for line in f:
            result.append(line.strip('\n').split(',')[0])
    return result


def save_pkl(name, dic):
    '''保存文件为pkl'''
    with open("{}.pkl".format(name), 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(dic, fo)
