# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
import pickle


def build_tokenizer(dict_path):
    '''
    参考：https://bert4keras.spaces.ac.cn/
    加载tokenizer
    '''
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
    sentences = tqdm(sentences)
    for sen in sentences:
        token_ids, segment_ids = tokenizer.encode(sen, maxlen=max_len)
        token_ids = seq_padding(token_ids, max_len)
        segment_ids = seq_padding(segment_ids, max_len)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)

    print('Generating a sentence embedding')

    result = model.predict(
        [np.array(token_ids_list), np.array(segment_ids_list)], verbose=0, batch_size=32)
    # if mask_if:  # 将句子padding为0的的部分mask掉
    #     result = result * mask
    return np.mean(result, axis=1)  # 应该是每个词都由多个数字表示，用mean取平均值来代表一个词


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


if __name__ == '__main__':
    # 设置bert文件路径
    dict_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/vocab.txt"
    config_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/bert_config.json"
    checkpoint_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/bert_model.ckpt"
    # 加载tokenizer
    tokenizer = build_tokenizer(dict_path)
    # 加载bert模型
    model = build_model(config_path, checkpoint_path)

    '''问答数据 向量化处理'''
    # 加载question文本
    data = pd.read_csv(r'../Flask/app/static/model04/dataset.csv')
    finance_data = data.drop_duplicates('question')
    finance_data = finance_data[(finance_data['question'] != '') & (
        finance_data['answer'] != '')]
    finance_data_question_list = finance_data['question'].astype(str).to_list()
    finance_data_answer_dict = dict(
        zip(finance_data.index, finance_data.answer))
    # 保存answer
    save_pkl('finance_data_answer_dict', finance_data_answer_dict)
    # 开始抽取文本特征，获取句向量
    sentence_emb = extract_emb_feature(
        model, tokenizer, finance_data_question_list, max_len=30)
    # 保存文本特征向量化的list为pkl
    save_pkl('sentence_emb', sentence_emb)
    # 把问题和问题对应的向量 组成dict
    res = dict(zip(finance_data_question_list, sentence_emb))
    # 字典文件可以用做向量检索
    save_pkl('sentence_emb_dict', res)

    # v1 = extract_emb_feature(model, tokenizer, ['吞卡证明啊'], max_len=30)
    # v2 = extract_emb_feature(model, tokenizer, ['手机为什么不能转钱？'], max_len=30)
    # print('v1:', v1)
    # print('v2:', v2)

'''
  0%|          | 0/328172 [00:00<?, ?it/s]

  1%|          | 3184/328172 [00:00<00:10, 31526.37it/s]

  2%|▏         | 6398/328172 [00:00<00:10, 31707.75it/s]

  3%|▎         | 9719/328172 [00:00<00:09, 32050.84it/s]

  4%|▍         | 13084/328172 [00:00<00:09, 32514.38it/s]
  .
  .
  .
  .
 95%|█████████▍| 310978/328172 [00:10<00:00, 26671.20it/s]

 96%|█████████▌| 313800/328172 [00:10<00:00, 26739.60it/s]

 96%|█████████▋| 316648/328172 [00:10<00:00, 27233.14it/s]

 97%|█████████▋| 319471/328172 [00:10<00:00, 27515.28it/s]

 98%|█████████▊| 322381/328172 [00:10<00:00, 27892.04it/s]

100%|██████████| 328172/328172 [00:11<00:00, 29450.29it/s]

Generating a sentence embedding

328172/328172 [==============================] - 6395s 19ms/step
'''
