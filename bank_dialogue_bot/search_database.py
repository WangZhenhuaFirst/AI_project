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
from get_embedding import build_tokenizer, build_model, extract_emb_feature
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


class SearchDataBase:
    '''生成回复'''
    # 定义历史对话记录history为全局变量
    global history

    def __init__(self):
        # 最大历史记录长度
        self.dict_name = r"../Flask/app/static/model04/search_database/sentence_emb_dict.pkl"
        self.ann_save_path = r"../Flask/app/static/model04/search_database/annoy.index"
        self.id2word_path = r"../Flask/app/static/model04/search_database/id2word.pkl"
        self.word2id_path = r"../Flask/app/static/model04/search_database/word2id.pkl"
        self.answer_path = r'../Flask/app/static/model04/search_database/finance_data_answer_dict.pkl'

        # 构建索引时，以下两行注释掉
        self.id2word = pickle.load(open(self.id2word_path, "rb"))
        self.word2id = pickle.load(open(self.word2id_path, "rb"))

        # Length of item vector that will be indexed
        self.annoy_index = AnnoyIndex(768, metric='angular')
        self.annoy_index.load(self.ann_save_path)  # 构建索引时本行注释掉
        self.answer = pickle.load(open(self.answer_path, "rb"))
        # 设置bert文件路径
        self.dict_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/vocab.txt"
        self.config_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/bert_config.json"
        self.checkpoint_path = r"../Flask/app/static/model04/search_database/chinese_L-12_H-768_A-12/bert_model.ckpt"
        # 加载tokenizer
        self.tokenizer = build_tokenizer(self.dict_path)
        # 加载bert模型
        self.model = build_model(self.config_path, self.checkpoint_path)
        # 定义阈值
        self.threshold = 0.96

    # 构建索引，只需构建一次
    def build_index(self, tree_num=500):
        # 自定义的 读取word2vec的函数
        items_vec = pickle.load(open(self.dict_name, "rb"))
        # Annoy: to search for points in space that are close to a given query point.
        # nearest neighbor search
        # 向量维度为768
        a = AnnoyIndex(768, metric='angular')
        i = 0
        id2word, word2id = dict(), dict()
        for word in items_vec.keys():
            a.add_item(i, items_vec[word])  # 添加的元素是 句子的embedding
            id2word[i] = word
            word2id[word] = i
            i += 1
        a.build(tree_num)
        # 如果路径中已有保存的文件会报错,OSError: Unable to open: Invalid argument (22)OSError: Unable to open: Invalid argument (22)
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
            print('weights: ', weights)
            if weights > self.threshold:
                # idxes是问题本身
                idxes_weight_item[idxes[n]] = weights
        if print_value:
            print(idxes_weight_item)
        if idxes_weight_item != {}:
            max_idxes = max(zip(idxes_weight_item.values(),
                                idxes_weight_item.keys()))[1]
            return self.answer[max_idxes]
        else:
            return {}


if __name__ == '__main__':
    SDB = SearchDataBase()
    # SDB.build_index(1000)  # 向量索引只需要构建一次
    s = time.time()
    answer_response = SDB.annoy_search('你好')
    print('answer_response: ', answer_response)
    e = time.time()
    print(e-s)


# # 读取pkl文件
# def load_dict(name):
#     with open("{}.pkl".format(name), 'rb') as fo:
#         dic = pickle.load(fo, encoding='bytes')
#     return dic


# # 构建索引，只需构建一次
# def build_index(tree_num = 500):
# 	global id2word, word2id

# 	# 自定义的读取word2vec的函数
# 	items_vec = load_dict(dict_name)
# 	# 向量维度为768
# 	a = AnnoyIndex(768, metric='angular')
# 	i = 0
# 	id2word, word2id = dict(), dict()
# 	for word in items_vec.keys():
# 		a.add_item(i, items_vec[word])
# 		id2word[i] = word
# 		word2id[word] = i
# 		i += 1
# 	a.build(tree_num)
# 	a.save(ann_save_path)
# 	pickle.dump(id2word, open(id2word_path, "wb"))
# 	pickle.dump(word2id, open(word2id_path, "wb"))

# # 实际运行时加载索引
# def annoy_init():
# 	global id2word, word2id, annoy_index
# 	id2word = pickle.load(open(id2word_path, "rb"))
# 	word2id = pickle.load(open(word2id_path, "rb"))
# 	annoy_index = AnnoyIndex(768, metric='angular')
# 	annoy_index.load(ann_save_path)


# # 近似检索，query为编码后的向量
# def annoy_search(query, topn=10):
#     global id2word, word2id, annoy_index
#     idxes, dists = annoy_index.get_nns_by_vector(query, topn, include_distances=True)
#     sim_words_item = {}
#     for i, j in zip(idxes,dists):
#         # 余弦相似度归一化的计算公式，
#         weights = 0.5*(abs(1-j))+0.5
#         sim_words_item[id2word[i]] = weights
#     return sim_words_item,idxes


# #build_index()
# annoy_init()
# v = extract_emb_feature(model, tokenizer, ['人工智能和区块链有什么关系？'], max_len = 30)[0]
# sim_words_item,idxes = annoy_search(v, topn=10)
# print(sim_words_item)
# # from sklearn.metrics.pairwise import cosine_similarity
# # x = extract_emb_feature(model, tokenizer, ['刘德华是谁啊'], max_len = 30)[0]
# # y = extract_emb_feature(model, tokenizer, ['朱丽倩是谁啊'], max_len = 30)[0]
# # cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))

# # idxes, dists = annoy_index.get_nns_by_vector(v, 10, include_distances=True)
# # #idxes = [id2word[i] for i in idxes]
# # #similars = list(zip(idxes, dists))
# # sim_words_item = {}
# # for i, j in zip(idxes,dists):
# #     # 余弦相似度归一化的计算公式，
# #     weights = 0.5*(abs(1-j))+0.5
# #     # 取权重大于定义的weight
# #     #if weights > weight:
# #     sim_words_item[id2word[i]] = weights

# #     #return sim_words_item
# # word2id[1]


# # 构建索引，只需构建一次
# def build_index(self, tree_num = 500):
#     # 自定义的读取word2vec的函数
#     items_vec = pickle.load(open(self.dict_name, "rb"))
#     # 向量维度为768
#     a = AnnoyIndex(768, metric='angular')
#     i = 0
#     id2word, word2id = dict(), dict()
#     for word in items_vec.keys():
#         a.add_item(i, items_vec[word])
#         id2word[i] = word
#         word2id[word] = i
#         i += 1
#     a.build(tree_num)
#     a.save(self.ann_save_path)
#     pickle.dump(id2word, open(self.id2word_path, "wb"))
#     pickle.dump(word2id, open(self.word2id_path, "wb"))


''' 基于annoy算法提取相似词'''
'''
# 保存字典文件为pkl
def save_dict(name,dic):
    with open("{}.pkl".format(name), 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(dic, fo)    
    
# 读取pkl文件
def load_dict(name):
    with open("{}.pkl".format(name), 'rb') as fo:     # 读取pkl文件
        dic = pickle.load(fo, encoding='bytes')      
    return dic


def get_similar_words(text, topn = 20, weight = 0.5):#
           
    sim_words = bk_index.get_nns_by_item(word_index[text], topn, include_distances=True)
    #print(keywords)
    sim_words_item = {}
    for item, j in zip(sim_words[0],sim_words[1] ):
        # 余弦相似度归一化的计算公式，
        weights = 0.5*(abs(1-j))+0.5
        # 取权重大于定义的weight
        if weights > weight:
            sim_words_item[reverse_word_index[item]] = weights  

    return sim_words_item  

bk_wv_model = load_dict('sentence_emb_dict')
word_index = OrderedDict()
for counter, key in enumerate(bk_wv_model.keys()):
    word_index[key] = counter

with open('bk_word_index.json', 'w') as fp:
    json.dump(word_index, fp)
# 保存词汇ID映射表，加速向量检索时需要用
save_dict('word_index',word_index)

bk_index = AnnoyIndex(768,'angular')   
i = 0    
for key in bk_wv_model.keys():
    v = bk_wv_model[key]
    bk_index.add_item(i, v)
    i += 1

bk_index.build(200)
# 保存向量索引文件，大概2.3G,注意：【【【保存路径不能有中文】】】
bk_index.save('H:/qa_index_build10.index')  
bk_index.save('H:/qa_index_build10.ann')   
# 准备一个反向id==>word映射词表
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
# 保存词汇反向id==>word映射词表，加速向量检索时需要用
save_dict('reverse_word_index',reverse_word_index)
text = '定期没到期手机能不能转吗？'
bk_index.get_nns_by_item(word_index[text], 11)
for item in bk_index.get_nns_by_item(word_index[text], 11):
    print(reverse_word_index[item])
bk_sim_words = get_similar_words('保险',1, weight = 0.2) 
reverse_word_index[3]
'''
