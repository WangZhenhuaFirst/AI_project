# -*- coding: utf-8 -*-
import transformers
import torch
import os
import time
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
# from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
# from train import create_model
import torch.nn.functional as F


class MyDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)


"""参考：https://github.com/yangjianxin1/GPT2-chitchat"""


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 使用top-k和/或nucleus (top-p)过滤对logit的分布 进行过滤
        参数:
            logits: 分布形态(词汇量)
            top_k > 0: 只保留具有最高概率的前k个token(top-k过滤).
            top_p > 0.0: 保留具有累积概率的顶级token>= top_p(核过滤).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # 批量大小1; 现在可以更新更多，但代码将不太清晰
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # 删除所有概率小于topk的最后一个token的token
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        # print('indices_to_remove: ', indices_to_remove)
        # 对于topk之外的其他元素的logits值设为负无穷
        logits[indices_to_remove] = filter_value
        # print('logits_after_remove_indices: ', logits)

    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # 删除累积概率高于阈值的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，以使第一个令牌也保持在阈值之上
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# GPT2生成回复
class Gpt2Generate:
    # 定义历史对话记录history全局变量
    global history

    def __init__(self):
        # 最大历史记录长度
        self.max_history_len = 5
        # token的最大长度
        self.max_len = 25
        # 重复惩罚项系数,repetition_penalty:Between 1.0 and infinity, 1.0 means no penalty
        self.repetition_penalty = 1.2
        self.temperature = 1
        # 只保留具有最高概率的前k个token
        self.topk = 8
        # 保留具有累积概率的顶级token>= top_p(核过滤)
        self.topp = 0
        # 定义初始化对话历史为空list
        self.history = []
        self.model_path = r'../Flask/app/static/model04/gpt2/dialogue_model'
        self.vocab_path = r'../Flask/app/static/model04/gpt2/vocab_small.txt'
        torch.cuda.empty_cache()
        # 判断是否有gpu
        use_cuda = torch.cuda.is_available()
        # 自动选择推理设备为GPU或CPU
        self.device = 'cuda' if use_cuda else 'cpu'
        # 加载预训练模型tokenizer (vocabulary)
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path)
        # 加载预训练模型 (weights)
        # GPT2LMHeadModel: The GPT2 Model transformer with a language modeling head on top
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        # 将模型设置为evaluation模式，关闭DropOut模块
        self.model.eval()

    # 生成response的方法
    def generate(self, text):
        global history
        # 定义response为空字符串
        response = ''
        # 判断对话历史长度超过5，则取最后max_history_len句，减少对话历史增长带来的内存负荷
        # print(len(self.history))
        if len(self.history) > 5:
            self.history = self.history[-self.max_history_len:]
        # print(len(self.history))
        self.history.append(self.tokenizer.encode(text))
        # print(len(self.history))
        # 每个input以[CLS]为开头
        input_ids = [self.tokenizer.cls_token_id]

        # 这里是否应该直接用self.history ??? 因为前面已经去掉超过5句的部分了 ？？？
        for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
            # 说明self.history中存的是tokenize之后的？？？在哪里转换的？
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)  # 每个句子间，加个分隔符[SEP]
        curr_input_tensor = torch.tensor(input_ids).long().to(self.device)
        generated = []
        # 最多生成max_len个token，即每次生成一个词，而不是一个句子
        for _ in range(self.max_len):
            outputs = self.model(input_ids=curr_input_tensor)
            # print('outputs: ', outputs)
            # 参考：https://huggingface.co/transformers/main_classes/output.html
            next_token_logits = outputs[0][-1, :]
            # print('next_token_logits: ', next_token_logits)
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(generated):
                next_token_logits[id] /= self.repetition_penalty
            next_token_logits = next_token_logits / self.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[self.tokenizer.convert_tokens_to_ids(
                '[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=self.topk, top_p=self.topp)
            # torch.multinomial表示从候选集合中无放回地抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 遇到[SEP]则表明response生成结束
            if next_token == self.tokenizer.sep_token_id:
                break
            print('next_token: ', next_token)
            print('next_token.item:', next_token.item())
            generated.append(next_token.item())
            curr_input_tensor = torch.cat(
                (curr_input_tensor, next_token), dim=0)
        # print(len(generated))
        self.history.append(generated)
        response = "".join(self.tokenizer.convert_ids_to_tokens(generated))
        return response


if __name__ == '__main__':
    gpt = Gpt2Generate()
    s = time.time()
    gpt_answer = gpt.generate('可以微信扫码么？')
    print('gpt_answer', gpt_answer)
    e = time.time()
    print(e-s)

    # # 下面是运行交互式对话
    # while True:
    #     print('机器人: '+ gpt.generate(input("我: ")))
