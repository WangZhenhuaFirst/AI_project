# -*- coding: utf-8 -*-

import time
from search_database import SearchDataBase
from baidu_spider import BaiduSpider
from gpt2_generate import Gpt2Generate, MyDataset, top_k_top_p_filtering


# 汇总所有问答类
class Response:
    def __init__(self):
        self.SDB = SearchDataBase()
        self.BS = BaiduSpider()
        self.GPT2G = Gpt2Generate()

    def generate(self, inputQ):
        # 基于SearchDataBase搜索答案，响应时间 0.12 秒
        sdb_answer = self.SDB.annoy_search(inputQ, print_value=False)
        if sdb_answer:
            return (sdb_answer, '来源: 数据库')
        # 基于BaiduSpider搜索答案，响应时间比较慢，2.5秒
        bs_answer = self.BS.search_answer(inputQ)
        if bs_answer[0]:
            return bs_answer
        # 基于Gpt2Generate搜索答案，响应时间 0.14 秒
        gpt2_answer = self.GPT2G.generate(inputQ)
        return (gpt2_answer, '来源: GPT2')


if __name__ == '__main__':
    response = Response()

    s = time.time()
    res = response.generate('你好')
    e = time.time()
    print(res, e-s)

    while True:
        print('机器人: ' + str(response.generate(input("我:   "))))
