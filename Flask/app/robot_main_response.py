# -*- coding: utf-8 -*-
import time
from app.robot_search_database import SearchDataBase
from app.robot_baidu_spider import BaiduSpider
from app.robot_gpt2_generate import Gpt2Generate, MyDataset, top_k_top_p_filtering


# 汇总所有问答类
class Response:

    @staticmethod
    def init():
        Response.SDB = SearchDataBase()
        Response.BS = BaiduSpider()
        Response.GPT2G = Gpt2Generate()

    @staticmethod
    def generate(inputQ):
        # 基于SearchDataBase搜索答案，响应时间 0.12 秒
        # print('inputQ: ', inputQ)
        sdb_answer = Response.SDB.annoy_search(inputQ, print_value=False)
        if sdb_answer:
            return (sdb_answer, '来源: 数据库')
        # 基于BaiduSpider搜索答案，响应时间比较慢，2.5秒
        bs_answer = Response.BS.search_answer(inputQ)
        if bs_answer[0]:
            return bs_answer
        # 基于Gpt2Generate搜索答案，响应时间 0.14 秒
        gpt2_answer = Response.GPT2G.generate(inputQ)
        return (gpt2_answer, '来源: GPT2')
