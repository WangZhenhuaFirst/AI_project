# -*- coding: utf-8 -*-
import requests
from lxml import etree
import re
from urllib import parse
from functools import lru_cache
#from .editDistance import edit_distance
import urllib.request
from app.robot_get_embedding import build_tokenizer, build_model, extract_emb_feature
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
import os


class BaiduSpider:
    '''根据爬虫生成结果，爬取目标为百度百科和百度知道，bdbk=百度百科；bdzd=百度知道'''

    def __init__(self):
        app = Flask(__name__)

        # 加载百度百科headers
        self.bdbk_headers = {
            'cookie': 'BAIDUID=AD1678FE9F99BCA3AA5D44D0951FE231:FG=1; shitong_key_id=2; Hm_lvt_6859ce5aaf00fb00387e6434e4fcc925=1590813354,1590815838,1590816542; ZD_ENTRY=empty; Hm_lpvt_6859ce5aaf00fb00387e6434e4fcc925=1590816869; shitong_data=9319c8497b6fb5f4ac719eb8aaa9f4009e557e3e2a78386f5df3100036141a96537211c62d3ec2dea2b829c21d570751ed93980b14ca80093ed1e5b85f7b985bb632923001036da9e0ef116c967ea7f8508e93691058343736b3e422e555653e14e34f35b546c1bcfa0ab322dbd46b26f7a93efadc61c820d76ae0475bc8ba16; shitong_sign=6496e316',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,la;q=0.6',
            'Host': 'baike.baidu.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        }
        # 加载百度知道headers
        self.bdzd_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'cookie': 'BAIDUID=AD1678FE9F99BCA3AA5D44D0951FE231:FG=1; shitong_key_id=2; Hm_lvt_6859ce5aaf00fb00387e6434e4fcc925=1590813354,1590815838,1590816542; ZD_ENTRY=empty; Hm_lpvt_6859ce5aaf00fb00387e6434e4fcc925=1590816869; shitong_data=9319c8497b6fb5f4ac719eb8aaa9f4009e557e3e2a78386f5df3100036141a96537211c62d3ec2dea2b829c21d570751ed93980b14ca80093ed1e5b85f7b985bb632923001036da9e0ef116c967ea7f8508e93691058343736b3e422e555653e14e34f35b546c1bcfa0ab322dbd46b26f7a93efadc61c820d76ae0475bc8ba16; shitong_sign=6496e316',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,la;q=0.6',
            'Host': 'zhidao.baidu.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        }
        # 定义余弦相似度阈值
        self.threshold = 0.985
        # 设置bert文件路径
        self.dict_path = r"model04/search_database/chinese_L-12_H-768_A-12/vocab.txt"
        self.config_path = r"model04/search_database/chinese_L-12_H-768_A-12/bert_config.json"
        self.checkpoint_path = r"model04/search_database/chinese_L-12_H-768_A-12/bert_model.ckpt"
        # 加载tokenizer
        self.tokenizer = build_tokenizer(
            os.path.join(app.static_folder, self.dict_path))
        # 加载bert模型
        self.model = build_model(os.path.join(app.static_folder, self.config_path), os.path.join(
            app.static_folder, self.checkpoint_path))

    # 使用requests获取网页源码
    @staticmethod
    def get_page(url, headers, encode):
        try:
            response = requests.get(url, headers=headers)
            # print(response.text)
            if response.status_code == 200:
                response.encoding = encode
                return response.text
            else:
                print('Request failed', response.status_code)
        except requests.exceptions.ConnectionError as e:
            print('Error', e.args)

    @staticmethod
    def bdbk_extract(html):
        patterns = [r'[\n\xa0]', r'<div.*?>', r'<i>',
                    r'<a.*?>', r'<sup.*?/sup>', r'</.*?>']
        for pattern in patterns:
            html = re.sub(pattern, '', html)
        return html

    # 解析百度百科网页
    @staticmethod
    def bdbk_parser(html):
        selector = etree.HTML(html)
        description = selector.xpath('//div[@class="lemma-summary"]/div')
        description = [etree.tostring(
            d, encoding='utf-8', method='html').decode('utf-8') for d in description]
        return description

    # 百度百科搜索
    def search_bdbk(self, inputQ):
        html = self.get_page('https://baike.baidu.com/item/' +
                             inputQ, headers=self.bdbk_headers, encode='utf-8')
        description = self.bdbk_parser(html)
        description = [self.bdbk_extract(d) for d in description]
        return description[0] if description else []

    @staticmethod
    def make_url(inputQ):
        # 百度知道基础页
        base_url = 'https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word='
        # 把用户输入的问题，转成gb2312编码格式
        inputQ_convert = parse.quote(inputQ, encoding='gb2312')
        # 返回的网页为基础页+已转化的问题字符串
        return base_url + inputQ_convert

    @staticmethod
    def bdzd_parser(html):
        # 解析百度知道第一页数据，提取问题，链接，答案
        selector = etree.HTML(html)
        questions = selector.xpath('//div[@class="list-inner"]/div/dl/dt/a')
        links = selector.xpath('//div[@class="list-inner"]/div/dl/dt/a/@href')
        questions = [(q.xpath('string(.)'), link)
                     for q, link in zip(questions, links)]
        return questions

    # 提取最终答案页中的最好的答案
    @staticmethod
    def inner_parser(html):
        selector = etree.HTML(html)
        # 假如网页中有class="best-text mb-10"则有最好的答案可以提取，如果没有，返回空数据
        best_answer = selector.xpath('//div[@class="best-text mb-10"]')
        # best_answer = selector.xpath('//div[@class="answer-text mb-10 line"]')
        best_answer = [a.xpath('string(.)') for a in best_answer]
        best_answer = ''.join(best_answer).strip()
        best_answer = re.sub(r'[\n\xa0]', '', best_answer)
        best_answer = re.sub(r'展开全部', '', best_answer)
        best_answer = re.sub(r'\s+', ' ', best_answer)
        return best_answer

    def search_bdzd(self, inputQ, print_value=False):
        # 获取百度知道第一页的10个问题和答案页
        html = self.get_page(self.make_url(
            inputQ), headers=self.bdzd_headers, encode='gb2312')
        # 第一页中网页对应的问题，链接，和答案提取出来
        questions = self.bdzd_parser(html)
        if questions:
            # 把用户输入的问题转换成向量，再传进去计算余弦相似度，减少多次转换向量的时间
            inputQ_vec = extract_emb_feature(
                self.model, self.tokenizer, [inputQ], max_len=30)[0]
            # best_answer包含  百度知道问题字符串，问题答案对应的网页链接，和用户输入的问题与百度知道问题的余弦相似度
            best_answer = sorted([(q, link, self.get_cosine_similarity(
                inputQ_vec, q)) for (q, link) in questions], key=lambda x: x[2])[-1]
            # 判断余弦相似度是否大于阈值
            if best_answer[2] > self.threshold:
                if print_value:
                    print(best_answer[0], best_answer[2])
                # 获取相似度最高的网页
                best_answer_html = self.get_page(
                    best_answer[1], headers=self.bdzd_headers, encode='gb2312')
                # 从获取的网页中提取被用户评价为最好回答的答案，如果没有最好的答案，返回空list
                best_answer = self.inner_parser(best_answer_html)
                return best_answer
        return []

    # 计算余弦相似度，q1为用户输入的问题向量，q2为需要匹配的问题字符串
    def get_cosine_similarity(self, q1, q2):
        x = q1.reshape(1, -1)
        y = extract_emb_feature(self.model, self.tokenizer, [
                                q2], max_len=30)[0].reshape(1, -1)
        return cosine_similarity(x, y)

    # 搜索答案
    def search_answer(self, inputQ):
        # 优先从百度百科搜索
        bdbk_result = self.search_bdbk(inputQ)
        # 百度百科中没有答案，则从百度知道搜索
        return (bdbk_result, '来源: 百度百科') if bdbk_result else (self.search_bdzd(inputQ), '来源: 百度知道')
