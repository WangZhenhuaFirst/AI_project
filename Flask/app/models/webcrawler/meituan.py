# -*- coding: utf-8 -*-
from app.models.webcrawler.agent import Agent
from app.models.rating import Rating
import requests
from fake_useragent import UserAgent
import random
import time
from datetime import datetime
import base64
import zlib
import uuid


class MeiTuan(object):
    '''美团数据爬取'''

    # 代理列表
    myagents = []
    header_tag = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Host': 'gz.meituan.com',
        'Referer': 'https://gz.meituan.com/meishi/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
    }

    headers_comment = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"
    }

    citylist = {'北京': 'bj', '上海': 'sh', '广州': 'gz', '深圳': 'sz', '天津': 'tj',
                '西安': 'xa', '重庆': 'cq', '杭州': 'hz', '南京': 'nj', '武汉': 'wh', '成都': 'cd'}

    typelist = {'代金券': 'c393', '蛋糕甜点': 'c11', '火锅': 'c17', '自助餐': 'c40', '小吃快餐': 'c36', '日韩料理': 'c28',
                '西餐': 'c35', '聚餐宴请': 'c395', '烧烤烤肉': 'c54', '东北菜': 'c20003', '川湘菜': 'c55', '江浙菜': 'c56',
                '香锅烤鱼': 'c20004', '粤菜': 'c57', '中式烧烤/烤串': 'c400', '西北菜': 'c58', '咖啡酒吧': 'c41', '京菜鲁菜': 'c59',
                '云贵菜': 'c60', '东南亚菜': 'c62', '海鲜': 'c63', '素食': 'c217', '台湾/客家菜': 'c227', '创意菜': 'c228',
                '汤/粥/炖菜': 'c229', '蒙餐': 'c232', '新疆菜': 'c233', '其他美食': 'c24'}

    @staticmethod
    def Init():
        Agent.Init()
        print('meituan - init')
        pass

    @staticmethod
    def Process():
        cityname, citycode = random.choice(list(MeiTuan.citylist.items()))
        foodname, foodtype = random.choice(list(MeiTuan.typelist.items()))

        url = 'https://{}.meituan.com/meishi/{}/'

        url = url.format(citycode, foodtype)
        print('meituan - rest url : {}'.format(url))

        originUrl = MeiTuan.str_replace(url)

        # 生成token
        token_encode = MeiTuan.encode_token(url)
        token = MeiTuan.str_replace(token_encode)

        MeiTuan.header_tag['Host'] = '{}.meituan.com'.format(citycode)
        MeiTuan.header_tag['Referer'] = 'https://{}.meituan.com/meishi/'.format(
            citycode)
        guid = str(uuid.uuid4())
        url = 'https://{}.meituan.com/meishi/api/poi/getPoiList?cityName={}&cateId={}&areaId=0&sort=&dinnerCountAttrId=&page=1&userId=&uuid={}&platform=1&partner=126&originUrl={}&riskLevel=1&optimusCode=20&_token={}'
        url = url.format(citycode, cityname,
                         foodtype.replace('c', ''), guid, originUrl, token)
        print('url: ', url)

        response = requests.get(url, headers=MeiTuan.header_tag,
                                proxies=Agent.GetAgent())

        if response.status_code == 200 and 'verify' not in response.url:
            json = response.json()
            print('json:', json)

            if 'data' in json and json['data']["poiInfos"] != None:
                for item in json['data']["poiInfos"]:
                    id = item['poiId']  # 店铺Id
                    title = item['title']  # 店名
                    print('meituan开始执行get_ratings')
                    # 获取评论
                    MeiTuan.get_ratings(id, title)

        print('meituan - finished!')

    @staticmethod
    def get_ratings(id, title):
        # 评论地址
        ratingurl = 'https://www.meituan.com/meishi/{}/'.format(id)
        guid = str(uuid.uuid4())
        for num in range(0, 381, 10):
            commentexist = False  # 默认评论不存在
            print("meituan - 正在爬取%s条............" % num)
            ajax_url = "https://www.meituan.com/meishi/api/poi/getMerchantComment?uuid={}&platform=1&partner=126&originUrl=https%3A%2F%2Fwww.meituan.com%2Fmeishi%2F{}%2F&riskLevel=1&optimusCode=10&id={}&userId=&offset=" + \
                str(num) + "&pageSize=10&sortType=1"
            ajax_url = ajax_url.format(id, guid, id)
            print('meituan - comment url : {}'.format(ajax_url))
            print('agent: ', Agent.GetAgent())
            response = requests.get(url=ajax_url, headers=MeiTuan.headers_comment,
                                    proxies=Agent.GetAgent())
            if response.status_code == 200 and 'verify' not in response.url:

                json = response.json()
                if 'data' in json and json['data']["comments"] != None:
                    for item in json['data']["comments"]:
                        reviewId = item["reviewId"]  # 评论ID
                        user_id = item["userId"]  # 用户Id
                        name = item["userName"]  # 用户名
                        comment = item["comment"]  # 评论
                        star = float(item["star"]) / 10  # 美团50分满分
                        timestamp = int(item["commentTime"])  # 评论时间戳
                        print('meituan开始执行insert')
                        rating = Rating(reviewId, user_id, name, id, title,
                                        star, comment, ratingurl, timestamp, '美团')
                        commentexist = rating.insert()

                        # 评论已存在
                        if commentexist:
                            break
                else:  # 无数据
                    break
            else:  # 代理被封
                break

            # 评论已存在
            if commentexist:
                break
            MeiTuan.sleep()

    @staticmethod
    def str_replace(string):
        return string.replace('/', '%2F') \
            .replace('+', '%2B') \
            .replace('=', '%3D') \
            .replace(':', '%3A')

    @staticmethod
    def encode_token(url):
        ts = int(datetime.now().timestamp() * 1000)
        token_dict = {
            'rId': 100900,
            'ver': '1.0.6',
            'ts': ts,
            'cts': ts + 100 * 1000,
            'brVD': [1010, 750],
            'brR': [[1920, 1080], [1920, 1040], 24, 24],
            'bI': ['https://gz.meituan.com/meishi/c11/', ''],
            'mT': [],
            'kT': [],
            'aT': [],
            'tT': [],
            'aM': '',
            'sign': 'eJwdjktOwzAQhu/ShXeJ4zYNKpIXqKtKFTsOMLUn6Yj4ofG4UjkM10CsOE3vgWH36df/2gAjnLwdlAPBBsYoR3J/hYD28f3z+PpUnmJEPqYa5UWEm0mlLBRqOSaP1qjEtFB849VeRXJ51nr56AOSVIi9S0E3LlfSzhitMix/mQwsrdWa7aTyCjInDk1mKu9nvOHauCQWq2rB/8laqd3cX+adv0zdzm3nbjTOdzCi69A/HQAHOOyHafMLmEtKXg=='
        }
        token_dict['bI'] = [url, '']
        # 二进制编码
        encode = str(token_dict).encode()
        # 二进制压缩
        compress = zlib.compress(encode)
        # base64编码
        b_encode = base64.b64encode(compress)
        # 转为字符串
        token = str(b_encode, encoding='utf-8')
        return token

    @staticmethod
    def sleep():
        sec = 1 + 10 * random.random()
        time.sleep(sec)
        print('sleep {}'.format(sec))
