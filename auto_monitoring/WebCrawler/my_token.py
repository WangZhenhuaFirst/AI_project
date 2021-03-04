# -*- coding: utf-8 -*-
from datetime import datetime
import base64
import zlib
import requests
import json
import re
import random
import time
import csv
from fake_useragent import UserAgent

agents = []
header_tag = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Host': 'gz.meituan.com',
    'Referer': 'https://gz.meituan.com/meishi/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
}

headers_keyword = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Host': 'apimobile.meituan.com',
    'Referer': 'https://xa.meituan.com/',
    'Referer': 'https://xa.meituan.com/s/%E9%BA%BB%E8%BE%A3/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
}

headers_comment = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"
}


def decode_token(token):
    # base64解码
    token_decode = base64.b64decode(token.encode())
    # 二进制解压
    token_string = zlib.decompress(token_decode)
    return token_string

# 生成token


def encode_token():
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
    # 二进制编码
    encode = str(token_dict).encode()
    # 二进制压缩
    compress = zlib.compress(encode)
    # base64编码
    b_encode = base64.b64encode(compress)
    # 转为字符串
    token = str(b_encode, encoding='utf-8')
    return token


def str_replace(string):
    return string.replace('/', '%2F') \
        .replace('+', '%2B') \
        .replace('=', '%3D') \
        .replace(':', '%3A')


def getAgent():
    ips = []  # 装载有效 IP
    for i in range(1, 2):
        headers = {
            "User-Agent": UserAgent().chrome  # chrome浏览器随机代理
        }
        ip_url = 'http://www.89ip.cn/index_{}.html'.format(i)
        html = requests.get(url=ip_url, headers=headers).text
        res_re = html.replace(" ", "").replace("\n", "").replace("\t", "")
        # 使用正则表达式匹配出IP地址及端口
        r = re.compile('<tr><td>(.*?)</td><td>(.*?)</td><td>')
        result = re.findall(r, res_re)
        for i in range(len(result)):
            ip = "http://" + result[i][0] + ":" + result[i][1]
            # 设置为字典格式
            proxies = {"http": ip}
            # 使用上面的IP代理请求百度，成功后状态码200
            baidu = requests.get("https://www.baidu.com/", proxies=proxies)
            if baidu.status_code == 200:
                ips.append(proxies)
        print("正在准备IP代理，请稍后。。。")

    return ips


def get_ratings(id, title):
    # 创建文件夹并打开
    fp = open("美团评论.csv", 'a',
              newline='', encoding='utf-8-sig')
    writer = csv.writer(fp)  # 我要写入
    # 写入内容
    # writer.writerow(("userName", "userId", "userUrl", "comment"))  # 运行一次

    for num in range(0, 381, 10):
        print("正在爬取%s条............" % num)
        ajax_url = "https://www.meituan.com/meishi/api/poi/getMerchantComment?uuid=60d8293a-8f06-4d5e-b13d-796bbda5268f&platform=1&partner=126&originUrl=https%3A%2F%2Fwww.meituan.com%2Fmeishi%2F{}%2F&riskLevel=1&optimusCode=10&id={}&userId=&offset=" + \
            str(num) + "&pageSize=10&sortType=1"
        ajax_url = ajax_url.format(id, id)
        print(ajax_url)
        reponse = requests.get(url=ajax_url, headers=headers_comment,
                               proxies=agents[random.randint(0, len(agents)-1)])
        json = reponse.json()
        if 'data' in json and json['data']["comments"] != None:
            for item in json['data']["comments"]:
                name = item["userName"]
                user_id = item["userId"]
                user_url = item["userUrl"]
                comment = item["comment"]
                result = (id, title, name, user_id, user_url, comment)
                writer.writerow(result)
        sleep()
    fp.close()


def sleep():
    sec = 1 + 15 * random.random()
    time.sleep(sec)
    print('sleep {}'.format(sec))


if __name__ == '__main__':
    token = [
        'eJxVjstuqzAURf/F06LYxkAgUgeQ0MvzkoQ8QFUHbngnJgGcpKG6/35dqR1UOtLeZ501OJ+gdzMwwwgZCEnglvdgBvAETTQgAT6Ii6qqsqxPsUIMIoHDb6bIhgTe+90CzF4xwkiaqujti6wFeMWGjCSMdIF+uiK6rIj5slwhgYrzyzCDsBwnLK/5lbaTw5lB0YeqhgeMofgECJ1thC7y+J30O/nPHorXhTvUZSta7t2z5sij+2iuqiuMqyT3lDH961/cpPO5/7IZojDYtlraKOfij7JtjiFG8yGyya3cO0TLCiiXZtMG9+xkLi1rSM9r4sEqXch6Qcan5WXbMs9edilVt3ubIXYKrHUXxXSJu8bmL5auGLt8nXgqbntVM6N459ZGjGwSnIp4rGoe1h+Qre5Dn+3plG4e88ZtF0fM/KvR3iKHXuerfSf3FtRPtMvIIXmi2Q2N2chI+95somyc15phQmdlOlH0cGgRBszmflI+P4N//wEWi44a',
        'eJxVjstuozAUht/F26LYBkxDpC4gocN1SEIuoGoWbsw1MQngJC1V372u1C5GOtJ/Od/i/wC9x8AMI2QipIBb3oMZwBM0MYACxCA/hBBVQwgjYmIFHP7vDGIq4LXfLcDsBcusPBL077tZy+IFmypSMJrK6tfr0qu6vG/KkxCohLgMMwjLccLzWlxpOzmcOZR+qGp4wBjKJUDifCNxqccfpT8qfnMkp0t2qMtWuty/s+Yo4vtoraorTKo09/Ux+xtcvLQLRPC8GeIo3LZG1ujn4o++bY4RRvMhdrRbuXc1gxVQLa2mDe/sZC1te8jOa82HVbZQp4U2Piwv25b7zrLLKNnuHY74KbTXXZzQJe4aRzzbU93c5evUJ7jtiWHFyc6rzQQ5WngqkrGqRVS/Qb66Dz3b00e6eZ83Xrs4Yh5czfYWu/Q6X+07tbfh9EQ7ph3SB8puaGQj19rXZhOzcV4bpgXdleXG8btLiyjkjgjS8ukJfH4B4qqN+w==',
        'eJxdjktvozAURv+Lt0WxjYFCpC4gocNzSEIeoGoWbswzMQngJFNG89/HldrNSFf6vnvuWdw/YPAZmGOELIQUcC8GMAd4hmYGUIAY5UXXdZUg3USEmAo4/scsSwHvw34J5m8YYaQ86+jXJ9lI8IYtFSkYmRJ9d012VZPzaflSArUQ13EOYTXNeNGIG+1mxwuHso91A48YQ/kJkDrfSl3m6SvpV4rvPZavS3dsqk62Iniw9iSSx2Sv6xtM66wItCn/GV79rA9F+LodkzjadUbeapfyh7ZrTzFGizFxyb06eMRgJVQru+2iBzvbK8cZ88uGBLDOl6pZkulpdd11PHBXfU713cHliJ8jZ9MnKV3hvnXFq2Nq1r7YZIGOu0E37CTd+42VIpdE5zKd6kbEzW/I149xYAf6TLcfi9bvlifMw5vV3ROP3hbrQ68ODjTPtGfkmD1RdkcTmzjp3tttwqZFY1g29Na2lyQfHi3jiLsizKqXF/D3Hwp7jhM=',
        'eJxdjktvozAURv+Lt0WxjYGESF1AQofnkIQ8QFUXbngndgI4yZTR/PdxpXZT6Urfd889i/sX9F4O5hghEyEF3IsezAGeoIkBFCAGedF1XSUIY4OougKOP5gxVcB7v1+C+StGGClTHb19ko0Er9hUkYLRTKLvrsmuanI+LU9KoBbiOswhrMYJKxpxo3xyvDAo+1A38IgxlJ8AqbOt1GWevpJ+pfjeI/m6dIem4rIV/iNvTyJ+jNa6vsGkTgtfG7PfwdVLu0AEL9shjsIdN7JWu5S/tF17ijBaDLFD7tXBJUZeQrWyWh4+8rO1su0hu2yID+tsqc5KMj6trjvOfGfVZVTfHRyG2Dm0N12c0BXuWke82DPN3Beb1Ncx73XDipO915gJckh4LpOxbkTU/IFs/Rj6/ECndPuxaD2+PGEW3Ex+j116W6wPndrbcHamXU6O6RPN72jMR0b4e7uN83HRGKYF3bXlxvGHS8soZI4I0ur5Gfz7D+r3jgA=',
        'eJxVjk1zqjAUhv9LtjKGoJjgjItqhYJVBI2od+4CNPJh+SgkonT635tOexd35sy873nOszgfoLbPYIxU1VBVBdxYDcYA9dX+CCiAN/KiE4wxIUhX8UABp/8ZwkQBUb17BuM/SCOqIrW/38SX4IdgTZKfOiKyakM5344tFZBwXjVjCOOun7OUi7Don8ocyt4kKTwhBOUfQOr5Vuoyr78Z/ib/ty/l49Jt0riQjTntOaPove2evIT1Nsk+dF/d+dAOp+ZjVnqNaKb3W0StwHGOmbfg9qVa0Wq/pgOhhlZtQHO4vvM4WTY0yNxZaTvdMd3qYv4yJ+tav+wRzA9vGy/DRS68a5uUD+YHC37sumRpISdbr95W+nC2Zz73H7zY9gZzen2vXO1QIirOeJGdjLBZWtxJ6K6IoWsEPvMMcvejMHoI196avWATHBbV9EJgOKLVyxMuA5G2VlSyvMXkVppCe971yGtWHxjcDWA7vbWm08WTCfj8As6yj6M='
    ]

    for i in range(0, len(token)):
        token1 = decode_token(token[i])
        print(token1)
    agents = getAgent()

    # 按照美团默认视频标签搜索
    # c11代表蛋糕甜点
    # gz代表广州
    cityname = '广州'
    originUrl = str_replace('https://gz.meituan.com/meishi/c11/')
    # 生成token
    token_encode = encode_token()
    token = str_replace(token_encode)
    url = 'https://gz.meituan.com/meishi/api/poi/getPoiList?cityName={}&cateId=11&areaId=0&sort=&dinnerCountAttrId=&page=1&userId=&uuid=88e10a4e-b46e-4a27-a4b5-2b2e56a72b46&platform=1&partner=126&originUrl={}&riskLevel=1&optimusCode=10&_token={}'
    url = url.format(cityname, originUrl, token)
    response = requests.get(url, headers=header_tag,
                            proxies=agents[random.randint(0, len(agents)-1)])
    if response.status_code == 200 and 'verify' not in response.url:
        data = response.json()['data']
        with open('data.json', 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False)
            print('Save data into json file successfully!')
            f.close()
    sleep()

    # 按照美团自带搜索关键字
    keyword = '麻辣'
    url = 'https://apimobile.meituan.com/group/v4/poi/pcsearch/10?uuid=88e10a4e-b46e-4a27-a4b5-2b2e56a72b46&userid=-1&limit=32&offset=0&cateId=-1&q={}&sort=default'
    url = url.format(keyword)
    response = requests.get(url, headers=headers_keyword,
                            proxies=agents[random.randint(0, len(agents)-1)])
    if response.status_code == 200 and 'verify' not in response.url:
        print()
        data = response.json()['data']
        with open('data1.json', 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False)
            print('Save data into json file successfully!')
            f.close()
        fp = open("美团店铺.csv", 'a',
                  newline='', encoding='utf-8-sig')
        writer = csv.writer(fp)  # 我要写入
        # 写入内容
        # writer.writerow(("ID", "title", "address"))  # 运行一次

        for item in data["searchResult"]:
            id = item["id"]
            title = item["title"]
            address = item["address"]
            result = (id, title, address)
            writer.writerow(result)

            # 获取评论
            get_ratings(id, title)

        fp.close()

    print('finished')
