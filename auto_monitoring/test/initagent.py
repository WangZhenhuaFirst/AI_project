import pymongo
import requests
from fake_useragent import UserAgent
import re
import random
import time

client = pymongo.MongoClient(
    "mongodb://wangzhenhua:wstc9093@localhost")  # defaults to port 27017

db = client.auto_monitoring


def sleep():
    sec = 1 + 3 * random.random()
    time.sleep(sec)
    print('sleep {}'.format(sec))


def getAgent():
    ips = []  # 装载有效 IP
    for i in range(1, 79):
        headers = {
            "User-Agent": UserAgent().chrome  # chrome浏览器随机代理
        }
        ip_url = 'http://www.89ip.cn/index_{}.html'.format(i)
        html = requests.get(url=ip_url, headers=headers).text
        res_re = html.replace(" ", "").replace("\n", "").replace("\t", "")
        # 使用正则表达式匹配出IP地址及端口
        r = re.compile('<tr><td>(.*?)</td><td>(.*?)</td><td>')
        result = re.findall(r, res_re)
        for l in range(len(result)):
            ip = "http://" + result[l][0] + ":" + result[l][1]
            # 设置为字典格式
            proxies = {"http": ip}
            # 使用上面的IP代理 请求百度，成功后状态码200
            baidu = requests.get("https://www.baidu.com/", proxies=proxies)
            if baidu.status_code == 200:
                ips.append(proxies)
                db.Agents.insert({'Agent': ip})
                print('added {} {}'.format(i, ip))
            # sleep()
        print("正在准备IP代理，请稍后。。。")

    return ips


getAgent()
print('Agent init finished')

agents = list(db.Agents.find())

print('测试-随机获取一个代理IP')
print(random.randint(0, len(agents)-1))
print(agents[random.randint(0, len(agents)-1)]['Agent'])
