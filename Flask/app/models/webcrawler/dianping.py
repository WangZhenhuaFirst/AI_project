# -*- coding: utf-8 -*-
import requests
from selectolax.parser import HTMLParser
import re
import time
import datetime
import random
from app.database import DB
from app.models.rating import Rating
from app.models.webcrawler.agent import Agent

# Fiddle4


class DianPing(object):
    r_sessoion = requests.session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362",
        "Cookie": "t_lxid=17195a3e54dc8-04bfd6b23b432f-71415a3b-e1000-17195a3e54dc8-tid; _hc.v=a98b9609-20a3-0a9c-bc73-e1fb38989b55.1587353478; _lxsdk_cuid=17195a3da3d0-0429c65bee7b53-71415a3b-e1000-17195a3da3e47; _lxsdk=17195a3da3d0-0429c65bee7b53-71415a3b-e1000-17195a3da3e47; _lxsdk_s=1719d96d20c-267-33e-4bd%7C%7C42",
        "Connection": "Keep-Alive",
        "Host": "www.dianping.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate"
    }
    headers2 = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362",
        "Host": "s3plus.meituan.net",
    }

    @staticmethod
    def Init():
        Agent.Init()
        print('dianping - init')
        pass

    @staticmethod
    def get_initial_num():
        a = DB.find_one("dianping_register", {"crawby": "jack"})
        b = DB.find_one("Ratings", {"source": "dazhongdianping"})
        if a:
            initial_num = a['initial_num']
        elif b:
            c = DB.find_max(
                'Ratings', {"source": "dazhongdianping"}, 'timestamp')
            if c:
                initial_num = b['reviewId']
            else:
                initial_num = 706754427
        else:
            initial_num = 706754427
        return initial_num

    ### 获得CSS文件内容 ###
    @staticmethod
    def get_css_content(html, headers, r_sessoion):
        print('------begin to get css content------')
        try:
            # with open(r'c:\temp\abcd.txt','w',encoding='utf8') as f:
            #     f.write(html)
            css_l = re.search(
                r'<link rel="stylesheet" type="text/css" href="(//s3plus.meituan.net.*?.css)">', html)
            css_link = 'http:' + css_l.group(1)
            # print('css_link',css_link)
            html_css = requests.get(css_link, headers=DianPing.headers2).text
            return html_css
        except:
            return 'error'

    ### 获得字典 ###
    @staticmethod
    def get_font_dic(css_content):
        print('------begin to get font dictionary------')
        # 获取svg链接和svg页面的html源码
        svg_l = re.search(
            r'svgmtsi.*?(//s3plus.meituan.net/v1.*?svg)\);', css_content)
        svg_link = 'http:' + svg_l.group(1)
        svg_html = DianPing.r_sessoion.get(svg_link).text

        # 解析出字典
        y_list = re.findall('d="M0 (.*?) H600"', svg_html)  # y_list的元素为str
        if not y_list:
            y_list = re.findall('y="(.*?)">', svg_html)  # y_list的元素为str
        font_dic = {}
        j = 0    # j为第j行
        font_size = int(re.findall(
            r'font-size:(.*?)px;fill:#333;}', svg_html)[0])
        for y in y_list:
            font_l = re.findall(r'<textPath xlink:href="#' + str(j + 1) +
                                '" textLength=".*?">(.*?)</textPath>', svg_html)
            if not font_l:
                font_l = re.findall(r'<text x="0" y="' +
                                    str(y) + '">(.*?)</text>', svg_html)
            font_list = re.findall(r'.{1}', font_l[0])
            for x in range(len(font_list)):    # x为每一行第x个字
                font_dic[str(x * font_size) + ',' + y] = font_list[x]
            j += 1
        return font_dic, y_list

    ### 把svg标签替换成文字 ###
    @staticmethod
    def get_html_full_review(html, css_content, font_dic, y_list):
        font_key_list = re.findall(r'<svgmtsi class="(.*?)"></svgmtsi>', html)
        for font_key in font_key_list:
            pos_key = re.findall(
                r'.' + font_key + '{background:-(.*?).0px -(.*?).0px;}', css_content)
            pos_x = pos_key[0][0]
            pos_y_original = pos_key[0][1]
            for y in y_list:
                if int(pos_y_original) < int(y):
                    pos_y = y
                    break
            html = html.replace('<svgmtsi class="' + font_key +
                                '"></svgmtsi>', font_dic[pos_x + ',' + pos_y])
        return html

    @staticmethod
    def get_html(url, headers):
        return DianPing.r_sessoion.get(url, headers=DianPing.headers)

    @staticmethod
    def Process():
        initial_num = DianPing.get_initial_num()
        results = [['id', 'userid', 'restid', 'rating', 'rating_env',
                    'rating_flavor', 'rating_service', 'timestamp', 'comment', 'url']]
        # initial_num = 706754427
        success_count = 0
        for i in range(1000):
            initial_num += 1
            url = "http://www.dianping.com/review/{}".format(initial_num)
            r = DianPing.get_html(url, DianPing.headers)
            print('url: ', url)
            print('r.status_code: ', r.status_code)
            if r.status_code == 200:
                if '页面无法访问' in r.text:
                    print('页面无法访问')
                    time.sleep(10 + 15 * random.random())
                    continue
                else:
                    html = r.content.decode()
                    b = re.findall(
                        "<div class='logo' id='logo'>验证中心</div>", html, re.S)
                    if b:
                        print('------需要验证呀，我先退出------')
                        break
            else:
                html = None
            print('html: ', html)
            if html:
                # print(HTMLParser(html).css("div[class='review-words']")[0].text())
                css_content = DianPing.get_css_content(
                    html, DianPing.headers, DianPing.r_sessoion)
                if css_content == 'error':
                    print('css_content: error')
                    time.sleep(10 + 15 * random.random())
                    continue
                font_dic, y_list = DianPing.get_font_dic(css_content)
                html_new = DianPing.get_html_full_review(
                    html, css_content, font_dic, y_list)

                temp = None
                if 'span class="score"' in html_new:
                    temp = HTMLParser(html_new).css(
                        "span[class='score']")[1].html.strip()
                if '口味' in str(temp):
                    for node in HTMLParser(temp).css("span[class='item']"):
                        if '环境' in node.text():
                            rating_env = ''.join(
                                [s for s in node.text() if s.isdigit() or s == '.'])
                        elif '服务' in node.text():
                            rating_service = ''.join(
                                [s for s in node.text() if s.isdigit() or s == '.'])
                        elif '口味' in node.text():
                            rating_flavor = ''.join(
                                [s for s in node.text() if s.isdigit() or s == '.'])
                        elif '食材' in node.text():
                            rating_food = ''.join(
                                [s for s in node.text() if s.isdigit() or s == '.'])
                    rating = sum([float(rating_env), float(
                        rating_service), float(rating_flavor)])/3
                    temp = HTMLParser(html_new).css(
                        "div[class='review-detail-nav']")[0].html
                    userid = HTMLParser(temp).css("span")[0].text().strip()
                    restid = HTMLParser(temp).css("a")[-1].text().strip()
                    comment = HTMLParser(html_new).css(
                        "div[class='review-words']")[0].text().strip().replace('\n', '')
                    timestamp = HTMLParser(html_new).css(
                        "span[class='time']")[0].text().strip()
                    if '更新于' in timestamp:
                        timestamp = timestamp.split('更新于')[1]
                    timestamp = time.mktime(datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M").timetuple()) * 1000

                    results.append([success_count, userid, restid, rating, float(rating_env), float(
                        rating_flavor), float(rating_service), timestamp, comment, url])
                    mystr = ' | '.join(str(x) for x in [success_count + 212, userid, restid, rating, float(
                        rating_env), float(rating_flavor), float(rating_service), timestamp, comment, url])
                    with open('./dazhong.txt', 'a', encoding='utf8') as the_file:
                        the_file.write(mystr + '\n')

                    print('------开始在数据库写数据------')
                    reviewId = initial_num  # 评论ID
                    user_id = ''  # 用户Id
                    name = userid  # 用户名
                    comment = comment  # 评论
                    star = rating  # 评分
                    timestamp = int(timestamp)  # 评论时间戳

                    rating = Rating(reviewId, user_id, name, '', restid,
                                    star, comment, url, timestamp, '大众点评')
                    rating.insert()

                    success_count += 1
                    print(results)
                    DianPing.update_db(initial_num)
                    if success_count == 3:
                        print('------爬到了3个数据，休息一下，免的被封------')
                        break
                time.sleep(10 + 15 * random.random())

    @staticmethod
    def update_db(initial_num):
        a = DB.find_one("dianping_register", {"crawby": "jack"})
        if a:
            DB.update("dianping_register", {"crawby": "jack"}, {
                      '$set': {'initial_num': initial_num}})
        else:
            DB.insert('dianping_register', {
                      "crawby": "jack", 'initial_num': initial_num})
        pass
