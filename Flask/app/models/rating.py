# -*- coding: utf-8 -*-

#!/usr/bin/env python
import datetime

from app.database import DB


class Rating(object):

    def __init__(self, reviewId, userId, username, restId,
                 resttitle, rating, comment, url, timestamp, source):
        self.reviewId = reviewId  # 评论ID
        self.userId = userId  # 用户ID
        self.username = username  # 用户名
        self.restId = restId  # 店铺ID
        self.resttitle = resttitle  # 店铺
        self.rating = rating  # 评分
        self.comment = comment  # 评论
        self.url = url  # 评论链接
        self.timestamp = timestamp  # 评论时间戳
        self.source = source  # 来源:美团、点评
        self.created_date = datetime.datetime.utcnow()  # 抓取时间

        # 用于处理情感分析,processed:0-未处理，1-已处理
        self.processed = 0
        self.processMessage = ''

    def insert(self):
        if self.reviewId != "":
            print('执行插入了1:', self.reviewId)
            if not DB.find_one("Ratings", {"reviewId": self.reviewId}):
                print()
                DB.insert(
                    collection='Ratings', data=self.json())
                insert_data = self.json()
                print('insert_data:', insert_data)
            else:
                return True
        else:
            print('执行插入了2')
            if not DB.find_one("Ratings", {"userId": self.userId, "restId": self.restId}):
                DB.insert(collection='Ratings', data=self.json())
            else:
                return True
        return False

    def json(self):
        return {
            'reviewId': self.reviewId,
            'userId': self.userId,
            'username': self.username,
            'restId': self.restId,
            'resttitle': self.resttitle,
            'rating': self.rating,
            'comment': self.comment,
            'url': self.url,
            'timestamp': self.timestamp,
            'source': self.source,
            'created_date': self.created_date,
            'processed': self.processed,
            'processMessage': self.processMessage
        }
