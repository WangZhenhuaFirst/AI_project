# -*- coding: utf-8 -*-

#!/usr/bin/env python
import datetime

from app.database import DB


class RatingProcessed(object):

    def __init__(self, reviewId, userId, username, restId,
                 resttitle, rating, comment, url, timestamp, source,
                 sentiment_label, sentiment_classification, negative_prob, positive_prob):
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

        self.sentiment_label = sentiment_label  # 评分
        self.sentiment_classification = sentiment_classification  # 评分
        self.negative_prob = negative_prob  # 负概率
        self.positive_prob = positive_prob  # 整概率
        self.created_date = datetime.datetime.utcnow()  # 抓取时间

    def insert(self):
        if self.reviewId > 0:
            if not DB.find_one("RatingProcessed", {"reviewId": self.reviewId}):
                DB.insert(collection='RatingProcessed', data=self.json())
            else:
                return True
        else:
            if not DB.find_one("RatingProcessed", {"userId": self.userId, "restId": self.restId}):
                DB.insert(collection='RatingProcessed', data=self.json())
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
            'sentiment_label': self.sentiment_label,
            'sentiment_classification': self.sentiment_classification,
            'negative_prob': self.negative_prob,
            'positive_prob': self.positive_prob,
            'created_date': self.created_date,
        }
