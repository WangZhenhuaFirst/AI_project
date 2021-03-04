import pymongo
import datetime
import re

client = pymongo.MongoClient(
    "mongodb://wangzhenhua:wstc9093@localhost")  # defaults to port 27017

db = client.auto_monitoring

# print the number of documents in a collection
# print (db.testcol.insert({'a':'test'}))
# print (db.testcol.find().next())
# print (db.Ratings.find().next())
# print (db.Ratings.find_one({"source": "美团"}))
# print(db.Ratings.find({"source": "美团"}).sort('timestamp', pymongo.DESCENDING).limit(1)[0]) #


db.RatingProcessed.insert_one({
    'reviewId': 1,
    'userId': 1,
    'username': '测试',
    'restId': 1,
    'resttitle': '测试餐厅',
    'rating': 0,
    'comment': '好吃',
    'url': '',
    'timestamp': 1587286800000,
    'source': '美团',
    'sentiment_label': 1,
    'sentiment_classification': '正面情感',
    'negative_prob': 0.01,
    'positive_prob': 99.9,
    'created_date': datetime.datetime.utcnow(),
})

db.RatingProcessed.insert_one({
    'reviewId': 2,
    'userId': 2,
    'username': '测试',
    'restId': 2,
    'resttitle': '测试餐厅',
    'rating': 0,
    'comment': '难吃',
    'url': '',
    'timestamp': 1587286800000,
    'source': '大众点评',
    'sentiment_label': 0,
    'sentiment_classification': '负面情感',
    'negative_prob': 99.9,
    'positive_prob': 0.01,
    'created_date': datetime.datetime.utcnow(),
})


# print(db.RatingProcessed.find().next())
# print(db.RatingProcessed.find().sort(
#     'created_date', pymongo.DESCENDING).limit(1)[0])

# for i in list(db.RatingProcessed.find({'comment': re.compile('难')}).sort('created_date', pymongo.DESCENDING).limit(5)):
#     print(i)
