# -*- coding: utf-8 -*-
import pymongo
from urllib import parse
from app.models.config import Config


class DB(object):

    if Config.MongoDbAuth:
        URI = 'mongodb://{}:{}@{}:{}'.format(
            Config.MongoDbUsername,
            Config.MongoDbPassword,
            Config.MongoDbHost,
            Config.MongoDbPort)
    else:
        URI = 'mongodb://{}:{}'.format(Config.MongoDbHost, Config.MongoDbPort)

    @staticmethod
    def init():
        client = pymongo.MongoClient(DB.URI)
        DB.DATABASE = client[Config.MongoDbName]

    @staticmethod
    def insert(collection, data):
        DB.DATABASE[collection].insert(data)

    @staticmethod
    def find_one(collection, query):
        return DB.DATABASE[collection].find_one(query)

    @staticmethod
    def find_all(collection, query=''):
        if query == '':
            find = DB.DATABASE[collection].find()
        else:
            find = DB.DATABASE[collection].find(query)
        return find

    @staticmethod
    def find_max(collection, query, column):
        return DB.DATABASE[collection].find(query).sort(column, pymongo.DESCENDING).limit(1)[0]

    @staticmethod
    def update(collection, query, data):
        return DB.DATABASE[collection].update_one(query, data, upsert=True)
