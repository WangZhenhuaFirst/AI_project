# -*- coding: utf-8 -*-
from app.database import DB
import random


class Agent(object):
    '''代理'''

    agents = []

    # 从数据库获取代理列表
    @staticmethod
    def Init():
        if len(Agent.agents) == 0:
            Agent.agents = list(DB.find_all('Agents'))

    # 随机获取一个代理
    @staticmethod
    def GetAgent():
        ip = Agent.agents[random.randint(0, len(Agent.agents)-1)]['Agent']
        proxies = {"http": ip}
        return proxies
