# -*- coding: utf-8 -*-
from app.models.webcrawler.meituan import MeiTuan
from app.models.webcrawler.dianping import DianPing
from app.models.service.rating_process_service import RatingProcessService


def MeiTuanSchd():
    '''美团调度器'''
    MeiTuan.Init()
    MeiTuan.Process()
    print('meituan finish')
    pass


def DianPingSchd():
    '''点评调度器'''
    DianPing.Init()
    DianPing.Process()
    print('dianping finish')
    pass


def RatingProcessSchd():
    '''数据处理调度器'''
    RatingProcessService.Init()
    RatingProcessService.Process()
    print('process finish')
    pass
