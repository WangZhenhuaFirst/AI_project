# -*- coding: utf-8 -*-

#!/usr/bin/env python

from gevent.pywsgi import WSGIServer
from flask import Flask
from app.database import DB
from flask_bootstrap import Bootstrap
from app.models.job import Job
from app.models.config import Config
from app.newsdata import NewsData
import pandas as pd
import os
from gensim.models import Word2Vec
from app.sentence2vec import sentence2vec
from flask_apscheduler import APScheduler
from app.models.scheduler import RatingProcessSchd
from app.robot_main_response import Response

from gevent import monkey

os.environ["GEVENT_SUPPORT"] = 'True'
monkey.patch_all()


def create_app():
    app = Flask(__name__)
    bootstrap = Bootstrap(app)
    app.config.from_object(Config)

    # 根据config启用mongodb
    if Config.MongoDbEnable:
        DB.init()

    register_blueprints(app)

    # 启动爬虫定时任务
    if Config.WebCrawlerEnable:
        scheduler = APScheduler()
        # 注册app
        scheduler.init_app(app)
        scheduler.start()

    # 启用数据处理
    if Config.ProcessDataEnable:
        RatingProcessSchd()

    Response.init()
    return app


def register_blueprints(app):
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)


if __name__ == "__main__":
    app = create_app()
    # app.run(threaded=False, debug=False)
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
