#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# app.py
from flask import Flask
from flask import jsonify
from flask import make_response
import gevent.pywsgi  # 导入相关的包
import gevent.monkey

gevent.monkey.patch_all()  # 可选内容，是否加载 猴子补丁
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/', methods=['GET'])
def index():
    data = {'project': 'api',
            'success': 'true',
            'message': 'Here Is Index.'}
    return make_response(jsonify(data))


if __name__ == '__main__':
    # app.run(debug=True, host='127.0.0.1', port=5000)  # 原flask默认部署
    gevent_server = gevent.pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    gevent_server.serve_forever()
