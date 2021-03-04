#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# app.py
from flask import Flask
from flask import jsonify
from flask import make_response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/', methods=['GET'])
def index():
    data = {'project': 'api',
            'success': 'true',
            'message': 'Here Is Index.'}
    return make_response(jsonify(data))


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)