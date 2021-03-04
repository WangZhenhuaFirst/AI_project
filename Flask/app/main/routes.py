# -*- coding: utf-8 -*-
from flask import render_template, request, jsonify, url_for, session
from app.main import bp  # noqa
from app.models.job import Job
from app.models.forms import NewsForm
from app.models.forms import SentimentAnalysisForm
# from app.models.forms import PDFKeyWordAutoHighlightForm
from app.newsdata import NewsData
from app.sentence2vec import sentence2vec
import json
from app.database import DB
import pymongo
import re
from bson import json_util
from app.sentiment_analysis import SentimentAnalysis
import os
from app.models.config import Config
import datetime
from flask import Flask
import sys
import uuid
import subprocess
# from app.pdf2txt import parse
from app.get_similar_keywords import Keywords, Annoysimilarwords
import jieba
from app.robot_main_response import Response

# index页面
@bp.route("/", methods=['Get', 'Post'])
def Index():
    cleanMemory(1)
    newform = GetNews()
    if newform.validate_on_submit():
        pass
    return render_template("Index.html", form=newform)

# about页面
@bp.route("/About", methods=['Get'])
def About():
    return render_template("About.html")

# 获取摘要信息
@bp.route("/GetNewSummary", methods=['Post'])
def GetNewSummary():
    NewsTitle = request.values['NewsTitle']
    NewsContent = request.values['NewsContent']
    NewSummaryLength = int(request.values['NewSummaryLength'])

    news_title = '{}'.format(NewsTitle)
    news_content = '{}'.format(NewsContent)
    # todo
    summary = sentence2vec.get_summarize(
        news_content, news_title, weight=0.7, top_n=NewSummaryLength)

    return jsonify({"result": summary})


@bp.route("/LoadData", methods=['GET'])
def LoadData():
    newform = GetNews()
    # d = newform.GetDict()
    # jsonret = json.dumps(newform.__dict__,ensure_ascii=False)
    return jsonify({"result":
                    {
                        "NewsTitle": newform.NewsTitle,
                        "NewsContent": newform.NewsContent,
                        "NewSummaryLength": newform.NewSummaryLength
                    }
                    })


def GetNews():
    newform = NewsData.GetNewData()
    return newform


@bp.route("/SentimentAnalysis/Index", methods=['Get', 'Post'])
def SentimentAnalysisIndex():
    cleanMemory(2)
    return render_template("SentimentAnalysis.html", form=None)

# 获取评论情感占比
@bp.route("/SentimentAnalysis/GetSentimentCount", methods=['Get'])
def GetSentimentCount():

    TotalSentiment = DB.DATABASE['RatingProcessed'].find().count()

    NegativeSentiment = DB.DATABASE['RatingProcessed'].find(
        {'sentiment_label': 0}).count()
    PositiveSentiment = DB.DATABASE['RatingProcessed'].find(
        {'sentiment_label': 1}).count()

    return jsonify({"result":
                    {
                        "TotalSentiment": TotalSentiment,
                        "NegativeSentiment": NegativeSentiment,
                        "PositiveSentiment": PositiveSentiment
                    }
                    })

# 获取来源占比
@bp.route("/SentimentAnalysis/getSentimentSource", methods=['Get'])
def getSentimentSource():

    MeiTuan = DB.DATABASE['RatingProcessed'].find({'source': '美团'}).count()
    DaZhongDianPing = DB.DATABASE['RatingProcessed'].find(
        {'source': '大众点评'}).count()

    return jsonify({"result":
                    {
                        "MeiTuan": MeiTuan,
                        "DaZhongDianPing": DaZhongDianPing,
                    }
                    })

# 获取最新评论
@bp.route("/SentimentAnalysis/getRecentSentiment", methods=['Get'])
def getRecentSentiment():

    result = DB.DATABASE['RatingProcessed'].find().sort(
        'created_date', pymongo.DESCENDING).limit(5)
    row = list(result)
    total = len(row)
    return json_util.dumps({"total": total,
                            "rows": row,
                            })


# 情感分析搜索
@bp.route("/SentimentAnalysis/Search", methods=['Get', 'Post'])
def SentimentAnalysisSearch():
    cleanMemory(2)
    return render_template("SentimentAnalysisSearch.html", form=None)

# 获取搜索结果
@bp.route("/SentimentAnalysis/GetSentimentAnalysisSearchResult", methods=['Get', 'Post'])
def GetSentimentAnalysisSearchResult():
    KeyWord = request.values['KeyWord']
    Limit = int(request.values['limit'])
    Offset = int(request.values['offset'])

    # 评论中包含关键字的数据
    if KeyWord != '':
        result = DB.DATABASE['RatingProcessed'].find({'comment': re.compile(KeyWord)}).sort(
            'created_date', pymongo.DESCENDING).skip(Offset).limit(Limit)
    else:
        result = DB.DATABASE['RatingProcessed'].find().sort(
            'created_date', pymongo.DESCENDING).skip(Offset).limit(Limit)
    total = result.count()

    return json_util.dumps({"total": total,
                            "rows": list(result),
                            })

# 情感分析-20分类
@bp.route("/SentimentAnalysis/Analysis", methods=['Get', 'Post'])
def SentimentAnalysisAnalysis():

    cleanMemory(2)

    if 'reviewId' in request.values:
        reviewId = int(request.values['reviewId'])
        result = DB.DATABASE['RatingProcessed'].find({'reviewId': reviewId}).sort(
            'created_date', pymongo.DESCENDING).limit(1)[0]
        AnalysisFrom = SentimentAnalysisForm(result['comment'])
    else:
        comment = SentimentAnalysis.get_test_comment()

        AnalysisFrom = SentimentAnalysisForm(comment)
    return render_template("SentimentAnalysisAnalysis.html", form=AnalysisFrom)


@bp.route("/SentimentAnalysis/GetSentimentAnalysis", methods=['Get', 'Post'])
def GetSentimentAnalysis():
    Comment = request.values['Comment']
    data = None
    if Comment != '':
        data = SentimentAnalysis.predict(Comment)

    # 评论分析
    return json_util.dumps({"result":
                            {
                                "data": data,
                            }
                            })


# PDF文件关键信息自动高亮
@bp.route("/PDFKeyWordAutoHighlight/Index", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightIndex():
    cleanMemory(3)
    return render_template("PDFKeyWordAutoHighlight.html", form=None)


@bp.route("/PDFKeyWordAutoHighlight/PdfUpload", methods=['Get', 'Post'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            app = Flask(__name__)

            guid = str(uuid.uuid1())
            filename = guid + os.path.splitext(file.filename)[1]
            base_path = os.path.abspath(
                os.path.dirname(os.path.dirname(__file__)))
            upload_path = os.path.join(base_path, Config.UPLOAD_FOLDER)
            file.save(os.path.join(upload_path, filename))

            session['UploadPath'] = upload_path

            session['Guid'] = guid
            # PDF文件名
            session['File'] = filename
            # html文件名
            session['Html'] = guid + '.html'
            # HighLight文件名
            session['HighLight'] = guid + '-HighLight.html'
            # HighLight次数
            session['HighLightCount'] = 0
            # pdf相对路径
            session['FilePath'] = Config.Uploads + filename
            # pdf绝对路径
            session['FileRelPath'] = os.path.join(upload_path, filename)
            # Html相对路径
            session['HtmlPath'] = Config.Uploads + guid + '.html'
            # Html绝对路径
            session['HtmlRelPath'] = os.path.join(upload_path, guid + '.html')
            # HighLight相对路径
            session['HighLightPath'] = Config.Uploads + \
                guid + '-HighLight.html'

            # HighLight绝对路径
            session['HighLightRelPath'] = os.path.join(
                upload_path, guid + '-HighLight.html')
            return render_template("PDFKeyWordAutoHighlight.html", form=None)
    return render_template("PDFKeyWordAutoHighlight.html", form=None)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in Config.ALLOWED_EXTENSIONS


@bp.route("/PDFKeyWordAutoHighlight/GetPdfFilePath", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightGetPdfFilePath():
    filepath = session.get('FilePath')

    # 评论分析
    return json_util.dumps({"result":
                            {
                                "path": filepath,
                            }
                            })


@bp.route("/PDFKeyWordAutoHighlight/GetHtmlOrgFilePath", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightGetHtmlOrgFilePath():
    filepath = session.get('HtmlPath')

    # 评论分析
    return json_util.dumps({"result":
                            {
                                "path": filepath,
                            }
                            })


@bp.route("/PDFKeyWordAutoHighlight/GetHighLightFilePath", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightGetHighLightFilePath():
    filepath = session.get('HighLightPath')

    # 评论分析
    return json_util.dumps({"result":
                            {
                                "path": filepath,
                            }
                            })


@bp.route("/PDFKeyWordAutoHighlight/ConvertPdf2Html", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightConvertPdf2Html():

    exist = os.path.exists(session.get('HtmlRelPath'))

    # html不存在，执行转html文件
    if not exist:
        base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        # 处理程序
        process_file = os.path.join(base_path, Config.pdf2htmlEX)

        # 输入路径
        input_file = session.get('FileRelPath')

        # 输出路径
        #output_file = Config.HtmlPath + session.get('Html')
        output_file = 'app' + session.get('HtmlPath')

        if os.name == 'nt':
            output_file = Config.HtmlPath + session.get('Html')
            subprocess.run([process_file, input_file, output_file])
        else:
            subprocess.run(['pdf2htmlEX', input_file, output_file])

    return json_util.dumps({"result":
                            {
                                "data": True,
                                "html": session.get('HtmlPath')
                            }
                            })


@bp.route("/PDFKeyWordAutoHighlight/ProcessPdf", methods=['Get', 'Post'])
def PDFKeyWordAutoHighlightProcessPdf():
    cleanMemory(3)

    # 初始化高亮文件名
    InitHighLightPath()

    pdfstr = parse(session.get('FileRelPath'))
    kw = request.values['KeyWord']

    if len(kw) > 0:
        kwlist = list(jieba.cut(kw))
        keyword = {}
        for i in kwlist:
            keyword.update(Annoysimilarwords.get_similar_words(i, weight=0.2))
    else:
        keyword = Keywords.get_keywords(sentence=pdfstr, topn=20)

    # 读取文件高亮
    ret = HighLightFile(keyword, session.get(
        'HtmlRelPath'), session.get('HighLightRelPath'))

    return json_util.dumps({"result":
                            {
                                "ret": ret,
                                "path": session.get('HighLightPath')
                            }
                            })


def InitHighLightPath():
    HighLightCount = int(session.get('HighLightCount'))
    HighLightCount = HighLightCount + 1
    session['HighLightCount'] = HighLightCount

    # HighLight相对路径
    session['HighLightPath'] = Config.Uploads + session.get('Guid') + \
        '-HighLight' + str(HighLightCount) + '.html'

    # HighLight绝对路径
    session['HighLightRelPath'] = os.path.join(session.get(
        'UploadPath'), session.get('Guid') + '-HighLight' + str(HighLightCount) + '.html')


def HighLightFile(keyword, inputfile, outputfile):
    file = open(inputfile, 'r', encoding='UTF-8')
    html = file.read()
    patstr = ' '.join(keyword.keys())
    highlightHtml = highlight_keywords(get_query_pat(patstr), html)

    with open(outputfile, "w", encoding='UTF-8') as f:
        f.write(highlightHtml)

    return True


def get_query_pat(query):
    return re.compile('({})'.format('|'.join(query.split())))


def highlight_keywords(pat, document):
    return pat.sub(repl='<span style="background:orange;">\g<1></span>', string=document)


# 对话机器人
@bp.route("/ChatRobot/Index", methods=['Get', 'Post'])
def ChatRobotIndex():
    cleanMemory(4)
    return render_template("ChatRobot.html", form=None)

# 对话
@bp.route("/ChatRobot/Chat", methods=['Get', 'Post'])
def ChatRobotChat():
    Content = request.values['Content']
    res = Response.generate(Content)
    # print('res: ', res)

    # 评论分析
    return json_util.dumps({"result": {"data": res}})


def cleanMemory(projectnum):
    '''按照项目编号清理内存'''
    # 文本摘要
    if projectnum == 1:
        project02clean()
        # project03clean()
        project04clean()
        project01init()

    # 情感分析
    elif projectnum == 2:
        project01clean()
        # project03clean()
        project04clean()
        project02init()
    # 关键信息自动高亮
    elif projectnum == 3:
        # project01clean()
        # project02clean()
        # project03clean()
        project03init()
    # project04
    elif projectnum == 4:
        project01clean()
        project02clean()
        # project03clean()
        project04init()


def project01init():
    sentence2vec.init()
    NewsData.init()


def project01clean():
    sentence2vec.clean()
    NewsData.clean()


def project02init():
    SentimentAnalysis.Init()


def project02clean():
    SentimentAnalysis.clean()


def project03init():
    Keywords.init()
    Annoysimilarwords.init()
    pass


def project03clean():
    Keywords.clean()
    Annoysimilarwords.clean()
    pass


def project04init():
    pass


def project04clean():
    pass
