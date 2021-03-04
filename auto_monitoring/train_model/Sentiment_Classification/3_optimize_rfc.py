# -*- coding: utf-8 -*-
import json
from sklearn.externals import joblib
import pandas as pd
import re
import jieba
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def metrics_result(y_test, y_pred, predict_prod):
    '''定义分类评估指标，y_test为真实的类别，y_pred为预测的类别，predict_prod为预测类别的概率'''
    precision_scores = precision_score(y_test, y_pred, average='weighted')
    recall_scores = recall_score(y_test, y_pred, average='weighted')
    f1_scores = f1_score(y_test, y_pred, average='weighted')
    accuracy_scores = accuracy_score(y_test, y_pred)
    fpr, tpr, threshold = roc_curve(y_test, predict_prod)
    auc_scores = auc(fpr, tpr)
    print('精确度:{0:.3f}'.format(precision_scores))
    print('召回率:{0:0.3f}'.format(recall_scores))
    print('f1-score:{0:.3f}'.format(f1_scores))
    print('准确度:{0:.3f}'.format(accuracy_scores))
    print('AUC:{0:.3f}'.format(auc_scores))
    return precision_scores, recall_scores, f1_scores, auc_scores, accuracy_scores


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=5, n_jobs=6)
    return cval.mean()


def optimize_rfc(data, targets):
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        return rfc_cv(
            # 参考：https://www.cnblogs.com/pinard/p/6160412.html
            n_estimators=int(n_estimators),
            # The minimum number of samples required to split an internal node
            # 限制子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分
            # 默认是2，如果样本量不大，不需要管这个值。如果样本量 数量级非常大，则推荐增大这个值。
            min_samples_split=int(min_samples_split),
            # RF划分时考虑的最大特征数 max_features
            # 如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。
            # N为样本总特征数
            # 一般用默认的"auto"就可以，如果特征数非常多，可以灵活使用其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999)
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=1)

    print("Final result:", optimizer.max)
    return optimizer.max['params']


def cut_text(text):
    '''中文分词函数，用正则去除多余的符号'''
    text = str(text)
    stopwords = [line.strip() for line in open(
        'chinese_stopwords.txt', encoding='UTF-8').readlines()]
    text = ''.join(re.findall('[\u4e00-\u9fff]', text))
    seg_list = jieba.cut(text)
    sentence_segment = []
    for word in seg_list:
        if word not in stopwords:
            sentence_segment.append(word)
    # sentence_segment.append(word)
    # 把已去掉停用词的sentence_segment，用' '.join()拼接起来
    seg_res = ' '.join(sentence_segment)
    return seg_res


def get_sentiment(txt, model=None):
    """ 获取文本情感
    :param txt: 输入的文本
    :return: 情感分析的结果，json格式
    """
    text = str(txt)
    text = cut_text(text)
    text_matrix = tfidf_vec.transform([text])
    text_pred = model.predict(text_matrix)
    text_prod = model.predict_proba(text_matrix)
    # print('预测结果',test_model_pred)
    # print(np.argmax(test_model_pred))
    if text_prod[0][0] > text_prod[0][1]:
        sentiment_label = 0
        sentiment_classification = '负面情感'
    else:
        sentiment_label = 1
        sentiment_classification = '正面情感'
    negative_prob = str(text_prod[0][0])
    positive_prob = str(text_prod[0][1])
    result = {'text': txt,
              'sentiment_label': sentiment_label,
              'sentiment_classification': sentiment_classification,
              'negative_prob': negative_prob,
              'positive_prob': positive_prob}
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # 读取分词文件
    data = pd.read_csv('seg_ratings_data.txt', sep='\t')
    # TfidfVectorizer 是 CountVectorizer + TfidfTransformer的组合，输出的各个文本各个词的TF-IDF值
    # min_df=5, max_features=10000
    tfidf_vec = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf_vec.fit_transform(data['comment'].astype('U'))
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, data['rating'], test_size=0.2, random_state=1)

    # 调参
    # print(Colours.yellow("--- Optimizing RandomForest ---"))
    # params = optimize_rfc(X_train, y_train)
    # rfc = RFC(n_estimators=params['n_estimators'],
    #           min_samples_split=params['min_samples_split'],
    #           max_features=params['max_features'])

    # 不调参
    rfc = RFC()

    rfc.fit(X_train, y_train)

    # rfc_y_pred为预测类别，rfc_y_prod为预测类别的概率
    rfc_y_pred = rfc.predict(X_test)
    rfc_y_prod = rfc.predict_proba(X_test)[:, 1]

    # 展示模型的各个评分
    rfc_ms = metrics_result(y_test, rfc_y_pred, rfc_y_prod)
    # 保存模型文件
    joblib.dump(rfc, '3_rfc_model.pkl')

    # 加载模型文件
    rfc_model = joblib.load('3_rfc_model.pkl')

    # 测试获取输入文本的情感
    text = '这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，问老板娘，说已经改了只给4碟，没有这个套餐了。\
    那我买这个券这个套餐，份量不给足我？ 你说改了就改了，我都不知情，那不你突然想加收就加收？\
    那我买这个套餐写明有这些东西，那你一样也不能少，无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？\
    这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。'
    predict_result = get_sentiment(text, model=rfc_model)
    print('predict_result:', predict_result)

"""
精确度:0.828
召回率:0.828
f1-score:0.828
准确度:0.828
AUC:0.907

predict_result: {
"text": "这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，问老板娘，说已经改了只给4碟，没有这个套餐了。\
那我买这个券这个套餐，份量不给足我？ 你说改了就改了，我都不知情，那不你突然想加收就加收？\
那我买这个套餐写明有这些东西，那你一样也不能少，无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？\
这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。",
"sentiment_label": 0, "sentiment_classification": "负面情感",
"negative_prob": "0.92", "positive_prob": "0.08"}
"""
