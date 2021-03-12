import json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import pandas as pd
import re
import jieba
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def metrics_result(y_test, y_pred, predict_prod):
    '''定义分类评估指标, y_test为真实的类别，y_pred为预测的类别，predict_prod为预测是各个类别的概率'''
    # average: This parameter is required for multiclass/multilabel targets.
    # If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data
    # weighted: Calculate metrics for each label, and find their average weighted by support
    accuracy_scores = accuracy_score(y_test, y_pred)
    precision_scores = precision_score(y_test, y_pred, average='weighted')
    recall_scores = recall_score(y_test, y_pred, average='weighted')
    f1_scores = f1_score(y_test, y_pred, average='weighted')
    fpr, tpr, threshold = roc_curve(y_test, predict_prod)
    auc_scores = auc(fpr, tpr)
    print('准确率:{0:.3f}'.format(accuracy_scores))
    print('精确率:{0:.3f}'.format(precision_scores))
    print('召回率:{0:0.3f}'.format(recall_scores))
    print('f1-score:{0:.3f}'.format(f1_scores))
    print('AUC:{0:.3f}'.format(auc_scores))
    return precision_scores, recall_scores, f1_scores, auc_scores, accuracy_scores


def svc_cv(C, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    # random_state Controls the pseudo random number generation for shuffling the data
    svm = LinearSVC(C=C, random_state=2)
    estimator = CalibratedClassifierCV(svm)
    # The simplest way to use cross-validation is to call the cross_val_score
    # helper function on the estimator and the dataset
    # estimator 指模型
    # cv determines the cross_validation splitting strategy
    # n_jobs: Number of jobs to run in parallel.
    cval = cross_val_score(estimator, data, targets,
                           scoring='roc_auc', cv=5, n_jobs=6)
    return cval.mean()


def optimize_svc(data, targets):
    """
    参考：https://github.com/fmfn/BayesianOptimization
    Apply Bayesian Optimization to SVC parameters.
    """

    def svc_crossval(expC):
        """
        Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = expC
        # gamma = 10 ** expGamma
        # cv，即 cross validation
        return svc_cv(C=C, data=data, targets=targets)

    # 贝叶斯优化，以 cross validation 为目标函数，得到最优的超参数 C
    optimizer = BayesianOptimization(
        f=svc_crossval,  # function
        # function's parameters with their corresponding bounds
        pbounds={"expC": (0, 2)},
        random_state=1234,
        verbose=2
    )
    # 改变n_iter可以增加或减少调参次数
    # How many steps of bayesian optimization you want to perform.
    # The more steps the more likely to find a good maximum you are
    optimizer.maximize(n_iter=10)

    # The best combination of parameters and target value found
    print("Final result:", optimizer.max)
    return optimizer.max['params']


def cut_text(text):
    '''中文分词，用正则去除多余的符号'''
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
    text_pred = svc_model.predict(text_matrix)
    text_prod = svc_model.predict_proba(text_matrix)
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
    # TfidfVectorizer是CountVectorizer + TfidfTransformer的组合，输出的是各个文本各个词的TF-IDF值
    # min_df=5, max_features=10000
    tfidf_vec = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf_vec.fit_transform(data['comment'].astype('U'))
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, data['rating'], test_size=0.2, random_state=1)  # ,stratify = y

    # 如果需要调参，请把注释去掉
    print(Colours.yellow("--- Optimizing SVM ---"))
    params = optimize_svc(X_train, y_train)  # 是为了获取最优超参数 C
    # Regularization parameter. The strength of the regularization is inversely proportional to C.
    svm = LinearSVC(C=params['expC'])

    # 获取调参后最优的参数；如果需要调参，请注释掉下面一行代码
    # svm = LinearSVC(C=0.3830389007577846)

    # 概率校准
    svc = CalibratedClassifierCV(svm)
    svc.fit(X_train, y_train)
    # svc_y_pred为预测类别，svc_y_prod为预测属于各个类别的概率
    svc_y_pred = svc.predict(X_test)
    # 参考：https://blog.csdn.net/u011630575/article/details/79429757
    svc_y_prod = svc.predict_proba(X_test)[:, 1]

    # 展示模型的各个评分
    svc_ms = metrics_result(y_test, svc_y_pred, svc_y_prod)
    # 保存模型文件
    joblib.dump(svc, '2_svc_model.pkl')

    # 加载模型文件
    svc_model = joblib.load('2_svc_model.pkl')

    # 测试获取输入文本的情感
    text = '这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，问老板娘，说已经改了只给4碟，没有这个套餐了。\
    那我买这个券这个套餐，份量不给足我？ 你说改了就改了，我都不知情，那不你突然想加收就加收？\
    那我买这个套餐写明有这些东西，那你一样也不能少，无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？\
    这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。'
    predict_result = get_sentiment(text, model=svc_model)
    print("predict_result:", predict_result)

"""
|   iter    |  target   |   expC    |
-------------------------------------
|  1        |  0.9308   |  0.383    |
|  2        |  0.9298   |  1.244    |
|  3        |  0.9301   |  0.8755   |
|  4        |  0.9296   |  1.571    |
|  5        |  0.9296   |  1.56     |
|  6        |  0.8901   |  0.000454 |
|  7        |  0.9294   |  2.0      |
|  8        |  0.9304   |  0.5916   |
|  9        |  0.9295   |  1.817    |
|  10       |  0.9306   |  0.5086   |
|  11       |  0.9299   |  1.06     |
|  12       |  0.9306   |  0.5022   |
|  13       |  0.9303   |  0.671    |
|  14       |  0.9306   |  0.4801   |
|  15       |  0.9297   |  1.36     |
=====================================
Final result: {'target': 0.9308367494445768,
'params': {'expC': 0.3830389007577846}}
精度:0.860
召回:0.860
f1-score:0.860
accuracy_scores:0.860
AUC:0.933


Final result: {"text": "这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，
问老板娘，说已经改了只给4碟，没有这个套餐了。 那我买这个券这个套餐，份量不给足我？
你说改了就改了，我都不知情，那不你突然想加收就加收？那我买这个套餐写明有这些东西，那你一样也不能少，
无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？
这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。", 
"sentiment_label": 0, "sentiment_classification": "负面情感",
"negative_prob": "0.9883462422119251", "positive_prob": "0.011653757788074886"}
"""
