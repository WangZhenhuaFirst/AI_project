# -*- coding: utf-8 -*-
import json
from sklearn.externals import joblib
import pandas as pd
import re
import jieba
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def bayes_parameter_opt_lgb(X, y,
                            init_round=1,
                            opt_round=1,
                            n_folds=5,
                            random_seed=6,
                            n_estimators=100,  # default=100, Number of boosted trees to fit
                            learning_rate=0.05,
                            output_process=True):
    # prepare data
    # categorical_feature = categorical_feats,
    # free_raw_data: 释放原始数据
    # if you want to re-use data, remember to set free_raw_data=False
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

    def lgb_eval(num_leaves,
                 feature_fraction,
                 bagging_fraction,
                 max_depth,
                 lambda_l1,
                 lambda_l2,
                 min_split_gain,
                 min_child_weight):
        params = {'application': 'binary', 'learning_rate': learning_rate,
                  'metric': 'auc', 'device': 'cpu'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight

        # Perform the cross-validation with given paramaters
        cv_result = lgb.cv(params,
                           train_data,
                           num_boost_round=100,  # number of boosting iterations, default = 100
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,  # 分层取样
                           verbose_eval=100,  # Whether to display the progress
                           metrics=['auc'],
                           # 早停阈值,如果一个验证集的度量在early_stopping_round 循环中没有提升，则停止训练
                           early_stopping_rounds=10)
        return max(cv_result['auc-mean'])

    # 定义调参空间
    lgbBO = BayesianOptimization(lgb_eval,
                                 {'num_leaves': (24, 60),  # max number of leaves in one tree
                                  # randomly select a subset of features on each iteration
                                  'feature_fraction': (0.1, 0.9),
                                  # randomly select part of data without resampling
                                  'bagging_fraction': (0.8, 1),
                                  'max_depth': (5, 8.99),
                                  'lambda_l1': (0, 5),  # L1 regularization
                                  'lambda_l2': (0, 3),
                                  # the minimal gain to perform split
                                  'min_split_gain': (0.001, 0.1),
                                  # 使一个结点分裂的最小海森值之和
                                  'min_child_weight': (5, 50)},
                                 random_state=0)
    # optimize
    # init_points: How many steps of random exploration you want to perform.
    # Random exploration can help by diversifying the exploration space
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return lgbBO.max['params']


def lgb_train(**opt_params):
    params = {'application': 'binary',  # 二分类模型
              'learning_rate': 0.1,
              'metric': 'auc',
              'device': 'cpu',
              'num_leaves': int(opt_params['num_leaves']),
              'feature_fraction': opt_params['feature_fraction'],
              'bagging_fraction': opt_params['bagging_fraction'],
              'max_depth': int(opt_params['max_depth']),
              'lambda_l1': opt_params['lambda_l1'],
              'lambda_l2': opt_params['lambda_l2'],
              'min_split_gain': opt_params['min_split_gain'],
              'min_child_weight': opt_params['min_child_weight'],
              'objective': 'binary',
              'boosting_type': 'gbdt',
              'verbose': 1}

    # 转换适合lgb的数据格式
    lgb_dtrain = lgb.Dataset(X_train, y_train)
    lgb_dtest = lgb.Dataset(X_test, y_test, reference=lgb_dtrain)
    # 显示训练过程中需要监控的数据
    lgb_watchlist = [lgb_dtest, lgb_dtrain]
    # 记录训练过程中监控的数据，需要定义一个空的dict
    lgb_progress = dict()
    # 用搜索得到的最佳参数训练新模型
    lgb_model = lgb.train(params,
                          lgb_dtrain,
                          num_boost_round=100,
                          early_stopping_rounds=10,
                          valid_sets=[lgb_dtest, lgb_dtrain],
                          valid_names=['test', 'train'],
                          evals_result=lgb_progress,
                          verbose_eval=100)
    return lgb_model


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
    print('准确率:{0:.3f}'.format(accuracy_scores))
    print('AUC:{0:.3f}'.format(auc_scores))
    return precision_scores, recall_scores, f1_scores, auc_scores, accuracy_scores


# 中文分词函数，用正则去除多余的符号
def cut_text(text):
    text = str(text)
    stopwords = [line.strip() for line in open(
        'chinese_stopwords.txt', encoding='UTF-8').readlines()]
    text = ''.join(re.findall('[\u4e00-\u9fff]', text))
    seg_list = jieba.cut(text)
    sentence_segment = []
    for word in seg_list:
        if word not in stopwords:
            sentence_segment.append(word)
    # 把已去掉停用词的sentence_segment拼接起来
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
    text_prod = model.predict(text_matrix)
    text_pred = model.predict_proba(text_matrix)
    print('text_prod:', text_prod)
    print('text_pred', text_pred)
    if text_prod[0] < 0.5:
        sentiment_label = 0
        sentiment_classification = '负面情感'
    else:
        sentiment_label = 1
        sentiment_classification = '正面情感'
    negative_prob = str(1 - text_prod[0])
    positive_prob = str(text_prod[0])
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

    # 开始调参，获取最优参数
    opt_params = bayes_parameter_opt_lgb(
        X_train, y_train, init_round=5, opt_round=5, n_folds=5, random_seed=6, n_estimators=100, learning_rate=0.1)

    # 使用最优参数进行训练
    # opt_params = {'application': 'binary',
    #               'learning_rate': 0.1,
    #               'metric': 'auc',
    #               'device': 'cpu',
    #               'num_leaves': int(59.72),
    #               'feature_fraction': 0.6073,
    #               'bagging_fraction': 0.9883,
    #               'max_depth': int(5.943),
    #               'lambda_l1': 4.948,
    #               'lambda_l2': 2.69,
    #               'min_split_gain': 0.09106,
    #               'min_child_weight': 5.322,
    #               'objective': 'binary',
    #               'boosting_type': 'gbdt',
    #               'verbose': 1,
    #               'metric': 'auc'}

    lgb_model = lgb_train(**opt_params)

    # If early stopping is enabled during training,
    # you can get predictions from the best iteration with bst.best_iteration
    # 使用best_iteration来做预测,lgb_bay_y_prod为概率值
    lgb_bay_y_prod = lgb_model.predict(
        X_test, ntree_limit=lgb_model.best_iteration)
    # 由于预测值为0-1的概率值，概率值，需要转成0或1
    lgb_bay_y_pred = [int(round(value)) for value in lgb_bay_y_prod]

    # 展示模型的各个评分
    lgb_bay_ms = metrics_result(y_test, lgb_bay_y_pred, lgb_bay_y_prod)
    # 保存模型文件
    joblib.dump(lgb_model, '5_lgb_model.pkl')

    # 加载模型文件
    lgb_model = joblib.load('5_lgb_model.pkl')

    # 测试获取输入文本的情感
    text = '这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，问老板娘，说已经改了只给4碟，没有这个套餐了。那我买这个券这个套餐，份量不给足我？ 你说改了就改了，我都不知情，那不你突然想加收就加收？那我买这个套餐写明有这些东西，那你一样也不能少，无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。'
    predict_result = get_sentiment(text, model=lgb_model)
    print('predict_result:', predict_result)

'''
------------lightgbm调参明细--------------------


|   iter    |  target   | baggin... | featur... | lambda_l1 | lambda_l2 | max_depth | min_ch... | min_sp... | num_le... |
-------------------------------------------------------------------------------------------------------------------------
[1000]	cv_agg's auc: 0.918881 + 0.00107081
[2000]	cv_agg's auc: 0.923376 + 0.000964134
[3000]	cv_agg's auc: 0.924427 + 0.000955539
|  1        |  0.9246   |  0.9098   |  0.6722   |  3.014    |  1.635    |  6.69     |  34.07    |  0.04432  |  56.1     |
[1000]	cv_agg's auc: 0.917766 + 0.00103872
[2000]	cv_agg's auc: 0.921487 + 0.000962244
[3000]	cv_agg's auc: 0.92225 + 0.000942644
|  2        |  0.9223   |  0.9927   |  0.4068   |  3.959    |  1.587    |  7.266    |  46.65    |  0.008033 |  27.14    |
[1000]	cv_agg's auc: 0.919955 + 0.000993665
[2000]	cv_agg's auc: 0.922856 + 0.000989007
|  3        |  0.9232   |  0.804    |  0.7661   |  3.891    |  2.61     |  8.905    |  40.96    |  0.04669  |  52.1     |
[1000]	cv_agg's auc: 0.919916 + 0.00104383
[2000]	cv_agg's auc: 0.924986 + 0.00102221
[3000]	cv_agg's auc: 0.926312 + 0.000989509
|  4        |  0.9266   |  0.8237   |  0.6119   |  0.7168   |  2.834    |  7.082    |  23.66    |  0.02719  |  51.87    |
[1000]	cv_agg's auc: 0.919538 + 0.00105602
[2000]	cv_agg's auc: 0.923894 + 0.000991848
[3000]	cv_agg's auc: 0.924739 + 0.000994501
|  5        |  0.9248   |  0.8912   |  0.5547   |  0.09395  |  1.853    |  7.442    |  32.76    |  0.09443  |  48.55    |
[1000]	cv_agg's auc: 0.918384 + 0.00106539
[2000]	cv_agg's auc: 0.92474 + 0.00100561
[3000]	cv_agg's auc: 0.927346 + 0.000927246
[4000]	cv_agg's auc: 0.928655 + 0.00092156
[5000]	cv_agg's auc: 0.929355 + 0.000906554
[6000]	cv_agg's auc: 0.929767 + 0.000897905
[7000]	cv_agg's auc: 0.929996 + 0.000905485
|  6        |  0.9301   |  0.8963   |  0.1282   |  4.562    |  0.1954   |  5.561    |  5.283    |  0.002552 |  24.09    |
[1000]	cv_agg's auc: 0.917722 + 0.00107635
[2000]	cv_agg's auc: 0.924376 + 0.00103482
[3000]	cv_agg's auc: 0.927096 + 0.000994993
[4000]	cv_agg's auc: 0.928489 + 0.000964058
[5000]	cv_agg's auc: 0.929255 + 0.000949583
[6000]	cv_agg's auc: 0.929758 + 0.000968959
[7000]	cv_agg's auc: 0.929994 + 0.000963651
[8000]	cv_agg's auc: 0.930099 + 0.000979089
|  7        |  0.9301   |  0.9883   |  0.6073   |  4.948    |  2.69     |  5.943    |  5.322    |  0.09106  |  59.72    |




精度:0.859
召回:0.859
f1-score:0.859
accuracy_scores:0.859
AUC:0.932




Training until validation scores don't improve for 100 rounds
[100]   train's auc: 0.873877   test's auc: 0.872512
[200]   train's auc: 0.892135   test's auc: 0.889239
[300]   train's auc: 0.901748   test's auc: 0.897792
[400]   train's auc: 0.907967   test's auc: 0.903153
[500]   train's auc: 0.912538   test's auc: 0.906892
[600]   train's auc: 0.916079   test's auc: 0.909713
[700]   train's auc: 0.919043   test's auc: 0.912048
[800]   train's auc: 0.921426   test's auc: 0.91388
[900]   train's auc: 0.92357    test's auc: 0.915423
[1000]  train's auc: 0.92538    test's auc: 0.916731
[1100]  train's auc: 0.927035   test's auc: 0.917881
[1200]  train's auc: 0.928591   test's auc: 0.918996
[1300]  train's auc: 0.92991    test's auc: 0.919855
[1400]  train's auc: 0.931192   test's auc: 0.920704
[1500]  train's auc: 0.932321   test's auc: 0.921447
[1600]  train's auc: 0.933372   test's auc: 0.922113
[1700]  train's auc: 0.934415   test's auc: 0.922764
[1800]  train's auc: 0.935333   test's auc: 0.923334
[1900]  train's auc: 0.936169   test's auc: 0.92386
[2000]  train's auc: 0.93695    test's auc: 0.92427
[2100]  train's auc: 0.93769    test's auc: 0.924732
[2200]  train's auc: 0.938383   test's auc: 0.925131
[2300]  train's auc: 0.939075   test's auc: 0.925494
[2400]  train's auc: 0.939664   test's auc: 0.925802
[2500]  train's auc: 0.940326   test's auc: 0.926149
[2600]  train's auc: 0.940903   test's auc: 0.926436
[2700]  train's auc: 0.94146    test's auc: 0.926732
[2800]  train's auc: 0.941965   test's auc: 0.926981
[2900]  train's auc: 0.942449   test's auc: 0.927208
[3000]  train's auc: 0.942906   test's auc: 0.927465
[3100]  train's auc: 0.943377   test's auc: 0.927709
[3200]  train's auc: 0.943835   test's auc: 0.927932
[3300]  train's auc: 0.944287   test's auc: 0.92815
[3400]  train's auc: 0.944705   test's auc: 0.928341
[3500]  train's auc: 0.945123   test's auc: 0.928524
[3600]  train's auc: 0.945529   test's auc: 0.9287
[3700]  train's auc: 0.9459     test's auc: 0.928862
[3800]  train's auc: 0.946275   test's auc: 0.929025
[3900]  train's auc: 0.946629   test's auc: 0.929184
[4000]  train's auc: 0.947029   test's auc: 0.929344
[4100]  train's auc: 0.947374   test's auc: 0.929485
[4200]  train's auc: 0.947682   test's auc: 0.929612
[4300]  train's auc: 0.947992   test's auc: 0.929722
[4400]  train's auc: 0.948312   test's auc: 0.929837
[4500]  train's auc: 0.948608   test's auc: 0.929947
[4600]  train's auc: 0.948905   test's auc: 0.930037
[4700]  train's auc: 0.949196   test's auc: 0.930129
[4800]  train's auc: 0.949505   test's auc: 0.930239
[4900]  train's auc: 0.949785   test's auc: 0.930347
[5000]  train's auc: 0.950079   test's auc: 0.930444
[5100]  train's auc: 0.950338   test's auc: 0.930553
[5200]  train's auc: 0.950591   test's auc: 0.930626
[5300]  train's auc: 0.950837   test's auc: 0.930716
[5400]  train's auc: 0.951074   test's auc: 0.930787
[5500]  train's auc: 0.95133    test's auc: 0.930874
[5600]  train's auc: 0.951561   test's auc: 0.930942
[5700]  train's auc: 0.951795   test's auc: 0.931013
[5800]  train's auc: 0.95201    test's auc: 0.931072
[5900]  train's auc: 0.952235   test's auc: 0.931134
[6000]  train's auc: 0.952487   test's auc: 0.931202
[6100]  train's auc: 0.952685   test's auc: 0.931258
[6200]  train's auc: 0.952896   test's auc: 0.931303
[6300]  train's auc: 0.953088   test's auc: 0.931355
[6400]  train's auc: 0.953306   test's auc: 0.931404
[6500]  train's auc: 0.95351    test's auc: 0.931448
[6600]  train's auc: 0.953733   test's auc: 0.931504
[6700]  train's auc: 0.953936   test's auc: 0.931545
[6800]  train's auc: 0.954131   test's auc: 0.931575
[6900]  train's auc: 0.954324   test's auc: 0.931618
[7000]  train's auc: 0.95453    test's auc: 0.931657
[7100]  train's auc: 0.954718   test's auc: 0.931699
[7200]  train's auc: 0.954878   test's auc: 0.931729
[7300]  train's auc: 0.955072   test's auc: 0.931755
[7400]  train's auc: 0.95525    test's auc: 0.931798
[7500]  train's auc: 0.955401   test's auc: 0.931831
[7600]  train's auc: 0.955597   test's auc: 0.931861
[7700]  train's auc: 0.95576    test's auc: 0.931893
[7800]  train's auc: 0.955929   test's auc: 0.93191
[7900]  train's auc: 0.956105   test's auc: 0.931941
[8000]  train's auc: 0.956299   test's auc: 0.931958
[8100]  train's auc: 0.956442   test's auc: 0.931999
[8200]  train's auc: 0.956608   test's auc: 0.932033
[8300]  train's auc: 0.956749   test's auc: 0.93205
[8400]  train's auc: 0.956918   test's auc: 0.932085
[8500]  train's auc: 0.957106   test's auc: 0.932106
[8600]  train's auc: 0.957254   test's auc: 0.932116
[8700]  train's auc: 0.957395   test's auc: 0.932132
[8800]  train's auc: 0.957578   test's auc: 0.932176
[8900]  train's auc: 0.957755   test's auc: 0.932202
[9000]  train's auc: 0.957925   test's auc: 0.932231
[9100]  train's auc: 0.958045   test's auc: 0.932246
[9200]  train's auc: 0.958214   test's auc: 0.932254
[9300]  train's auc: 0.958366   test's auc: 0.932271
[9400]  train's auc: 0.958491   test's auc: 0.932279
[9500]  train's auc: 0.958679   test's auc: 0.932304
[9600]  train's auc: 0.958853   test's auc: 0.93232
[9700]  train's auc: 0.959013   test's auc: 0.93235
[9800]  train's auc: 0.959135   test's auc: 0.932355
[9900]  train's auc: 0.959293   test's auc: 0.932361
[10000] train's auc: 0.959456   test's auc: 0.932365
[10100] train's auc: 0.959571   test's auc: 0.932376
[10200] train's auc: 0.959699   test's auc: 0.932383
[10300] train's auc: 0.959813   test's auc: 0.932388
[10400] train's auc: 0.959959   test's auc: 0.932398
[10500] train's auc: 0.960083   test's auc: 0.932408
[10600] train's auc: 0.960213   test's auc: 0.932431
[10700] train's auc: 0.96038    test's auc: 0.932422
Early stopping, best iteration is:
[10606] train's auc: 0.96022    test's auc: 0.932433



'''
