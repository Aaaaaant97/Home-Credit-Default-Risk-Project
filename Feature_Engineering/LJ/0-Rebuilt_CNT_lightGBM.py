# %% [code]
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf


test_2_df = pd.read_csv('/Users/revelyn/Codefild/kaggle/data/application_test.csv')
test_1_df = pd.read_csv('kaggle/data/application_train.csv')
train_df= pd.read_csv('kaggle/data/previous_application.csv')

# test_2_df = pd.read_csv('/Users/revelyn/Codefild/kaggle/data/application_test.csv', nrows=10000)
# test_1_df = pd.read_csv('/Users/revelyn/Codefild/kaggle/data/application_train.csv', nrows=10000)
# train_df  = pd.read_csv('/Users/revelyn/Codefild/kaggle/data/previous_application.csv', nrows=20000)


train_df=train_df.dropna(axis=0,subset=['CNT_PAYMENT'])
train_df['CNT_PAYMENT'] = train_df['CNT_PAYMENT'].astype('int')




for i in [test_1_df, test_2_df, train_df]:
    i['screwratio1']=(i.AMT_CREDIT-i.AMT_GOODS_PRICE)/i.AMT_GOODS_PRICE
    i['screwratio2']=(i.AMT_CREDIT-i.AMT_GOODS_PRICE)/i.AMT_CREDIT
    i['saint_CNT']=i.AMT_CREDIT/i.AMT_ANNUITY
    i['angel_CNT']=i.AMT_GOODS_PRICE/i.AMT_ANNUITY
    i['simple_diff']=i.AMT_CREDIT-i.AMT_GOODS_PRICE

feats=['saint_CNT', 'AMT_ANNUITY', 'angel_CNT', 'AMT_GOODS_PRICE', 'screwratio2', 'screwratio1', 'AMT_CREDIT','simple_diff']
train_df=train_df.fillna(-1)


import lightgbm as lgb

clf = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=50,
    max_depth=11,
    min_split_gain=0.0222415,
    min_child_weight=39.3259775,
    n_jobs=4,          # 代替 nthread
    verbosity=-1       # 代替 silent/verbose
)

print('fitting')

# 使用训练集做 eval_set（如果有验证集，最好用验证集）
clf.fit(
    train_df[feats],
    train_df['CNT_PAYMENT'],
    eval_set=[(train_df[feats], train_df['CNT_PAYMENT'])],
    eval_metric='multi_logloss',
    callbacks=[
        lgb.early_stopping(50),     # 50轮没有提升就停
        lgb.log_evaluation(100)     # 每100轮打印一次日志
    ]
)

print('training done')



print('training on previous apps done')
for frame in [[test_1_df,'train'],[test_2_df,'test']]:
    test_df=frame[0]
    tag=frame[1]
    j=clf.predict_proba(test_df[feats], verbose= 500)
    test_df=test_df.fillna(-1)
    gc.collect()
    feature_importance_df = pd.DataFrame()
    sqsum=[]
    test_df['CERTAINTY']=0
    print(np.arange(0,j.shape[1]-1))
    print(j.shape)
    for k in np.arange(0,j.shape[1]-1):
            test_df['CNT_prob_'+str(k)]=j[:,k]
            test_df['CNT_prob_sq_'+str(k)]=test_df['CNT_prob_'+str(k)]*test_df['CNT_prob_'+str(k)]
            test_df['CERTAINTY']+=test_df['CNT_prob_sq_'+str(k)]

    predictions=pd.DataFrame()
    for k in np.arange(0,j.shape[1]-1):
            predictions[str(clf.classes_[k])]=j[:,k]
    predictions['best_guess']=predictions.idxmax(axis=1)
    predictions['best_guess']=predictions.best_guess.astype('int')
    test_df['REBUILT_CNT']=predictions['best_guess']
    print('starting the long, arduous task of computing interest rates')
    x=[]
    #must loop here - np.rate has a bug
    for i in range(0,len(test_df.index)):    

            x.append(
                    npf.rate(
                        test_df['REBUILT_CNT'][i],          # 期数 nper
                        test_df['AMT_ANNUITY'][i],      # 每期付款 pmt
                        -test_df['AMT_CREDIT'][i],      # 贷款本金 pv（负号表示支出）
                        0.0                             # fv (未来值)
                    )
                )
    test_df['RATE_CREDIT']=x
    del x
    x=[]
    #must loop here - np.rate has a bug
    for i in range(0,len(test_df.index)):    
            x.append(
            npf.rate(
                test_df['REBUILT_CNT'][i],           # 分期期数 nper
                test_df['AMT_ANNUITY'][i],       # 每期还款 pmt
                -test_df['AMT_GOODS_PRICE'][i],  # 商品价格 pv（记得加负号）
                0.0                              # 未来值 fv
            )
        )
    test_df['RATE_GOODS']=x
    del x
    test_df[['RATE_GOODS','SK_ID_CURR','REBUILT_CNT','RATE_CREDIT','CERTAINTY']].to_csv('Rebuilt_CNT_'+tag+'.csv', index= False)
# feature_importance_df = pd.DataFrame()
# feature_importance_df["feature"] = feats
# feature_importance_df["importance"] = clf.feature_importances_
# feature_importance_df[['feature', 'importance']].to_csv('importances.csv', index= False)

# def display_importances(feature_importance_df_):
#     cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
#     best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
#     plt.figure(figsize=(8, 10))
#     sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
#     plt.title('LightGBM Features (avg over folds)')
#     plt.tight_layout()
#     plt.savefig('lgbm_importances01.png')
    
# display_importances(feature_importance_df)








        

