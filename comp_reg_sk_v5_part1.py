# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:42:15 2019




v5 只用Xgb 和 lgbm

@author: Administrator
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer,mean_squared_error

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.svm import LinearSVR, SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingCVRegressor
from datetime import datetime

#from imblearn.over_sampling import SMOTE

from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

#def plt_data_na():
#    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
#    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0 ].index).sort_values(ascending = False)
#    f, ax = plt.subplots(figsize = (15,21))
#    plt.xticks(rotation = '90')
#    sns.barplot(x = all_data_na.index, y = all_data_na)
#    plt.xlabel('Features', fontsize=15)
#    plt.ylabel('Percent of missing values', fontsize=15)
#    plt.title('Percent missing data by feature', fontsize=15)
#    plt.show()
#
#def mcoor(train):    
#    plt.figure(figsize=(100, 64))  # 指定绘图对象宽度和高度
#    colnm = train.columns.tolist()  # 列表头
#    mcorr = train[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
#    mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
#    mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
#    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
#    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
#    plt.show()

train = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv")
test = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_test_complete.csv")


Id = test[['id']]

train_ = train[~train['happiness'].isin([-8])]
Y = train_[['happiness']]
train_ = train_.drop('happiness', axis = 1)

all_data = pd.concat([train_, test], axis=0, ignore_index = True)
all_data = all_data.drop('id', axis = 1)
all_data = all_data.drop('survey_time', axis =1)
all_data = all_data.drop('edu_other', axis =1)
all_data = all_data.drop('invest_other', axis =1)
all_data = all_data.drop('join_party', axis =1)
all_data = all_data.drop('s_work_status', axis =1)
all_data = all_data.drop('s_work_type', axis =1)
all_data = all_data.drop('work_type', axis =1)
all_data = all_data.drop('work_yr', axis =1)
all_data = all_data.drop('work_status', axis =1)
all_data = all_data.drop('work_manage', axis =1)
all_data = all_data.drop('property_other', axis =1)



##缺失值排序图
#b= all_data.isnull().sum()
#plt_data_na()


#所有小于0的数替换为nan
all_data[all_data < 0] = np.nan

# 取出nan列
all_data_na = all_data.isnull().sum()
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0 ].index).sort_values(ascending = True)
na_index_name = all_data_na.index.tolist()

#添加特征

one_hot_col = ['survey_type','province','city','county','gender','nationality',
                        'religion','political','hukou','work_exper','insur_1','insur_2','insur_3','insur_4'
                        ,'invest_0','invest_1','invest_2','invest_3','invest_4','invest_5','invest_6','invest_7','invest_8'
                        ,'marital','f_political','f_work_14','m_political','m_work_14'
                        ,'hukou_loc','edu_status','s_political','s_hukou','s_work_exper']
#
numerical_col = [ 'income','floor_area','height_cm','weight_jin','house','son','daughter','f_birth', 'inc_exp'
                 ,'public_service_1','public_service_2','public_service_3','public_service_4','public_service_5'
                 ,'public_service_6','public_service_7','public_service_8','public_service_9','s_income'
                 ,'s_birth','minor_child', 'marital_1st', 'family_income','marital_now','edu_yr','birth','religion_freq'
                 ,'edu','health','health_problem','depression','media_1','media_2','media_3','media_4','media_5','media_6'
                 ,'leisure_1','leisure_2','leisure_3','leisure_4','leisure_5','leisure_6','leisure_7','leisure_8'
                 ,'leisure_9','leisure_10','leisure_11','leisure_12','socialize','relax','learn','socia_outing'
                 ,'equity','class','class_10_before','class_10_after','class_14','family_m','family_status','car'
                 ,'f_edu','m_edu','status_peer','status_3_before', 'view' , 'inc_ability','trust_1','trust_2','trust_3'
                 ,'trust_4','trust_5','trust_6','trust_7','trust_8','trust_9','trust_10','trust_11','trust_12','trust_13','neighbor_familiarity'
                 ,'social_neighbor','social_friend','s_edu','m_birth']
#                 ,'kids','income_diff','fit','income/edu','income*floor_area','income*health','positive_level','edu_diff']
                
na_index_one_hot_col = [i for i in na_index_name if i in one_hot_col]
na_index_num_col = [i for i in na_index_name if i in numerical_col]

one_hot_cols = all_data[one_hot_col]
numeric_cols = all_data[numerical_col]

#随机森林填补

for i in range(len(na_index_one_hot_col)):
    df = all_data
    fillc = all_data[[na_index_one_hot_col[i]]]
    df_ = all_data.drop(na_index_one_hot_col[i], axis =1)

    df_0 = df_.fillna(df.mean())
    #现在的标签为 当前要填充列中的 非空值
    Ytrain = fillc.dropna()
    # 当前要填充列的空值
    Ytest = fillc[fillc.isnull().values == True]
    test_index = Ytest.index
    #
    Xtrain = df_0.loc[Ytrain.index]
    Xtest = df_0.loc[Ytest.index]
    #
    rfc = RandomForestClassifier(n_estimators = 100)
    rfc = rfc.fit(Xtrain, Ytrain)
    
    Ypredict = rfc.predict(Xtest)
    Ypredict = pd.DataFrame(Ypredict, index = test_index, columns = [na_index_one_hot_col[i]])
    
    fillc_new = Ytrain.append(Ypredict).sort_index(axis = 0)
    all_data = all_data.drop(na_index_one_hot_col[i], axis =1)
    all_data = pd.concat([all_data, fillc_new], axis = 1, ignore_index = False)
    print(len(na_index_one_hot_col) - i-1, 'left')
    
    
for i in range(len(na_index_num_col)):
    df = all_data
    fillc = all_data[[na_index_num_col[i]]]
    df_ = all_data.drop(na_index_num_col[i], axis =1)

    df_0 = df_.fillna(df.mean())
    #现在的标签为 当前要填充列中的 非空值
    Ytrain = fillc.dropna()
    # 当前要填充列的空值
    Ytest = fillc[fillc.isnull().values == True]
    test_index = Ytest.index
    #
    Xtrain = df_0.loc[Ytrain.index]
    Xtest = df_0.loc[Ytest.index]
    #
    rfr = RandomForestRegressor(n_estimators = 100)
    rfr = rfr.fit(Xtrain, Ytrain)
    
    Ypredict = rfr.predict(Xtest)
    Ypredict = pd.DataFrame(Ypredict, index = test_index, columns = [na_index_num_col[i]])
    
    fillc_new = Ytrain.append(Ypredict).sort_index(axis = 0)
    all_data = all_data.drop(na_index_num_col[i], axis =1)
    all_data = pd.concat([all_data, fillc_new], axis = 1, ignore_index = False)
    print(len(na_index_num_col) - i-1, 'left')

#
pd.DataFrame(all_data).to_csv('all_data_na_filled.csv', index = 0, header = 1) 

#all_data = pd.read_csv('all_data_na_filled.csv')

#添加特征
all_data['kids'] = (all_data['son'] + all_data['daughter'])
all_data['income_diff'] = all_data['income'] - all_data['s_income']
all_data['fit'] = all_data['weight_jin'] / all_data['height_cm']
all_data['income_edu'] = all_data['income'] / all_data['edu']
all_data['income*floor_area'] = all_data['income'] * all_data['floor_area']
all_data['income*health'] = all_data['income'] * all_data['health']
all_data['positive_level'] = all_data['class_10_after'] - all_data['class']
all_data['edu_diff'] = all_data['edu'] - all_data['s_edu']


one_hot_col = ['survey_type','province','city','county','gender','nationality',
                'religion','political','hukou','work_exper','insur_1','insur_2','insur_3'
                ,'insur_4','invest_0','invest_1','invest_2','invest_3','invest_4','invest_5'
                ,'invest_6','invest_7','invest_8','marital','f_political','f_work_14',
                'm_political','m_work_14','hukou_loc','edu_status','s_political','s_hukou'
                ,'s_work_exper']
#
numerical_col = [ 'income','floor_area','height_cm','weight_jin','house','son','daughter','f_birth', 'inc_exp'
                 ,'public_service_1','public_service_2','public_service_3','public_service_4','public_service_5'
                 ,'public_service_6','public_service_7','public_service_8','public_service_9','s_income'
                 ,'s_birth','minor_child', 'marital_1st', 'family_income','marital_now','edu_yr','birth','religion_freq'
                 ,'edu','health','health_problem','depression','media_1','media_2','media_3','media_4','media_5','media_6'
                 ,'leisure_1','leisure_2','leisure_3','leisure_4','leisure_5','leisure_6','leisure_7','leisure_8'
                 ,'leisure_9','leisure_10','leisure_11','leisure_12','socialize','relax','learn','socia_outing'
                 ,'equity','class','class_10_before','class_10_after','class_14','family_m','family_status','car'
                 ,'f_edu','m_edu','status_peer','status_3_before', 'view' , 'inc_ability','trust_1','trust_2','trust_3'
                 ,'trust_4','trust_5','trust_6','trust_7','trust_8','trust_9','trust_10','trust_11','trust_12','trust_13'
                 ,'neighbor_familiarity','social_neighbor','social_friend','s_edu','m_birth'
                 ,'kids','income_diff','fit','income_edu','income*floor_area','income*health','positive_level','edu_diff']
                


one_hot_cols = all_data[one_hot_col]
numeric_cols = all_data[numerical_col]

rest_cols = all_data.drop(one_hot_col, axis = 1)
rest_cols = rest_cols.drop(numerical_col, axis = 1)

#one_hot
one_hot_cols = one_hot_cols.astype(str)
one_hot_cols_ = pd.get_dummies(one_hot_cols)

#标准化
numeric_col_means = numeric_cols.mean()
numeric_col_std = numeric_cols.std()
numeric_cols = (numeric_cols - numeric_col_means) / numeric_col_std
#
##
final_features = pd.concat([one_hot_cols_, numeric_cols, rest_cols], axis=1, ignore_index = False)


#不要轻易覆盖之前版本
pd.DataFrame(final_features).to_csv('final_features.csv', index = 0, header = 1)   
   




