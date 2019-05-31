# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:42:15 2019

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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVR, SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingCVRegressor
from datetime import datetime



warnings.filterwarnings('ignore')

def plt_data_na():
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0 ].index).sort_values(ascending = False)
    f, ax = plt.subplots(figsize = (15,21))
    plt.xticks(rotation = '90')
    sns.barplot(x = all_data_na.index, y = all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()

train = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv")
test = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_test_complete.csv")

all_data = pd.concat([train, test], axis=0, ignore_index = False)



#threshold = 0.1
#corr_matrix = train.corr().abs()
#drop_col=corr_matrix[corr_matrix['happiness']<threshold].index
#all_data.drop(drop_col,axis=1,inplace=True)

Id = test[['id']]
Y = train[['happiness']]
train = train.drop('happiness', axis = 1)
Y[Y < 0] = 0
#Y = (Y - Y.mean()) / Y.std()
#y_train = y_train.astype(str)
#y_train = pd.get_dummies(y_train)

all_data = pd.concat([train, test], axis=0, ignore_index = False)
all_data = all_data.drop('id', axis = 1)
all_data = all_data.drop('survey_time', axis =1)
all_data = all_data.drop('edu_other', axis =1)
all_data = all_data.drop('invest_other', axis =1)
all_data = all_data.drop('property_other', axis =1)
all_data = all_data.drop('join_party', axis =1)
all_data = all_data.drop('s_work_status', axis =1)
all_data = all_data.drop('work_status', axis =1)
all_data = all_data.drop('work_type', axis =1)
all_data = all_data.drop('s_work_type', axis =1)
all_data = all_data.drop('work_manage', axis =1)
all_data = all_data.drop('m_birth', axis =1)
all_data = all_data.drop('trust_11', axis =1)
all_data = all_data.drop('trust_12', axis =1)
#均值填补
work_yr = all_data[['work_yr']]
work_yr = work_yr.fillna(work_yr.mean())
all_data = all_data.drop('work_yr', axis = 1)
all_data = pd.concat([all_data, work_yr], axis=1, ignore_index = False)

edu_yr = all_data[['edu_yr']]
edu_yr = edu_yr.fillna(edu_yr.mean())
all_data = all_data.drop('edu_yr', axis = 1)
all_data = pd.concat([all_data, edu_yr], axis=1, ignore_index = False)

s_income = all_data[['s_income']]
s_income = s_income.fillna(s_income.mean())
all_data = all_data.drop('s_income', axis = 1)
all_data = pd.concat([all_data, s_income], axis=1, ignore_index = False)

s_birth = all_data[['s_birth']]
s_birth = s_birth.fillna(s_birth.mean())
all_data = all_data.drop('s_birth', axis = 1)
all_data = pd.concat([all_data, s_birth], axis=1, ignore_index = False)

minor_child = all_data[['minor_child']]
minor_child = minor_child.fillna(minor_child.mean())
all_data = all_data.drop('minor_child', axis = 1)
all_data = pd.concat([all_data, minor_child], axis=1, ignore_index = False)

marital_1st = all_data[['marital_1st']]
marital_1st = marital_1st.fillna(marital_1st.mean())
all_data = all_data.drop('marital_1st', axis = 1)
all_data = pd.concat([all_data, marital_1st], axis=1, ignore_index = False)

family_income = all_data[['family_income']]
family_income = family_income.fillna(family_income.mean())
all_data = all_data.drop('family_income', axis = 1)
all_data = pd.concat([all_data, family_income], axis=1, ignore_index = False)

#0填补
marital_now = all_data[['marital_now']]
marital_now = marital_now.fillna(0)
all_data = all_data.drop('marital_now', axis = 1)
all_data = pd.concat([all_data, marital_now], axis=1, ignore_index = False)

#众数填补
s_work_exper = all_data[['s_work_exper']]
s_work_exper_mode = (s_work_exper.mode()).iat[0,0]
s_work_exper = s_work_exper.fillna(s_work_exper_mode)
all_data = all_data.drop('s_work_exper', axis = 1)
all_data = pd.concat([all_data, s_work_exper], axis=1, ignore_index = False)

s_edu = all_data[['s_edu']]
s_edu_mode = (s_edu.mode()).iat[0,0]
s_edu = s_edu.fillna(s_edu_mode)
all_data = all_data.drop('s_edu', axis = 1)
all_data = pd.concat([all_data, s_edu], axis=1, ignore_index = False)

s_hukou = all_data[['s_hukou']]
s_hukou_mode = (s_hukou.mode()).iat[0,0]
s_hukou = s_hukou.fillna(s_hukou_mode)
all_data = all_data.drop('s_hukou', axis = 1)
all_data = pd.concat([all_data, s_hukou], axis=1, ignore_index = False)

s_political = all_data[['s_political']]
s_political_mode = (s_political.mode()).iat[0,0]
s_political = s_political.fillna(s_political_mode)
all_data = all_data.drop('s_political', axis = 1)
all_data = pd.concat([all_data, s_political], axis=1, ignore_index = False)

edu_status = all_data[['edu_status']]
edu_status_mode = (edu_status.mode()).iat[0,0]
edu_status = edu_status.fillna(edu_status_mode)
all_data = all_data.drop('edu_status', axis = 1)
all_data = pd.concat([all_data, edu_status], axis=1, ignore_index = False)

social_friend = all_data[['social_friend']]
social_friend_mode = (social_friend.mode()).iat[0,0]
social_friend = social_friend.fillna(edu_status_mode)
all_data = all_data.drop('social_friend', axis = 1)
all_data = pd.concat([all_data, social_friend], axis=1, ignore_index = False)

social_neighbor = all_data[['social_neighbor']]
social_neighbor_mode = (social_neighbor.mode()).iat[0,0]
social_neighbor = social_neighbor.fillna(edu_status_mode)
all_data = all_data.drop('social_neighbor', axis = 1)
all_data = pd.concat([all_data, social_neighbor], axis=1, ignore_index = False)

hukou_loc = all_data[['hukou_loc']]
hukou_loc_mode = (hukou_loc.mode()).iat[0,0]
hukou_loc = hukou_loc.fillna(edu_status_mode)
all_data = all_data.drop('hukou_loc', axis = 1)
all_data = pd.concat([all_data, hukou_loc], axis=1, ignore_index = False)

#b= all_data.isnull().sum()
#plt_data_na()
#
#all_data_mode = all_data.mode()
#all_data_mode = all_data_mode.iloc[[0],:]

#所有小于0的数替换为0，去除错误参数
all_data[all_data < 0] = 0
#添加特征
all_data['kids'] = (all_data['son'] + all_data['daughter'])


#
#one-hot
#取出one_hot处理列

one_hot_col = ['survey_type','province','city','county','gender','nationality',
                        'religion','political','hukou','work_exper','insur_1','insur_2','insur_3','insur_4'
                        ,'invest_0','invest_1','invest_2','invest_3','invest_4','invest_5','invest_6','invest_7','invest_8'
                        ,'marital','f_political','f_work_14','m_political','m_work_14'
                        ,'hukou_loc','edu_status','s_political','s_hukou','s_work_exper']

numerical_col = ['income','floor_area','height_cm','weight_jin','house','son','daughter','f_birth', 'inc_exp'
                 ,'public_service_1','public_service_2','public_service_3','public_service_4','public_service_5'
                 ,'public_service_6','public_service_7','public_service_8','public_service_9','work_yr','s_income'
                 ,'s_birth','minor_child', 'marital_1st', 'family_income','marital_now','edu_yr','kids','birth','religion_freq'
                 ,'edu','health','health_problem','depression','media_1','media_2','media_3','media_4','media_5','media_6'
                 ,'leisure_1','leisure_2','leisure_3','leisure_4','leisure_5','leisure_6','leisure_7','leisure_8'
                 ,'leisure_9','leisure_10','leisure_11','leisure_12','socialize','relax','learn','socia_outing'
                 ,'equity','class','class_10_before','class_10_after','class_14','family_m','family_status','car'
                 ,'f_edu','m_edu','status_peer','status_3_before', 'view' , 'inc_ability','trust_1','trust_2','trust_3'
                 ,'trust_4','trust_5','trust_6','trust_7','trust_8','trust_9','trust_10','trust_13','neighbor_familiarity'
                 ,'social_neighbor','social_friend','s_edu']


one_hot_col = all_data[one_hot_col]
numeric_cols = all_data[numerical_col]
rest_col = all_data.drop(one_hot_col, axis = 1)
rest_col = rest_col.drop(numerical_col, axis = 1)
#one_hot_col = pd.DataFrame(OneHotEncoder(categories = 'auto').fit_transform(one_hot_col).toarray())
one_hot_col = one_hot_col.astype(str)
one_hot_col_ = pd.get_dummies(one_hot_col)
#标准化
#numerical_col_std = pd.DataFrame(StandardScaler().fit_transform(numerical_col))
numeric_col_means = numeric_cols.mean()
numeric_col_std = numeric_cols.std()
numeric_cols = (numeric_cols - numeric_col_means) / numeric_col_std

#
final_features = pd.concat([one_hot_col_, numeric_cols, rest_col], axis=1, ignore_index = False)

#final_features = PCA(600).fit_transform(final_features)


X = final_features[:8000]
x_test = final_features[8000:]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, shuffle=True, random_state=42)

#########################################
print('START ML', datetime.now(), )
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
# rmsle
def mse(y_, y_pred):
    return mean_squared_error(y_, y_pred)

# build our model scoring function
def cv_mse(model):
    rmse = -cross_val_score(model, X_train, Y_train,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds)
    return (rmse)
#
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


#
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))


svr = LinearSVR(C = 0.18)

lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       #min_data_in_leaf=2,
                                       #min_sum_hessian_in_leaf=11
                                       )
    
RandomForest = RandomForestRegressor(
                                        n_estimators = 3000
                                        ,oob_score = True
#                                        ,n_jobs = -1
                                        ,random_state = 90
                                        ,max_depth = 10
                                        ,max_features = 120
                                        ,min_samples_leaf = 1  #1
                                        ,min_samples_split = 2  #3
                                    ) #

xgboost = XGBRegressor(
                       learning_rate=0.01
                       ,n_estimators=5000
                       ,max_depth=6
                       , min_child_weight=0
    #                   ,gamma=0
                       ,subsample=0.7          
                       ,colsample_bytree=0.7  
                       ,objective='reg:linear'
    #                   ,nthread=-1
                       ,scale_pos_weight=1
                       ,seed=27              
    #                   ,reg_alpha=0.00006
                       )


GBR = GradientBoostingRegressor(
                                 n_estimators = 5000
                                ,random_state = 90
                                ,max_depth = 5        
                                ,max_features = 20    
                                )


stack_gen = StackingCVRegressor(regressors=(
                                             lasso
                                            ,elasticnet
                                            ,lightgbm
                                            ,svr
                                            ,GBR
                                            ,xgboost
                                            ,RandomForest
                                            
                                            )
                                            ,meta_regressor=xgboost
                                            ,use_features_in_secondary=True
                                            )   

score = cv_mse(lasso)
print("lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )

score = cv_mse(ridge)
print("ridge score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )

score = cv_mse(elasticnet)
print("elasticnet score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )

score = cv_mse(svr)
print("linear_svr score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

score = cv_mse(RandomForest)
print("RandomForest score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

score = cv_mse(GBR)
print("GBR score: {:.4f} ({:.4f})".format(score.mean(), score.std()) )

score = cv_mse(xgboost)
print("Xgboost score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

score = cv_mse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )

#########################################################################
print('START Fitting', datetime.now(), )

print(datetime.now(), 'Lasso')
lasso_model_full_data = lasso.fit(X_train, Y_train)

print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X_train, Y_train)

print(datetime.now(), 'elastic')
elastic_model_full_data = elasticnet.fit(X_train, Y_train)

print(datetime.now(), 'linear_svr')
svr_model_full_data = svr.fit(X_train, Y_train)

print(datetime.now(), 'RandomForest')
RandomForest_model_full_data = RandomForest.fit(X_train, Y_train)

print(datetime.now(), 'GBR')
GBR_model_full_data = GBR.fit(X_train, Y_train)

print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(X_train, Y_train)

print(datetime.now(), 'lightgbm')
lightgbm_model_full_data = lightgbm.fit(X_train, Y_train)

print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(np.array(X_train),np.array(Y_train))

print('FINISH Fitting', datetime.now(), )
######################################################################


def blend_models_predict(X_train):
    return (
            (0.05 * elastic_model_full_data.predict(X_train)) + \
            (0.05 * lasso_model_full_data.predict(X_train))  + \
#            (0.1 * ridge_model_full_data.predict(X_train))  + \
#            (0.1 * svr_model_full_data.predict(X_train)) + \
            (0.05 * RandomForest_model_full_data.predict(X_train)) + \
            (0.05 * GBR_model_full_data.predict(X_train)) + \
            (0.2 * xgb_model_full_data.predict(X_train))   + \
            (0.3 * lightgbm_model_full_data.predict(X_train)) + \
            (0.3* stack_gen_model.predict(np.array(X_train)))
            )
print('blend',mse(Y_val, blend_models_predict(X_val))) 

print('RMSLE score on train data:')
print(mse(Y_train, blend_models_predict(X_train)))  

if len(X_val)>0: 
    print('RMSLE score on val data:')
    print('lasso',mse(Y_val, lasso_model_full_data.predict(X_val)))
    print('elastic',mse(Y_val, elastic_model_full_data.predict(X_val)))
    print('ridge',mse(Y_val, ridge_model_full_data.predict(X_val)))
    print('svr',mse(Y_val, svr_model_full_data.predict(X_val)))
    print('rfr',mse(Y_val, RandomForest_model_full_data.predict(X_val)))
    print('gbr',mse(Y_val, GBR_model_full_data.predict(X_val)))
    print('xgb',mse(Y_val, xgb_model_full_data.predict(X_val)))
    print('lgbm',mse(Y_val, lightgbm_model_full_data.predict(X_val)))
    print('stack',mse(Y_val, stack_gen_model.predict(np.array(X_val))))
    print('blend',mse(Y_val, blend_models_predict(X_val)))  


y_pred_test_lasso = pd.DataFrame(lasso_model_full_data.predict(x_test), columns = ['lasso'])
y_pred_test_ridge = pd.DataFrame(ridge_model_full_data.predict(x_test), columns = ['ridge'])
y_pred_test_elastic = pd.DataFrame(elastic_model_full_data.predict(x_test), columns = ['elastic'])
y_pred_test_svr = pd.DataFrame(svr_model_full_data.predict(x_test), columns = ['svr'])
y_pred_test_rfr = pd.DataFrame(RandomForest_model_full_data.predict(x_test), columns = ['rfr'])
y_pred_test_gbr = pd.DataFrame(GBR_model_full_data.predict(x_test), columns = ['gbr'])
y_pred_test_xgb = pd.DataFrame(xgb_model_full_data.predict(x_test), columns = ['xgb'])
y_pred_test_lightgbm = pd.DataFrame(lightgbm_model_full_data.predict(x_test), columns = ['lgbm'])
y_pred_test_stack = pd.DataFrame(stack_gen_model.predict(np.array(x_test)), columns = ['stack'])
y_pred_test_blend = pd.DataFrame(blend_models_predict(x_test),columns = ['blend_'] )

y_pred_val_blend = pd.DataFrame(blend_models_predict(X_val),columns = ['blend_val'] )
all_results = pd.concat([y_pred_test_lasso
                         ,y_pred_test_ridge
                         ,y_pred_test_elastic                        
                         ,y_pred_test_svr
                         ,y_pred_test_rfr
                         ,y_pred_test_gbr
                         ,y_pred_test_xgb
                         ,y_pred_test_lightgbm
                         ,y_pred_test_stack
                         ,y_pred_test_blend]
                         ,axis=1
                         ,ignore_index = False)

#fixed = []
#for i in range(len(X_val)):
#    a = y_pred_val_blend.iloc[i,0]
#    if  a<1.9:
#        a = 0
##    elif 2.2<a<2.5:
##        a = 1
##    elif 3.1<a<3.6:
##        a = 3
##    elif 3.6<a<4.03:
##        a = 4
#    fixed.append(a)
#y_pred_val_blend_fixed = pd.DataFrame(fixed, columns = ['fixed_blend'] )
##y_pred_val_blend_fixed[y_pred_test_blend < 0.5] = -8
##y_pred_val_blend_fixed[ y_pred_test_blend > 4.1] = 5
#print(mse(Y_val, y_pred_val_blend))  
#print(mse(Y_val, y_pred_val_blend_fixed))     
#VAL_ = pd.concat([ Y_val, y_pred_test_blend, y_pred_val_blend_fixed], axis=1, ignore_index = False)
#
#fixed = []
#for i in range(len(x_test)):
#    a = y_pred_val_blend.iloc[i,0]
#    if  a<1.9:
#        a = 0
##        elif 1.9<a<2.5:
##            a = 1
##        elif 2.97<a<3.03:
##            a = 3
##        elif 3.97<a<4.03:
##            a = 4
#    fixed.append(a)
#y_pred_test_blend_fixed = pd.DataFrame(fixed, columns = ['fixed_blend'] )

#Y = (Y - Y.mean()) / Y.std()
y_pred_test_blend_fixed = y_pred_test_blend 
y_pred_test_blend_fixed[y_pred_test_blend < 0.5] = -8
y_pred_test_blend_fixed[ y_pred_test_blend > 5] = 5
#   
#

    
ID = test[['id']]
SUB_RESULT = pd.concat([ ID, y_pred_test_blend], axis=1, ignore_index = False)
SUB_RESULT_fixed = pd.concat([ ID, y_pred_test_blend_fixed], axis=1, ignore_index = False)

pd.DataFrame(SUB_RESULT).to_csv('comp_reg_sk_v3_80%train_unfixed.csv', index = 0, header = 1)   
#pd.DataFrame(SUB_RESULT_fixed).to_csv('comp_reg_sk_v2_98%train_fixed.csv', index = 0, header = 1)






