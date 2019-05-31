# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:28:36 2019

@author: Administrator
"""
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.svm import LinearSVR, SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingCVRegressor
from datetime import datetime

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

final_features = pd.read_csv("final_features.csv")  
train = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv")
test = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_test_complete.csv")

Id = test[['id']]
train_ = train[~train['happiness'].isin([-8])]
Y = train_[['happiness']]

#final_features = PCA(2).fit_transform(final_features)

X = final_features[:len(Y)]
x_test = final_features[len(Y):]




#sm = SMOTE(random_state = 42) 
#X,Y = sm.fit_sample(X,Y)
##
##Y = pd.DataFrame(Y)
##X = pd.DataFrame(X)
##plt.scatter(X[:,0], X[:,1], alpha = 0.1)
##plt.show()
##
#from sklearn.cluster import KMeans
#X_cluster = pd.concat([pd.DataFrame(X), pd.DataFrame(Y ,columns = ['happiness'])], axis=1, ignore_index = False)
##
#n_clusters = 5
#cluster = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
##
#y_pred = cluster.labels_
#centroid = cluster.cluster_centers_
##
#X_cluster = pd.concat([X_cluster, pd.DataFrame(y_pred, columns = ['y_pred'])], axis=1, ignore_index = False)
##
##
#color = ["pink","red","orange","gray",'blue','purple']
#fig, ax1 = plt.subplots(1)
#for i in range(n_clusters):   
#    ax1.scatter( X_cluster.loc[X_cluster['y_pred'] ==i][0]
#                ,X_cluster.loc[X_cluster['y_pred'] ==i][1]
##                ,marker='o'
#                ,s=10
#                ,c=color[i]
#                ,alpha = .7
#                )
####
#plt.show()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, shuffle=True, random_state=42)
######################lasso_feature####################


########################################
print('START ML', datetime.now(), )
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
# rmsle
def mse(y_, y_pred):
    return mean_squared_error(y_, y_pred)

# build our model scoring function
def cv_mse(model):
    rmse = -cross_val_score(model, X_train, Y_train,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds)
    return (rmse)
##
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
#
#
##
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds))
#
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))


svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=3000,
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
                       ,n_estimators=3000
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
                                 n_estimators = 3000
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
#
#score = cv_mse(lasso)
#print("lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )
#
#score = cv_mse(ridge)
#print("ridge score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )
#
#score = cv_mse(elasticnet)
#print("elasticnet score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )
#
#score = cv_mse(svr)
#print("linear_svr score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
##
#score = cv_mse(RandomForest)
#print("RandomForest score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
##
#score = cv_mse(GBR)
#print("GBR score: {:.4f} ({:.4f})".format(score.mean(), score.std()) )
#
#score = cv_mse(xgboost)
#print("Xgboost score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#
#score = cv_mse(lightgbm)
#print("lightgbm score: {:.4f} ({:.4f})".format(score.mean(), score.std()),  )

#########################################################################
print('START Fitting', datetime.now(), )

print(datetime.now(), 'Lasso')
lasso_model_full_data = lasso.fit(X_train, Y_train)

print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X_train, Y_train)

print(datetime.now(), 'elastic')
elastic_model_full_data = elasticnet.fit(X_train, Y_train)

print(datetime.now(), 'svr')
svr_model_full_data = svr.fit(X_train, Y_train)

print(datetime.now(), 'RandomForest')
RandomForest_model_full_data = RandomForest.fit(X_train, Y_train)
#
print(datetime.now(), 'GBR')
GBR_model_full_data = GBR.fit(X_train, Y_train)
#
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
            (0.1 * RandomForest_model_full_data.predict(X_train)) + \
            (0.1 * GBR_model_full_data.predict(X_train)) + \
            (0.2 * xgb_model_full_data.predict(X_train))   + \
            (0.2 * lightgbm_model_full_data.predict(X_train)) + \
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
    y_pred_val_blend = pd.DataFrame(blend_models_predict(X_val),columns = ['blend_val'] )



y_pred_test_blend = pd.DataFrame(blend_models_predict(x_test),columns = ['happiness'] )

y_pred_test_blend_fixed = y_pred_test_blend
y_pred_test_blend_fixed[y_pred_test_blend < 0.5] = -8
y_pred_test_blend_fixed[ y_pred_test_blend > 5] = 5
#   
##
#
#    
ID = test[['id']]
#SUB_RESULT = pd.concat([ ID, y_pred_test_blend], axis=1, ignore_index = False)
SUB_RESULT_fixed = pd.concat([ ID, y_pred_test_blend_fixed], axis=1, ignore_index = False)

#pd.DataFrame(SUB_RESULT).to_csv('comp_reg_sk_v5_.csv', index = 0, header = 1)   
pd.DataFrame(SUB_RESULT_fixed).to_csv('comp_reg_sk_v5_part2-5-31(2).csv', index = 0, header = 1)



#print('Blend with Top Kernals submissions', datetime.now(),)
#sub_1 = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv"")
#sub_2 = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv")#top4
#sub_3 = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv")#top4
#sub_4 = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\happniess_dig\happiness_train_complete.csv"")#top1
##sub_4 = pd.read_csv('../input/all-you-need-is-pca-lb-0-11421-top-4/submission.csv')
##sub_5 = pd.read_csv('../input/house-prices-solution-0-107-lb/submission.csv') # fork my kernel again)


#START ML 2019-05-31 12:27:32.760679
#lasso score: 0.4731 (0.0367)
#ridge score: 0.4772 (0.0388)
#elasticnet score: 0.4736 (0.0368)
#linear_svr score: 0.5311 (0.0404)
#RandomForest score: 0.4818 (0.0313)
#GBR score: 0.4925 (0.0389)
#Xgboost score: 0.4546 (0.0329)
#lightgbm score: 0.4568 (0.0344)
#START Fitting 2019-05-31 14:18:18.554508
#2019-05-31 14:18:18.554508 Lasso
#2019-05-31 14:18:23.216080 ridge
#2019-05-31 14:18:36.152888 elastic
#2019-05-31 14:19:05.211111 svr
#2019-05-31 14:19:33.854185 RandomForest
#2019-05-31 14:23:10.895136 GBR
#2019-05-31 14:23:38.414487 xgboost
#2019-05-31 14:30:40.170627 lightgbm
#2019-05-31 14:30:43.644829 StackingCVRegressor
#FINISH Fitting 2019-05-31 15:36:21.268284
#RMSLE score on train data:
#0.1433990065975023
#RMSLE score on val data:
#lasso 0.47700162073597685
#elastic 0.47693613105565874
#ridge 0.4792485567255294
#svr 0.5150388822660136
#rfr 0.48885529126067284
#gbr 0.4844100181093161
#xgb 0.4585262320240624
#lgbm 0.4569237872712435
#stack 0.46482501904418005
#blend 0.45165699839195583