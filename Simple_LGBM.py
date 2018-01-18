import pandas as pd
import numpy as np

from datetime import datetime,date

import lightgbm as lgb
from sklearn import *


#Loading Data
data = {
    'tra': pd.read_csv('./input/air_visit_data.csv'),
    'as': pd.read_csv('./input/air_store_info.csv'),
    'hs': pd.read_csv('./input/hpg_store_info.csv'),
    'ar': pd.read_csv('./input/air_reserve.csv'),
    'hr': pd.read_csv('./input/hpg_reserve.csv'),
    'id': pd.read_csv('./input/store_id_relation.csv'),
    'tes': pd.read_csv("./input/sample_submission.csv"),
    'hol': pd.read_csv('./input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

#Feature Engineering
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['day'] = data['tra']['visit_date'].dt.day
data['tra']['dow'] = data['tra']['visit_date'].dt.weekday
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['week'] = data['tra']['visit_date'].dt.week
data['tra']['quarter'] = data['tra']['visit_date'].dt.quarter
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
data['tra']['year_mth'] = data['tra']['year'].astype(str)+'-'+data['tra']['month'].astype(str)

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['day'] = data['tes']['visit_date'].dt.day
data['tes']['dow'] = data['tes']['visit_date'].dt.weekday
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['week'] = data['tes']['visit_date'].dt.week
data['tes']['quarter'] = data['tes']['visit_date'].dt.quarter
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
data['tes']['year_mth'] = data['tes']['year'].astype(str)+'-'+data['tes']['month'].astype(str)

tmp = data["tra"].groupby(["air_store_id"])[["visitors"]].mean()
tmp = tmp.reset_index()
tmp.columns = ["air_store_id","meanvisitors"]

data["tra"] = pd.merge(data["tra"],tmp,how="outer",on="air_store_id")
data["tes"] = pd.merge(data["tes"],tmp,how="left",on="air_store_id")


#Preparing Datasets
data["tra"]["visit_date"] = pd.to_datetime(data["tra"]["visit_date"])
data["tes"]["visit_date"] = pd.to_datetime(data["tes"]["visit_date"])

train_x = data["tra"][data["tra"]["visit_date"] < datetime(2017,4,10)].drop(['air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)
train_y = np.log1p(data["tra"][data["tra"]["visit_date"] < datetime(2017,4,10)]['visitors'].values)

val_x = data["tra"][data["tra"]["visit_date"] > datetime(2017,4,10)].drop(['air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)
val_y = np.log1p(data["tra"][data["tra"]["visit_date"] > datetime(2017,4,10)]['visitors'].values)

print("train_data_shape",train_x.shape, train_y.shape)
test_x = data["tes"].drop(['id', 'air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)

#Training and predict Models

gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=60,
    learning_rate=0.01,
    n_estimators=10000)

gbm.fit(train_x, train_y, eval_metric='rmse')
predict_y = gbm.predict(test_x)
data["tes"]['visitors'] = np.expm1(predict_y)


#Validation Test

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

predict_val = gbm.predict(val_x)
print("Validation _Test_Result",RMSLE(val_y,predict_val))
predict_y = gbm.predict(test_x)


data["tes"][['id', 'visitors']].to_csv(
    'gbm_submission.csv', index=False, float_format='%.3f')

print("Done")