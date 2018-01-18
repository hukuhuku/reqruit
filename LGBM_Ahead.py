import pandas as pd
import numpy as np

from datetime import datetime,date

import lightgbm as lgb

from sklearn import *

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

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

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

wkend_holidays = data["hol"].apply(lambda x: (x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1, axis=1)
data["hol"].loc[wkend_holidays, 'holiday_flg'] = 0

data["tra"]["visit_date"] = pd.to_datetime(data["tra"]["visit_date"])
data["tes"]["visit_date"] = pd.to_datetime(data["tes"]["visit_date"])
data["hol"]["visit_date"] = pd.to_datetime(data["hol"]["visit_date"])

del(data["hol"]["day_of_week"])

data["tra"] = pd.merge(data["tra"],data["hol"],on="visit_date")
data["tes"] = pd.merge(data["tes"],data["hol"],on="visit_date")
data["tra"]["visit_date"] = pd.to_datetime(data["tra"]["visit_date"])
data["tes"]["visit_date"] = pd.to_datetime(data["tes"]["visit_date"])

tmp = data["tra"].groupby(["air_store_id"])[["visitors"]].mean()
tmp = tmp.reset_index()
tmp.columns = ["air_store_id","meanvisitors"]

data["tra"] = pd.merge(data["tra"],tmp,how="outer",on="air_store_id")
data["tes"] = pd.merge(data["tes"],tmp,how="left",on="air_store_id")

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']).dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime']).dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data["ar"]["visit_date"] = pd.to_datetime(data["ar"]["visit_date"])

data["tra"] = pd.merge(data["tra"],data["ar"],on=["air_store_id","visit_date"])
data["tes"] = pd.merge(data["tes"],data["ar"],on=["air_store_id","visit_date"])

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

#Model Fitting
gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=60,
    learning_rate=0.01,
    n_estimators=10000)

train_x = data["tra"][data["tra"]["visit_date"] < datetime(2017,4,10)].drop(['air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)
train_y = np.log1p(data["tra"][data["tra"]["visit_date"] < datetime(2017,4,10)]['visitors'].values)

val_x = data["tra"][data["tra"]["visit_date"] > datetime(2017,4,10)].drop(['air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)
val_y = np.log1p(data["tra"][data["tra"]["visit_date"] > datetime(2017,4,10)]['visitors'].values)

print(train_x.shape, train_y.shape)
test_x = data["tes"].drop(['id', 'air_store_id', 'visit_date', 'visitors',"year_mth"], axis=1)

gbm.fit(train_x, train_y, eval_metric='rmse')

#valdation test
predict_val = gbm.predict(val_x)
print("Validation _Test_Result",RMSLE(val_y,predict_val))#0.399025630854
predict_y = gbm.predict(test_x)

data["tes"]['visitors'] = np.expm1(predict_y)
data["tes"][['id', 'visitors']].to_csv(
    'gbm_submission5.csv', index=False, float_format='%.3f')

