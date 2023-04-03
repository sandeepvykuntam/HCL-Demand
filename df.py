#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import sklearn
# from scikit-learn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import os
from datetime import timedelta, datetime, date
from pandas import to_datetime
from prophet import Prophet
import warnings
import holidays
import statistics
import random
import pickle


reg = pickle.load(open('reg.pkl', 'rb'))
m_holi = pickle.load(open('m_holi.pkl', 'rb'))
model = pickle.load(open('lstm.pkl', 'rb'))

scaler_x = joblib.load('scaler_x')

scaler_y = joblib.load('scaler_y')

def data_process(data):
    data = data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
    data['Date'] = pd.to_datetime(data['date'])
    data['month'] = data.Date.dt.month
    data['Year'] = data.Date.dt.year
    data['dayofweek'] = data.Date.dt.day
    data['quarter'] = data['Date'].apply(lambda x : x.quarter)

    def weather(mon):
        if mon<11:
            return random.randint(10, 25)
        else:
            return random.randint(-5, 8)

    data['Weather'] = data['month'].apply(weather)
    data =data.drop(['date','value'],axis=1)
    data2 = data[(data['store'] ==1) & (data['item']== 216418)]
    return data2

def data_preprocessing_xgb(data2):
    data3 = data2[['Year', 'month', 'dayofweek', 'quarter','Sales_New', 'promotion', 'Weather', 'holiday']]
    predict_xgb = data3.drop(['Sales_New'],axis=1)
    y_xgb = data3['Sales_New']
    return predict_xgb,y_xgb
    
def data_preprocessing_lstm(data2):
    data4 = data2[['Year', 'month', 'dayofweek', 'quarter','Sales_New', 'promotion', 'Weather', 'holiday']]
    x_lstm = data4.drop(['Sales_New'],axis=1)
    y_lstm = data4.filter(['Sales_New'])
    x_scaled = scaler_x.transform(x_lstm)
    x_scaled = np.array(x_scaled)
    x_scaled = np.reshape(x_scaled, (x_scaled.shape[0], x_scaled.shape[1], 1))
    return x_scaled
    
def data_preprocessing_fb(data2):
    train_fb = data2.filter(["Date"])
    train_fb.rename(columns ={'Date' : 'ds'}, inplace = True)
    train_fb = train_fb.reset_index(drop=True)
    return train_fb

def ensemble_pred(data_pred):
    predict_xgb,y_xgb = data_preprocessing_xgb(data_pred)
    sales_pred_xgb = reg.predict(predict_xgb)
    
    x_scaled = data_preprocessing_lstm(data_pred)
    predictions = model.predict(x_scaled)
    sales_pred_lstm = scaler_y.inverse_transform(predictions)
    
    train_fb = data_preprocessing_fb(data_pred)
    sales_pred_fb = m_holi.predict(train_fb)
    
    ensemble_pred = data_pred.filter(['Date','Sales_New'])
    ensemble_pred= ensemble_pred.rename(columns={'Sales_New':'Actual_sales'})
    ensemble_pred['sales_pred_xgb'] = sales_pred_xgb
    ensemble_pred['sales_pred_fb'] = sales_pred_fb['yhat'].values
    ensemble_pred['sales_pred_lstm'] = sales_pred_lstm
    ensemble_pred['predicted_sales'] = ensemble_pred.iloc[:,2:5].mean(axis=1)
    return ensemble_pred


uploaded_file = 'dec2022_data.csv'

df = pd.read_csv(uploaded_file)
ensemble_pred1=ensemble_pred(df)

predic = ensemble_pred1.set_index('Date')
_ = predic[['Actual_sales','predicted_sales']].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("sales")
plt.legend()
plt.show()

