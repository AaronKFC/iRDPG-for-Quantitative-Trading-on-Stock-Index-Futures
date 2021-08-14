# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:24:45 2021

@author: Yoga
"""

import os 
import pandas as pd
import talib

cwd = os.getcwd()
# df =  pd.read_csv(cwd + '/raw_data/IF_2015to2018.csv',parse_dates=True,index_col=0) 
df =  pd.read_csv(cwd + '/raw_data/IC_2015to2018.csv',parse_dates=True,index_col=0) 

#加入return as column
dfReturn=pd.DataFrame((df['close']-df['open'])/df['open'],columns=['return'])
df=pd.concat([df,dfReturn],axis=1)

#加入techinical index 
# see all talib technical func: talib.get_functions()
macdhist = talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)[2]
EMA_7 = talib.EMA(df.close,timeperiod=7)
EMA_21 = talib.EMA(df.close,timeperiod=21)
EMA_56 = talib.EMA(df.close,timeperiod=56)
RSI = talib.RSI(df.close,timeperiod=56)
BB_up = talib.BBANDS(df.close)[0]
BB_mid = talib.BBANDS(df.close)[1]
BB_low = talib.BBANDS(df.close)[2]
slowK = talib.STOCH(df.high,df.low,df.close)[0]
slowD = talib.STOCH(df.high,df.low,df.close)[1]
df['MACD']=macdhist
df['EMA_7']=EMA_7
df['EMA_21']=EMA_21
df['EMA_56']=EMA_56
df['RSI']=RSI
df['BB_up']=BB_up
df['BB_mid']=BB_mid
df['BB_low']=BB_low
df['slowK']=slowK
df['slowD']=slowD

df.dropna(inplace=True)
# df.to_csv('IF_tech.csv') #相對位置，保存在getwcd()獲得的路徑下
df.to_csv('IC_tech.csv') #相對位置，保存在getwcd()獲得的路徑下


