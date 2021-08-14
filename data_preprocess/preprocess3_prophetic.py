# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:51:17 2021

@author: AllenPC
"""
import argparse
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta,datetime
from tqdm import tqdm

def Prophetic(file):  #原佳瑋的版本
    #read file
    df=pd.read_csv(file,parse_dates=True,index_col=0)
        
    # print("techinical index not defined")
    '''
    concate technical index
    '''
    
    #concate prophetic expert action
    df = pd.concat([df,pd.Series(name = 'phtAction', dtype = object)],axis=1) #add column 'phtAction'
    calendar = np.unique(df.index.date) #list all the trading date
    for i in tqdm(range(len(calendar))): #for each trading day
        if i == calendar.size-1:
            mask = df.index >= str(today)
        else:
            today = calendar[i]
            tomorrow = calendar[i+1]
            mask = (df.index >= str(today)) & (df.index < str(tomorrow)) #pick today's data
        
        todayMarket = df[mask]['open'] #extract today's open price
        phtAction = todayMarket.copy().rename("phtAction") #to record today's prophetic expert action
        #assign prophetic expert action, here are the possible patterns
        # pattern schematic:           if block                         else block                 
        #        11         |-------<-L->-------<-H->-----| or |-------<-H->-------<-L->-----|
        #        10         |-------<-L->---------------<-H or |-------<-H->---------------<-L
        #        01         L->-----------------<-H->-----| or H->-----------------<-L->-----|
        #        00         L->-------------------------<-H or H->-------------------------<-L
        if todayMarket.argmin() < todayMarket.argmax(): #(↘)↗(↘) lowest price appears earlier than highest price
            if todayMarket.argmin() != 0:#↘↗(↘) lowest price doesn't appear in the beginning
                phtAction[:todayMarket.argmin()] = -1 #short
            phtAction[todayMarket.argmin():todayMarket.argmax()] = 1 #long
            phtAction[todayMarket.argmax():] = -1 #short
        
        else: #(↗)↘(↗) highest price appears earlier than lowest price
            if todayMarket.argmax() != 0:#↗↘(↗) highest price doesn't appear in the beginning
                phtAction[:todayMarket.argmax()] = 1 #long
            phtAction[todayMarket.argmax():todayMarket.argmin()] = -1 #short
            phtAction[todayMarket.argmin():] = 1 #long
        df.update(phtAction) #save result
        #check if today's prophetic expert action is correct or not
        #print(df[str(today)][['open','phtAction']].to_string())
    
    ### store to .csv file
    # df.to_csv('prophetic_0616.csv',index=False)
    df.to_csv('IC_prophetic.csv',index=False)
            
    
    #depulicate unnormalized data to create initial normalized column
    # nColumns=len(df.columns)
    # for i in range(nColumns):
    #     print(df.columns[i])
    #     newLabel = "norm{}".format(df.columns[i].capitalize())
    #     df[newLabel] = df[df.columns[i]]
    
    # #normalize到[-1,1], 抓進來後再normalized
    # df[df.columns[nColumns:]]=2*((df[df.columns[nColumns:]] - df[df.columns[nColumns:]].min()) / (df[df.columns[nColumns:]].max() - df[df.columns[nColumns:]].min())) -1 #https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    
    nColumns=['open','close','high','low','volume',
              'MACD','EMA_7','EMA_21','EMA_56','RSI',
              'BB_up','BB_mid','BB_mid','BB_low','slowK','slowD']
    for cn in nColumns:
        newLabel = "norm{}".format(cn.capitalize())
        df[newLabel] = df[cn]
        df[newLabel] = 2*((df[newLabel] - df[newLabel].min()) / (df[newLabel].max() - df[newLabel].min())) -1
        
    #deal with the datetime column
    df=df.reset_index()#['date']=pd.to_datetime(df['date'])
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df


# fn = 'out_tech_DT_Prophetic_0606.csv'
# fn = 'IF_tech_0604.csv.csv'
fn = 'IC_tech_oriDT_0627.csv'
Prophetic(fn)
