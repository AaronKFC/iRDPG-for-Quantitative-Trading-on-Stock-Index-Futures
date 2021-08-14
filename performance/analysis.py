# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:41:10 2021

@author: AllenPC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eps = 1e-8

def total_return(returns):
    '''Total return rate'''
    return returns[-1]

def sharpe_ratio(returns, freq=243, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe ratio"""
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)
    
def volatility(returns):
    '''measure the uncertainty of return rate'''
    return np.std(returns)

def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (peak - trough) / (peak + eps)


ret_rdpg = pd.read_csv('DayRet_RDPG.csv', header=None)
ret_rdpg = ret_rdpg.values
ret_rdpg_db = pd.read_csv('DayRet_RDPG-DB.csv', header=None)
ret_rdpg_db = ret_rdpg_db.values
ret_rdpg_bc = pd.read_csv('DayRet_RDPG-BC.csv', header=None)
ret_rdpg_bc = ret_rdpg_bc.values
ret_irdpg = pd.read_csv('DayRet_iRDPG.csv', header=None)
ret_irdpg = ret_irdpg.values

##### Calculate the Policy performance #####

ret_lst = [ret_rdpg, ret_rdpg_db, ret_rdpg_bc, ret_irdpg]
Tr = []
Sr = []
Vol = []
Mdd = []
i=1
for r in ret_lst:
    acc_r = np.cumsum(r)
    tr = total_return(acc_r)
    mdd = max_drawdown(acc_r)
    sr = sharpe_ratio(r)
    vol = volatility(r)
    # acc_r_ret = (np.roll(acc_r,1)-acc_r) / acc_r
    # acc_r_ret = acc_r_ret[1:]
    # sr = sharpe_ratio(acc_r_ret)
    # vol = volatility(acc_r_ret)
    
    Tr.append(np.round(tr, 2))
    Sr.append(np.round(sr, 3))
    Vol.append(np.round(vol, 3))
    Mdd.append(np.round(mdd, 2))

performance_dic = {'Tr':Tr, 'Sr':Sr, 'Vol':Vol, 'Mdd':Mdd}
performance = pd.DataFrame(performance_dic)#,index=[0])
performance.to_csv('Policy_Performance.csv',  index=False)


###########################################################################
'''Ablation Plot'''
df_daily=pd.read_csv("Dataset/IF_2015to2018_day.csv",parse_dates=True,index_col=0)
df_daily=df_daily['2018-05-09':'2019-05-08']
day_lst=df_daily.index
day_lst=[d.date() for d in day_lst] 

acc_ret = np.squeeze(np.cumsum(ret_lst,1))

def cum_ret(acc_ret):
    plt.plot(day_lst, acc_ret[0], 'r--')
    plt.plot(day_lst, acc_ret[1], ':', color='C1')
    plt.plot(day_lst, acc_ret[2], 'b-.')
    plt.plot(day_lst, acc_ret[3], 'g-')
    plt.ylabel('Cumulative Return(%)')
    plt.legend(['RDPG', 'RDPG-DB', 'RDPG-BC', 'iRDPG'], 
                loc=None, fontsize='small')
    # plt.grid()
    plt.show()

cum_ret(acc_ret)


###########################################################################
'''Generalizability Plot'''
ret_irdpg_IF = pd.read_csv('DayRet_iRDPG_IF.csv', header=None)
ret_irdpg_IF = ret_irdpg_IF.values
ret_irdpg_IC = pd.read_csv('DayRet_iRDPG_IC.csv', header=None)
ret_irdpg_IC = ret_irdpg_IC.values

gen_ret_lst = [ret_irdpg_IF, ret_irdpg_IC]
# for r in gen_ret_lst:
#     acc_r = np.cumsum(r)
gen_acc_ret = np.squeeze(np.cumsum(gen_ret_lst,1))

def generalizability(gen_acc_ret):
    # plt.plot(day_lst, acc_ret[0], 'r--')
    # plt.plot(day_lst, acc_ret[1], ':', color='C1')
    plt.plot(day_lst, gen_acc_ret[0], 'g-')
    plt.plot(day_lst, gen_acc_ret[1], 'b--')
    plt.ylabel('Cumulative Return(%)')
    plt.legend(['iRDPG (IF)', 'iRDPG (IC)'], 
                loc=None, fontsize='small')
    # plt.grid()
    plt.show()

generalizability(gen_acc_ret)



