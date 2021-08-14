
import pandas as pd
import numpy as np
import datetime


def min2day_v2(df,lag_ps):
    intraday = df;
    #preparation
    intraday['range1']=intraday['high'].rolling(lag_ps).max()-intraday['close'].rolling(lag_ps).min()
    intraday['range2']=intraday['close'].rolling(lag_ps).max()-intraday['low'].rolling(lag_ps).min()
    intraday['range']=np.where(intraday['range1']>intraday['range2'], intraday['range1'], intraday['range2'])
    
    return intraday

#signal generation
#even replace assignment with pandas.at
#it still takes a while for us to get the result
#any optimization suggestion besides using numpy array?
def signal_generation(df, intraday, param, column, lag_ps, stop_pr, is_prophetic):
    
    #as the lags of days have been set to 5  
    #we should start our backtesting after 4 workdays(這裡改成前四分鐘算) of current month
    #cumsum is to control the holding of underlying asset
    #sigup and siglo are the variables to store the upper/lower threshold  
    #upper and lower are for the purpose of tracking sigup and siglo
    signals=df[df.index>=intraday['date0'].iloc[lag_ps-1]]
    signals['signals']=0
    signals['cumsum']=0
    signals['upper']=0.0
    signals['lower']=0.0
    sigup=float(0)
    siglo=float(0)
    
    #for traversal on time series
    #the tricky part is the slicing
    #we have to either use [i:i] or pd.Series
    #first we set up thresholds at the beginning of london market which is est 3am
    #if the price exceeds either threshold
    #we will take long/short positions  
    
    for i in signals.index:
        #note that intraday and dataframe have different frequencies
        #obviously different metrics for indexes
        #we use variable date for index convertion
        # """ date: 年月日時分"""
        # date ='%s-%s-%s %s-%s-%s%s' %(i.year,i.month, i.day, i.hour, i.minute, i.second, i.second)
        ## date = '2015-1-5 9-20-00'
        
        # market opening
        if is_prophetic:
            
        # 從Daily return來算開般的action signal？
            time = pd.to_datetime(i)
            td1 = datetime.timedelta(hours=6)
            td2 = datetime.timedelta(minutes=-31)#
            time_shift = time + td1 + td2
            time_shift = str(time_shift)
            
            if (i.hour==9 and i.minute==31):
                if signals['open'][i] > signals['close'][time_shift]:
                    signals.at[i,'signals']=-1
                if signals['open'][i] <= signals['close'][time_shift]:
                    signals.at[i,'signals']=1
        
        #set up thresholds
        # if (i.hour==9 and i.minute==16):
        #     sigup=float(param*intraday['range'][date]+pd.Series(signals[column])[i])
        #     siglo=float(-(1-param)*intraday['range'][date]+pd.Series(signals[column])[i])

        if (i.hour==9 and i.minute==31) or (i.hour==10 and i.minute==31) \
            or (i.hour==13 and i.minute==1) or (i.hour==14 and i.minute==1):
            sigup=float(param[0]*intraday['range'][i]+pd.Series(signals[column])[i])
            siglo=float(-(1-param[1])*intraday['range'][i]+pd.Series(signals[column])[i])

        #thresholds got breached
        #signals generating
        if (sigup!=0 and pd.Series(signals[column])[i]>sigup):
            signals.at[i,'signals']=1
        if (siglo!=0 and pd.Series(signals[column])[i]<siglo):
            signals.at[i,'signals']=-1
        
        
        #check if signal has been generated
        #if so, use cumsum to verify that we only generate one signal for each situation
        if pd.Series(signals['signals'])[i]!=0:
            signals['cumsum']=signals['signals'].cumsum()
            # print(signals['cumsum'])
            
            ##############################
            #if same signal happens continuously, we regard it as no signal, and do nothing
            if (pd.Series(signals['cumsum'])[i]>1 or pd.Series(signals['cumsum'])[i]<-1):
                signals.at[i,'signals']=0
            
            # 加上停利
            # if pd.Series(signals['cumsum'])[i]>1:
            #     if pd.Series(signals[column])[i] <= sigup*(1+stop_pr):
            #         signals.at[i,'signals']=0
            #     if pd.Series(signals[column])[i] > sigup*(1+stop_pr):
            #         signals.at[i,'signals']=-2
            # if pd.Series(signals['cumsum'])[i]<-1:
            #     if pd.Series(signals[column])[i] >= siglo*(1-stop_pr):
            #         signals.at[i,'signals']=0
            #     if pd.Series(signals[column])[i] < siglo*(1-stop_pr):
            #         signals.at[i,'signals']=2
            ###############################
            
            #if the price goes from below the lower threshold to above the upper threshold during the day
            #we reverse our positions from short to long
            if (pd.Series(signals['cumsum'])[i]==0):
                if (pd.Series(signals[column])[i]>sigup):
                    signals.at[i,'signals']=2
                if (pd.Series(signals[column])[i]<siglo):
                    signals.at[i,'signals']=-2
                    
        #by the end of london market, which is est 12pm
        #we clear all opening positions
        #the whole part is very similar to London Breakout strategy
        if i.hour==15 and i.minute==0:
            
            # sigup,siglo=float(0),float(0) #6/17下午註解掉
            
            signals['cumsum']=signals['signals'].cumsum()
            # print(signals['cumsum'])
            signals.at[i,'signals']=-signals['cumsum'][i:i]
            
        #keep track of trigger levels
        signals.at[i,'upper']=sigup
        signals.at[i,'lower']=siglo

    return signals



def main():
    
    #similar to London Breakout
    #my raw data comes from the same website
    # http://www.histdata.com/download-free-forex-data/?/excel/1-minute-bar-quotes
    #just take the mid price of whatever currency pair you want
    # fileName = 'IF_tech_0604.csv'
    fileName = 'IC_tech.csv'
    df=pd.read_csv(fileName)
    df.rename(columns={"Unnamed: 0": "date0"}, inplace=True)
    df.set_index(pd.to_datetime(df['date0']),inplace=True) # 去除
    # df=df.drop(columns=['date0','money'])
    # df=df['2015-12-20 09:16:00':'2016-03-31 15:00:00']  # tiny data
    df=df['2015-12-20 09:16:00':'2019-05-09 15:00:00']  # all data
    
    ############################################################################
    #lag_periods is the lags of periods
    #param is the parameter of trigger range, it should be smaller than one
    #normally ppl use 0.5 to give long and short 50/50 chance to trigger
    lag_periods=240
    param=[0.5,0.5] #best
    # param=[0.45,0.45]
    stop_profit_rate = 0.02
    is_prophetic = False
    # is_prophetic = True
    ############################################################################

    #these three variables are for the frequency convertion from minute to intra daily
    # column='open'	
    column='close'	
    
    intraday = min2day_v2(df, lag_periods)
    signals=signal_generation(df,intraday,param,column,lag_periods, stop_profit_rate, is_prophetic)
    # signals=signals['2015-12-31 09:16:00':'2016-03-31 15:00:00']
    signals=signals['2015-12-31 09:16:00':'2019-05-09 15:00:00']
    # signals.to_csv("out_DT_0523.csv")
    
    for t in range(300,len(signals)-1,1):
        if signals['cumsum'][t]==0:
            signals['cumsum'][t] = signals['cumsum'][-1]
        elif signals['cumsum'][t] > 1:
            signals['cumsum'][t] = 1
        elif signals['cumsum'][t] <-1:
            signals['cumsum'][t] = -1
    
    
    ##### save to csv file #####
    # signals.to_csv("IF_tech_oriDT.csv")
    signals.to_csv("IC_tech_oriDT.csv")
    
    tot_sum = np.sum(abs(signals['cumsum'].values))
    print('tot_sum=',tot_sum)


if __name__ == '__main__':
    main()







