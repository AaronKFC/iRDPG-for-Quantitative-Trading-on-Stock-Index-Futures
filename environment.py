
# import talib
import argparse
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import timedelta,datetime


def randomDate(start, end, frmt): 
    #random選時間idx,IC:9:31~15:00
    #random選擇哪日期idx
    stime = time.mktime(time.strptime(start, frmt))
    etime = time.mktime(time.strptime(end, frmt))
    ptime = stime + random.random() * (etime - stime)
    return time.strftime(frmt,time.localtime(int(ptime)))


def preProcessData(file):  #原佳瑋的版本
    #read file
    df=pd.read_csv(file,parse_dates=True,index_col=0)
    '''
    concate technical index
    '''
    #depulicate unnormalized data to create initial normalized column
    # nColumns=len(df.columns)
    # for i in range(nColumns):
    #     print(df.columns[i])
    #     newLabel = "norm{}".format(df.columns[i].capitalize())
    #     df[newLabel] = df[df.columns[i]]
    
    # #normalize到[-1,1], 抓進來後再normalized
    # df[df.columns[nColumns:]]=2*((df[df.columns[nColumns:]] - df[df.columns[nColumns:]].min()) / (df[df.columns[nColumns:]].max() - df[df.columns[nColumns:]].min())) -1 #https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    
    # nColumns=['open','close','high','low','volume',
    #           'MACD','EMA_7','EMA_21','EMA_56','RSI',
    #           'BB_up','BB_mid','BB_low','slowK','slowD',
    #           'upper','lower']  # {upper:DT_buyline, lower:DT_sellline}
    # nColumns=['volume']
    # for cn in nColumns:
    #     newLabel = "norm{}".format(cn.capitalize())
    #     df[newLabel] = df[cn]
    #     df[newLabel] = 2*((df[newLabel] - df[newLabel].min()) / (df[newLabel].max() - df[newLabel].min())) -1
    
    nColumns=['open','close','high','low', 'EMA_7','EMA_56', 'upper','lower']# {upper:DT_buyline, lower:DT_sellline}
    for cn in nColumns:
        newLabel = "norm{}".format(cn.capitalize())
        df[newLabel] = df[cn]
        df[newLabel] = df[newLabel] / 4000
    
    nColumns=['RSI', 'slowK','slowD',]
    for cn in nColumns:
        newLabel = "norm{}".format(cn.capitalize())
        df[newLabel] = df[cn]
        df[newLabel] = df[newLabel] / 100
        
    nColumns=['MACD']
    for cn in nColumns:
        newLabel = "norm{}".format(cn.capitalize())
        df[newLabel] = df[cn]
        df[newLabel] = df[newLabel] / 5
        
    #deal with the datetime column
    df=df.reset_index()#['date']=pd.to_datetime(df['date'])
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df

    
def episode(df, stateWinlen, date, data_mode, duration, ep_idx):
    
    while True: #如果抽到非工作日的那天 , 就重抽 , 直到抽到有資料的那天
        if data_mode=='random':
            randomDay= randomDate(date[0],date[1],'%Y-%m-%d') #randomly choose one day
            dfDay=df[df['date'].dt.date == datetime.strptime(randomDay, '%Y-%m-%d').date()] #取出那天的資料
            if dfDay.shape[0]!=0: 
                randomDayIndex=df[df['date'].dt.date == datetime.strptime(randomDay, '%Y-%m-%d').date()].index[0] #取出那天第一筆資料的index
                dfYesterDay=df.iloc[(randomDayIndex-stateWinlen+1):randomDayIndex]
                dfDay=pd.concat([dfYesterDay,dfDay],axis=0)
                dfDay=dfDay.reset_index(drop=True) #重設取到的那天加上前天最後十分鐘的資料的index
                break
                
        elif data_mode=='time_order':
            df_daily=pd.read_csv("data_preprocess/raw_data/IF_2015to2018_day.csv",parse_dates=True,index_col=0)
            df_daily=df_daily['2018-05-09':'2019-05-08']
            day_lst=df_daily.index
            time_idx=day_lst[ep_idx].date()
            print('time_idx=', time_idx)
            
            dfDay=df[df['date'].dt.date == day_lst[ep_idx].date()]
        
            if dfDay.shape[0]!=0: 
                DayIndex=df[df['date'].dt.date == time_idx].index[0] #取出那天第一筆資料的index
                dfYesterDay=df.iloc[(DayIndex-stateWinlen+1):DayIndex]
                dfDay=pd.concat([dfYesterDay,dfDay],axis=0)
                dfDay=dfDay.reset_index(drop=True) #重設取到的那天加上前天最後十分鐘的資料的index
                break

    #(datetime.strptime(randomDay, '%Y-%m-%d')+timedelta(days = 1)).strftime('%Y-%m-%d')
    return dfDay
    

class environment():
    def __init__(self, data_fn, data_mode='random', duration='test', is_demo=False, \
                 is_intraday=True, is_lack_margin=True, args=None):  # data_mode = ['demo','train','test']
        '''
        class state包含:
            #假設stateWinlen分鐘為一個區間,並以1分鐘為單位做rolling window
            1.market observation: normalized 開盤,normalized 收盤,normalized 每分鐘的最高價,normalized 每分鐘的最低價,normalized 每分鐘的volumn,normalized 技術指標們(區間裡的第一分鐘可能會用到過去的資料才能算技術指標)
            2.account observation: 現有資產(就是保證金margin,會隨著trade而變動,與每點300塊有關),profit
        '''
        self.seed = args.seed
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        ##### preprocess market data #####
        self.data_mode = data_mode
        self.duration = duration
        self.is_demo = is_demo
        self.is_BClone = args.is_BClone
        self.is_PER_replay = args.is_PER_replay
        self.is_intraday = is_intraday
        self.is_lack_margin = is_lack_margin
        self.lackM_ratio = args.lackM_ratio
        
        
        # if self.is_BClone:# or args.is_pretrain:
        # # if self.is_BClone:
        #     print('Using data of "DT Strategy + Prophetc"')
        #     self.Data=preProcessData("data_preprocess/out_tech_DT_Prophetic_0606.csv")  # DT + Prophetic
        # else:
        #     print('Using data of original "DT Strategy"')
        #     self.Data=preProcessData("data_preprocess/out_tech_oriDT_0607.csv")  # origianl DT
        
        self.Data=preProcessData(data_fn)  
        
        if self.duration == 'train':
            self.time_range = ('2016-01-01','2018-05-08')
        elif self.duration == 'test':
            self.time_range = ('2018-05-08','2019-05-08')
        
        ##### market observation setting #####
        self.temp_observation = ['normOpen','normClose','normHigh','normLow',
                                 'normUpper','normLower', 'normEma_7','normEma_56',
                                 'normRsi','normSlowk','normSlowd','normMacd']
        account_obs = ['profit','normMargin']
        self.market_observation = deepcopy(self.temp_observation)
        for cn in account_obs:
            self.market_observation.append(cn)
        
        ##### episode setting & initialze #####
        self.stepIdx=0
        self.stateWinlen= args.seq_len ########## 特別留意，slicing時不用減1，在indecing時則記得減1。 ###########
        self.done=False
        self.mktOb=episode(self.Data, self.stateWinlen, self.time_range, self.data_mode, self.duration, ep_idx=0) #調用episode()
        # print('mktOB_len=',len(self.mktOb))
        self.finalStep = self.mktOb.index[-1] - self.stateWinlen +1
        # print('finalStep=',self.finalStep)
        
        
        ##### account setting #####
        self.principal = 500000  #本金先設著等之後算return會用到？
        self.margin = 500000
        self.lack_margin = False
        self.num_lack_margin = 0
        
        ##### trading setting #####
        '''
        下單狀態(self.position)：{-1=空單, 0=無單, 1=多單}
        交易訊號(self.action)：{1=long, -1=short}
        '''
        self.transFee=2.3*(10**(-5))
        self.slip = 0.2  # constant slippage
        self.profit=0
        self.position=0
        self.hold_float=0
        self.step_hold_profit=0
        
        ### DSR parameters ### (DSR: Differential Sharp Ratio)
        self.R_max = args.Reward_max_clip
        self.At0 = 0
        self.Bt0 = 0
        self.eta = 1/100000 
        self.SRt0 = 0
        
        self.ep_prev_action = random.choice([1,-1])  # 只有最一開始是random initialize，後面就會愈學愈好
        
                  
    def reset(self, ep_idx):
        self.infos = []
        self.done=False
        self.stepIdx=0
        
        ### 剛開盤的掛單情況 ###
        if self.is_intraday:
           self.position=0
           
        self.margin = self.principal
        self.profit = 0
        self.hold_float=0
        self.step_hold_profit=0
        
        self.mktOb=episode(self.Data, self.stateWinlen, self.time_range, self.data_mode, self.duration, ep_idx)  #每次reset時，都重新從market中fetch一段episode出來
        # print('mktOB_date=',self.mktOb.date)
        # print('mktOB=',self.mktOb)
        
        '''##### initial state setting #####'''
        self.state0 = self.mktOb[self.temp_observation][self.stepIdx : (self.stepIdx + self.stateWinlen)]
        init_margin = np.ones(self.stateWinlen)*self.margin / self.principal
        init_profit = np.zeros(self.stateWinlen)
        self.state0['profit'] = init_profit
        self.state0['margin'] = init_margin
        
        ### DSR parameters ### 
        self.At0 = 0
        self.Bt0 = 0
        
        return self.state0
        
        
    def step(self, action):
        step_init_margin=0
        self.stepIdx+=1
        
        if self.is_BClone == True or self.is_PER_replay:
            action_bc = self.mktOb['phtAction'][self.stepIdx + (self.stateWinlen -1)]
            if action_bc==0 and self.is_PER_replay:
                action_bc=random.choice([1,-1]) #radnomly choose an action
            
            if action_bc==-1:
                action_bc = np.array([1., 0.])
            elif action_bc==1:
                action_bc = np.array([0., 1.])
        else:
            action_bc = None
        
        ##### Demonstration Buffer ##### 
        ###(注意：進行此模式時，就算有action吃進來，在以下if也會被改成是demo的action)
        if self.is_demo == True:
            action = self.mktOb['cumsum'][self.stepIdx + (self.stateWinlen -1)]
            if action == 0:
                if self.stepIdx != self.finalStep:
                    action = self.ep_prev_action
                elif self.stepIdx == self.finalStep:
                    action = self.step_prev_action
        
        #因為吃進來的action是np.argmax(act)=0 or 1，所以把0轉成-1
        if action==0:
            action=-1
        
        
        '''##### current state setting #####'''
        ### slice current market observations from the same episode ###
        self.state1 = self.mktOb[self.temp_observation][self.stepIdx : (self.stepIdx +self.stateWinlen)]
        
        ### update new account observations ###
        ### slice last nine historical profits from privious_state 
        state_profit0 = self.state0['profit'][1:self.stateWinlen].values
        state_margin0 = self.state0['margin'][1:self.stateWinlen].values 
        
        ### Then excute trading() to get current_profit
        if self.stepIdx != self.finalStep:
            self.trading(action) #調用trading() => self.profit和self.margin 會自動計算
            ### append this current_profit to privious_profits
            state_profit1 = np.append(state_profit0, self.profit / 10)
            state_margin1 = np.append(state_margin0, self.margin / self.principal)
        else:
            self.trade_at_terminate(action) #調用trade_at_terminate() => self.profit和self.margin 會自動計算
            ### append this current_profit to privious_profits
            state_profit1 = np.append(state_profit0, self.profit / 10)
            state_margin1 = np.append(state_margin0, self.margin / self.principal)
        
        ### finally combine market observation and completed current_profits
        self.state1['profit'] = state_profit1
        self.state1['margin'] = state_margin1 
        self.state1['normMargin'] = state_margin1  #state的浮動margin對本金normalize
        
        
        '''##### termination setting #####'''
        if self.is_lack_margin and self.margin <= (self.lackM_ratio*self.principal):
            print('lack_margin happen!!!!!!!=',self.margin)
            self.trade_at_terminate(action) #強制出場並計算profit
            self.done=True
            self.ep_prev_action = self.step_prev_action
            self.lack_margin = True
            self.num_lack_margin += 1
        
        if self.stepIdx==self.finalStep: 
            self.done=True
            self.lack_margin = False
            ### 每天收盤的強制平倉，在self.trade_at_terminate()已處理
        
        '''##### reward Calculation #####'''
        step_init_margin = self.state0['margin'].values#.reset_index()
        self.step_init_margin = step_init_margin[-1]
        # reward = self.DSR_reward()
        reward = self.DSR_reward2()
        # reward = reward + 0.5*np.clip(self.profit,-self.R_max, self.R_max)
        # reward2 = self.DSR_reward2()
        # reward = reward + reward2 * 0.5
        
        
        
        '''##### store important info #####'''
        info = {#'lackM':self.lack_margin,  ### 此行檢查用
                # 'step_hold':self.step_hold_profit,  ### 此行檢查用
                'profit': self.profit,
                'margin': self.step_init_margin * self.principal, #要先scale回來？
                'step_ret': self.profit / self.step_init_margin,
                'episode_len':len(self.mktOb),
                'num_lack_margin':self.num_lack_margin
                }
        self.infos.append(info)

        ### current_state，到下一step時就會變成"privious_state" (state0：privious_state)
        self.state0 = self.state1
        self.step_prev_action = action
        
        next_state = self.state1[self.market_observation]
        
        return action_bc, next_state, reward, self.done, self.infos#, self.mktOb, self.lack_margin#, self.Data 
    
    
    def trading(self, action):
        if action==0:
            print('trading_action is zero, which is wrong.')
            raise AssertionError('trading_action is zero')
            
        pt0 = self.mktOb['close'][self.stepIdx + (self.stateWinlen-1) - 1] #p_{t-1} , 因為做當沖所以用Open
        pt1 = self.mktOb['close'][self.stepIdx + (self.stateWinlen-1)] # p_{t} , 注意9:31分進場取的open其實是9:30分的open
        # print(f'pt0={pt0}, pt1={pt1}')
        
        '''### 依「交易訊號&持單狀態」執行交易rules ###'''
        if action == 1:  #action為交易訊號：{1=long, -1=short}
            if self.position == 1:  #position為agent下單狀態：{-1=空單, 0=無單, 1=多單}
                self.profit = 0
                self.step_hold_profit = (pt1 - pt0)*self.position
                self.hold_float += self.step_hold_profit
                self.margin += self.hold_float*300
                # self.margin = self.principal + self.hold_float*300
                # self.position = self.position  # 即position不變
            elif self.position == 0: 
                self.profit = 0
                # self.hold_float = self.hold_float  #即開倉long後，hold值不會馬上變化
                # self.margin = self.margin  #即開倉long後，margin不會馬上變化
                self.position = 1
                # self.step_hold_profit = self.step_hold_profit
            elif self.position ==-1:
                self.profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*abs(action-self.position)*pt1 + self.hold_float
                self.step_hold_profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*abs(action-self.position)*pt1
                self.hold_float = 0  #change position，所以position歸零
                self.margin += self.hold_float*300
                # self.margin = self.principal + self.hold_float*300
                self.position = 1
                
                
        elif action == -1:
            if self.position == 1:
                self.profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*abs(action-self.position)*pt1 + self.hold_float
                # self.step_hold_profit = self.profit
                self.step_hold_profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*abs(action-self.position)*pt1
                self.hold_float = 0
                self.margin += self.hold_float*300
                # self.margin = self.principal + self.hold_float*300
                self.position = -1
            elif self.position == 0:
                self.profit = 0
                # self.hold_float = self.hold_float  #即開倉short後，hold值不會馬上變化
                # self.margin = self.margin  #即short後，margin不會馬上變化
                self.position = -1
                # self.step_hold_profit = self.step_hold_profit
            elif self.position ==-1:
                self.profit = 0
                self.step_hold_profit = (pt1 - pt0)*self.position
                self.hold_float += self.step_hold_profit
                self.margin += self.hold_float*300
                # self.margin = self.principal + self.hold_float*300
                # self.position = self.position  # 即position不變
                
    
    def trade_at_terminate(self, action):
        '''注意：這邊的price應該要用raw data的值，而不是normalize過的'''
        pt0 = self.mktOb['close'][self.stepIdx + (self.stateWinlen-1) -1] #p_{t-1} , 因為做當沖所以用Open
        pt1 = self.mktOb['close'][self.stepIdx + (self.stateWinlen-1)] #因照4/23大家決議：在收盤時一定平倉，不會持倉到隔日。
        
        ### 因為是強制平倉，所以不用管交易訊號action
        # print('before_terminate_position=',self.position)
        if self.position == 0:  #position為agent下單狀態：{-1=空單, 0=無單, 1=多單}
            self.profit = 0
            self.hold_float = 0
            # self.margin = self.margin
            # self.position = 0 #在terminate時，就算有交易訊號，但還是維持postion=0
        
        else: # self.position ==-1 或 1
            #即該episode在最後收盤terminate時，才進入以下的強制平倉操作。
            #注意此時的pt1是當天最後一分k的close價。
            self.profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*2*pt1 + self.hold_float
            self.step_hold_profit = ((pt1-pt0)*self.position-2*self.slip) - self.transFee*2*pt1  
            self.hold_float += self.step_hold_profit  #lack margin時的profit是用hold_float算，所以這邊還不能reset
            self.margin += self.hold_float*300
            # self.margin = self.principal + self.hold_float*300  #IF&IC股指期貨每點皆為$300CNY
            # self.position = 0
            
        self.ep_prev_action = action  # 若非intraday交易的話，此action為隔日的action
        
    
    def DSR_reward(self):
        self.eta = 1/self.stepIdx
        # self.eta = 1/240
        
        ### Calculate step return #####
        # Rt1 = self.profit*300 / (self.step_init_margin * self.principal)
        Rt1 = self.step_hold_profit*300 / (self.step_init_margin * self.principal)
        
        ### update At0 & Bt0 ###
        self.At0 = self.eta*Rt1 + (1-self.eta)*self.At0
        self.Bt0 = self.eta*Rt1**2 + (1-self.eta)*self.Bt0
        
        if self.stepIdx==1:
            SRt1 = 0
            DSR = 0
            
        else:
            K_eta = np.sqrt(1/(1-self.eta))
            # K_eta = np.sqrt((1-self.eta/2)/(1-self.eta))
            # print(f'Rt1={Rt1:.6f}, At0={self.At0:.6f}, Bt0={self.Bt0:.8f}, K_eta={K_eta:.3f}')
            if np.sqrt(self.Bt0 - (self.At0)**2)==0:
                # print('Sharp Ratio has problem of "Zero Denominator" !!!!!!!!!!!!!!!!!!!!!')
                # print('Numerator=', self.At0)
                # print('Denominator=', np.sqrt(self.Bt0 - (self.At0)**2))
                SRt1 = 0
            else:
                SRt1 = self.At0 / K_eta / np.sqrt(self.Bt0 - (self.At0)**2)
            
            diffSR = SRt1 - self.SRt0
            DSR = diffSR / self.eta *5  #乘以5是故意的，可以不用
            
        # DSR = np.clip(DSR, -self.R_max, self.R_max)
        self.SRt0 = SRt1
        # print(f'step_init_margin={self.step_init_margin:.6f}, DSR={DSR:.6f}, Rt1={Rt1:.6f}, step_hold_profit={self.step_hold_profit:.2f}, SRt1={SRt1:.4f}')
        return DSR
    
    def DSR_reward2(self):
        self.eta = 1/self.stepIdx
        # self.eta = 1/240
        
        ### Calculate step return #####
        Rt1 = self.profit*300 / (self.step_init_margin * self.principal)
        # Rt1 = self.step_hold_profit*300 / (self.step_init_margin * self.principal)
        
        ### update At0 & Bt0 ###
        self.At0 = self.eta*Rt1 + (1-self.eta)*self.At0
        self.Bt0 = self.eta*Rt1**2 + (1-self.eta)*self.Bt0
        
        if self.stepIdx==1:
            SRt1 = 0
            DSR = 0
            
        else:
            K_eta = np.sqrt(1/(1-self.eta))
            # K_eta = np.sqrt((1-self.eta/2)/(1-self.eta))
            # print(f'Rt1={Rt1:.6f}, At0={self.At0:.6f}, Bt0={self.Bt0:.8f}, K_eta={K_eta:.3f}')
            if np.sqrt(self.Bt0 - (self.At0)**2)==0:
                # print('Sharp Ratio has problem of "Zero Denominator" !!!!!!!!!!!!!!!!!!!!!')
                # print('Numerator=', self.At0)
                # print('Denominator=', np.sqrt(self.Bt0 - (self.At0)**2))
                SRt1 = 0
            else:
                SRt1 = self.At0 / K_eta / np.sqrt(self.Bt0 - (self.At0)**2)
            
            diffSR = SRt1 - self.SRt0
            DSR = diffSR / self.eta *5  #乘以5是故意的，可以不用
            
        # DSR = np.clip(DSR, -self.R_max, self.R_max)
        self.SRt0 = SRt1
        # print(f'step_init_margin={self.step_init_margin:.6f}, DSR={DSR:.6f}, Rt1={Rt1:.6f}, step_hold_profit={self.step_hold_profit:.2f}, SRt1={SRt1:.4f}')
        return DSR
        
    # def DSR_reward(self):
    #     # print('stepIdx=',self.stepIdx)
    #     # self.eta = 1/240
    #     self.eta = 1/self.stepIdx
    #     ### Rt1有三種表示法：1.直接就是profit。  2.profit/pt0  3.prorit*300/margin
    #     # Rt1 = self.profit #方法1
    #     # pt0 = self.mktOb['open'][self.stepIdx + self.stateWinlen]
    #     # Rt1 = self.profit / pt0  #方法2
    #     # Rt1 = self.profit*300 / self.margin
    #     # print(self.step_init_margin)
    #     # Rt1 = self.profit*300 / self.step_init_margin
    #     Rt1 = self.step_hold_profit*300 / (self.step_init_margin * self.principal)
    #     # Rt1 = self.step_hold_profit*300 / self.principal
    #     # print('Rt1=',Rt1)
    #     # print(f'step_init_margin={self.step_init_margin:.6f}, step_hold_profit={self.step_hold_profit:.2f}, Rt1={Rt1:.6f}',)
        
    #     ### calculate deltaA&B by Rt1 ###
    #     deltaAt1 = Rt1 - self.At0
    #     deltaBt1 = Rt1**2 - self.Bt0
        
    #     # print(f'Rt1={Rt1:.6f}, At0={self.At0:.6f}, self.Bt0={self.Bt0:.8f}, deltaAt1={deltaAt1:.6f}, deltaBt1={deltaBt1:.8f}')
        
    #     ### calculate DSR ###
    #     if (self.Bt0 - self.At0**2)**(3/2)==0:
    #         print('Sharp Ratio has problem of "Zero Denominator" !!!!!!!!!!!!!!!!!!!!!')
    #         print('Numerator=', (self.Bt0*deltaAt1 - 0.5*self.At0*deltaBt1))
    #         print('Denominator=', (self.Bt0 - self.At0**2)**(3/2))
    #         Dt1 = 0
    #     else:
    #         Dt1 = (self.Bt0*deltaAt1 - 0.5*self.At0*deltaBt1) / (self.Bt0 - self.At0**2)**(3/2)
    #     ### 
    #     # if self.stepIdx == 1 :
    #     #     Dt1 = 0
    #     # else:
    #     #     Dt1 = (self.Bt0*deltaAt1 - 0.5*self.At0*deltaBt1) / (self.Bt0 - self.At0**2)**(3/2)
        
    #     ### update At0 & Bt0 ###
    #     self.At0 = self.At0 + self.eta * deltaAt1
    #     self.Bt0 = self.Bt0 + self.eta * deltaBt1
        
    #     ### clip Dt to avoid value explore ###
    #     # Dt1 = np.clip(Dt1, -self.R_max, self.R_max)
        
    #     print(f'step_init_margin={self.step_init_margin:.6f}, DSR={Dt1:.3f}, Rt1={Rt1:.6f}, step_hold_profit={self.step_hold_profit:.2f}')
        
    #     return Dt1  #即DSR_reward
        
    
######################################################################################  
###### Environment Testing for debug #######
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Env code testing')
    parser.add_argument('--seed', default=602, type=int, help='')
    parser.add_argument('--is_BClone', default=True, action='store_true')
    # parser.add_argument('--is_BClone', default=False, action='store_true', help='original Dual Thrust')
    parser.add_argument('--seq_len', default=15, type=int, help='sequence length of input state')
    parser.add_argument('--Reward_max_clip', default=15., type=float, help='max DSR reward for clipping')
    parser.add_argument('--max_episode_length', default=240, type=int, help='')  #(原RDPG設500，但我們每天都是跑240分鐘trading)
    parser.add_argument('--is_PER_replay', default=True)
    parser.add_argument('--lackM_ratio', default=0.9, type=int, help='lack margin stop ratio')
    
    args = parser.parse_args()
    
    ##### simulate training #####
    # # is_intraday = True
    # is_intraday = False
    # # is_demo = True
    # is_demo = False
    # data_mode = 'random'
    # duration = 'train'
    
    ##### simulate testing#####
    ##### or Run Dual Thrust Trading #####
    is_intraday = True
    # is_intraday = False
    is_demo = True  # True的話就是跑cumsum的action
    # is_demo = False
    data_mode = 'time_order'
    duration = 'test'
    
    is_lack_margin = True
    # is_lack_margin = False
    
    # env=enviornment(data_mode='demo') 
    # data_fn = "data_preprocess/out_tech_DT_Prophetic_0606.csv"
    data_fn = "data_preprocess/IF_tech_oriDT.csv"
    # data_fn = "data_preprocess/prophetic_0616.csv"
    env=environment(data_fn=data_fn, data_mode=data_mode, duration=duration, is_demo=is_demo, 
                    is_intraday=is_intraday, is_lack_margin=is_lack_margin, args=args)
    
    
    ### 假的prophetic ###
    a1 = np.ones(120)
    a2 = -np.ones(88)
    a3 = np.ones(32)
    action_pro = np.concatenate([a1,a2,a3])
    # action=policy(env.state)
    # next_state,reward,done = env.step(action)
    
    e=0 #episode index
    i=1 #因為前面已有env.reset()，所以應該從i=1開始loop，如此這個i就會等於stepIdx
    day_ret=[]
    while e<=(243-1): 
    # while e<=(0): #try跑3個episode
        step_reward=[]
        step_hold_f=[]
        step_profit=[]
        init_state=env.reset(e)
        # print(init_state)
        while i<(240):
        # while i<(2):
            action=random.choice([1,-1])
            # action=random.choice([1,1])
            # action = action_pro[i]
            # next_state, reward, done, infos, data = env.step(action)
            # action_bc, next_state, reward, done, infos, mktOb, lackM = env.step(action)
            action_bc, next_state, reward, done, infos = env.step(action)
            # print(DSR(next_state))
            # print(next_state)
            temp_profit = infos[i-1]['profit']
            # temp_hold = infos[i-1]['step_hold']  #測試用
            # print(f'whileIdx={i}: DSR={reward:.2f}, step_hold={temp_hold:.2f}, profit={temp_profit:.2f}, done={done}')
            # print(f'whileIdx={i}: DSR={reward:.2f}, profit={temp_profit:.2f}, done={done}')
            # print('')
            # info = infos[i-1]
            # step_hold_f.append(infos[i-1]['hold_float'])  #測試用
            step_profit.append(infos[i-1]['profit'])
            step_reward.append(reward)
            i+=1
            if done:
                # if lackM:
                #     print('length=',i)
                # print(f'termination_step={i}')
                break
        # if lackM:
        #     break
        # print('episodeIdx=',e+1)
        # print(f'termination_step={i}')
        # print(f'done={done}')
        # episode_len=infos[0]['episode_len']
        # print(f'episode_len={episode_len}')
        day_profit = np.sum(step_profit)*300
        print('day_profit=',day_profit)
        day_margin_init = infos[0]['margin']
        # print(day_margin_init)
        eps_ret = day_profit / day_margin_init#[0]
        day_ret.append(eps_ret)
        e+=1
        i=1
    
    # plt.xlabel('# of episode')
    # plt.ylabel('daily_return')
    plt.plot(day_ret)
    # plt.savefig("daily_return.png")
    
    if is_lack_margin:
        trade_fn = 'AccRet_lackTrue'
    else:
        trade_fn = 'AccRet_lackFalse'
    
    day_acc_ret = np.cumsum(day_ret)
    plt.xlabel('# of episode')
    plt.ylabel('accumulated_return')
    plt.plot(day_acc_ret)
    plt.savefig(trade_fn +'.png')
    
    trade_his = pd.DataFrame(day_acc_ret)
    trade_his.to_csv(trade_fn +'.csv', index=False)
    
    reward_his = pd.DataFrame(step_reward)
    reward_his.to_csv('step_reward.csv', index=False)
    
    # start = '2016-01-01 09:16:00' #'00:00:00'
    # end = '2018-05-08 00:00:00' #'12:12:12'
    # frmt='%Y-%m-%d %H:%M:%S'
    # #IC跟IF之後會分開train, 現在先以IF為主 , IF:9:16~11:30,13:01~15:15 , IC:9:31~11:30,13:01~15:00
    # print(randomDate(start,end,frmt))




# ##################
# mktOb_c = mktOb['close'][14:]
# mktOb_diff = -mktOb_c.diff(1)
# mktOb_diff.sum()
# mktOb_cum = mktOb_diff[2:].cumsum()

# pt0=3867.4
# pt1=3869.0
# position=-1
# slip=0.2
# transFee=2.3*(10**(-5))
# action=-1
# hold_float=9.4
# final_profit = ((pt1-pt0)*position-2*slip) - transFee*abs(action-position)*pt1 + hold_float
# final_profit_money = final_profit*300
            