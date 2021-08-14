import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from util import *

eps = 1e-8

class Evaluator(object):
    def __init__(self, test_episodes, max_episode_length=None, args=None):
        self.tot_test_episodes = test_episodes  #我們test那年有243個交易日
        self.max_episode_length = max_episode_length   #每天最長是240分鐘，最多交易240次
        self.test_episodes = args.test_episodes
        self.day_rewards = np.array([]).reshape(self.tot_test_episodes, 0)
        self.is_BClone = args.is_BClone

    def __call__(self, env, agent_test, description, lackM=True, debug=False, save=True):

        self.is_training = False
        observation = None
        day_reward = []
        day_ret=[]
        
        for episode in range(self.tot_test_episodes):
            # reset at the start of episode
            observation = env.reset(episode)
            observe_cuda = to_tensor(np.array([observation])).cuda()
            
            episode_steps = 0
            episode_reward = 0.
            step_profit=[]
                
            assert observation is not None
            
            # start episode
            done = False
            while not done:
                
                # agent_test.reset_rnn_hidden(done=True) 
                
                action, _ = agent_test.select_action(observe_cuda, noise_enable=False, decay_epsilon=False)
                act = np.argmax(action)

                _, observation, reward, done, infos = env.step(act)
                observation = deepcopy(observation.values)
                observe_cuda = to_tensor(np.array([observation])).cuda()
                
                if done:
                    agent_test.reset_rnn_hidden(done=True) 
                    
                ##### Calculate (DSR) reward & append step_profit #####
                episode_reward += reward
                step_profit.append(infos[episode_steps]['profit'])
                episode_steps += 1
                
            ##### Calculate Accumulated Return #####
            day_profit = np.sum(step_profit)*300
            # print(day_profit)
            day_margin_init = infos[0]['margin']
            # print(day_margin_init)
            eps_ret = day_profit / day_margin_init#[0]
            day_ret.append(eps_ret)
            
            if debug: prYellow(f'[Evaluate][Episode{episode}, Len{episode_steps}]: epi_reward={episode_reward:.2f}  day_profit={day_profit:.2f}')
            day_reward.append(episode_reward)
        
        day_reward = np.array(day_reward).reshape(-1,1)
        self.day_rewards = np.hstack([self.day_rewards, day_reward])
        day_acc_ret = np.cumsum(day_ret)
        day_ret_pd = pd.Series(day_ret)
        day_ret_pd.to_csv('results/DayRet' +description +'.csv', index=False, header=False)
        
        plt.cla()
        plt.xlabel('# of episode')
        plt.ylabel('accumulated_return')
        plt.plot(day_acc_ret)
        if lackM:
            fig_fn = description +'_lackTrue.png'
        else:
            fig_fn = description +'_lackFalse.png'
            
        plt.savefig('results/test_AccRet' +fig_fn)
        
        
        ##### Calculate the Policy performance #####
        day_ret = np.array(day_ret)
        Tr = self.total_return(day_acc_ret)
        Sr = self.sharpe_ratio(day_ret, freq=self.test_episodes)
        Vol = self.volatility(day_ret)
        Mdd = self.max_drawdown(day_acc_ret)
        performance_dic = {'Tr':Tr, 'Sr':Sr, 'Vol':Vol, 'Mdd':Mdd}
        performance = pd.DataFrame(performance_dic,index=[0])
        performance.to_csv('results/PolicyPerf' +description +'.csv',  index=False)
        print('performance=', performance)

        return np.mean(day_reward)
    
    
    def total_return(self, returns):
        '''Total return rate'''
        return returns[-1]
    
    def sharpe_ratio(self, returns, freq=243, rfr=0):
        """ Given a set of returns, calculates naive (rfr=0) sharpe ratio"""
        return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)
        
    def volatility(self, returns):
        '''measure the uncertainty of return rate'''
        return np.std(returns)
    
    def max_drawdown(self, returns):
        """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
        peak = returns.max()
        trough = returns[returns.argmax():].min()
        return (peak - trough) / (peak + eps)
    
