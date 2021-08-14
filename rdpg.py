import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

from evaluator import Evaluator
from memory import EpisodicMemory
from PER_memory import PrioritizedReplayBuffer
from agent import Agent
from util import *
# from util import prGreen, prYellow, prRed, to_tensor, soft_update, 
import matplotlib.pylab as plt

DEMO_flag = 1


class RDPG(object):
    def __init__(self, demo_env, env, args):
        
        ### Create Environment Classes ###
        self.env = env
        self.demo_env = demo_env
        
        ##### Create Replay Buffer #####
        self.is_PER_replay = args.is_PER_replay
        if self.is_PER_replay:
            
            ### Porioritized experience replay (PER) buffer ###
            self.memory = PrioritizedReplayBuffer(args.PER_size, args.seed, alpha=args.p_alpha, 
                                                  beta_init=1.0, beta_inc_n=100, max_t=args.exp_traj_len)
        
        else:
            ### Original replay buffer ###
            self.memory = EpisodicMemory(capacity=args.rmsize, 
                                         max_train_traj_len=args.exp_traj_len,  # 整段episode被分段存取
                                         window_length=args.window_length)
            
        ### Evaluator is used for test trained agent ###
        self.evaluate = Evaluator(args.test_episodes,  #我們test那年有243個交易日
                                  max_episode_length=args.max_episode_length, #每天最長是240分鐘，最多交易240次
                                  args=args)  
        
        ##### Model Setting #####
        self.rnn_mode = args.rnn_mode
        self.seq_len = args.seq_len
        self.hidden_rnn = args.hidden_rnn
        self.num_layer = args.num_rnn_layer
        self.agent = Agent(args)
        if torch.cuda.is_available() : 
            self.agent.cuda()
        
        ### Hyper-parameters
        self.batch_size = args.bsize
        self.exp_traj_len = args.exp_traj_len
        self.max_episode_length = args.max_episode_length
        self.tau = args.tau
        self.discount = args.discount
        self.warmup = args.warmup
        self.a_update_freq = args.a_update_freq
        
        ##### Behavior Cloning Setting #####
        self.is_BClone = args.is_BClone
        self.is_Qfilt = args.is_Qfilt
        self.use_Qfilt = args.use_Qfilt
        if self.is_BClone:
            self.lambda_Policy = args.lambda_Policy
            self.lambda_BC = 1-self.lambda_Policy
        else:
            self.lambda_Policy = 1
            self.lambda_BC = 1-self.lambda_Policy
        # self.lambda_BC = args.lambda_BC
        self.BC_loss_func = nn.MSELoss(reduce=False)
        # self.BC_loss_func = nn.BCELoss(reduce=False)
        
        ##### PER demonstration Setting #####
        self.lambda_balance = args.lambda_balance
        self.small_const = args.small_const
        self.priority_const = args.priority_const
        
        self.is_demo_warmup = args.is_demo_warmup
        if self.is_demo_warmup:
            demo_protect_size = (self.max_episode_length /self.exp_traj_len) *self.warmup
            self.memory.set_protect_size(int(demo_protect_size))
        
        self.is_pretrain = args.is_pretrain
        self.pretrain_itrs = args.Pretrain_itrs
        
        ### Optimizer and LR_scheduler ###
        beta1 = args.beta1
        beta2 = args.beta2
        self.rnn_optim = Adam(self.agent.rnn.parameters(), lr=args.r_rate, betas=(beta1, beta2))
        self.rnn_scheduler = Scheduler.StepLR(self.rnn_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.critic_optim  = Adam(self.agent.critic.parameters(), lr=args.c_rate, betas=(beta1, beta2))
        self.critic_scheduler = Scheduler.StepLR(self.critic_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.actor_optim  = Adam(self.agent.actor.parameters(), lr=args.a_rate, betas=(beta1, beta2))
        self.actor_scheduler = Scheduler.StepLR(self.actor_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        
        
        ### initialized values 
        self.demoN_ratio = 0
        self.priority = 0
        self.actor_loss = 0
        self.BC_loss = 0
        self.BC_loss_Qf = 0
        self.tot_policy_loss = 0
        self.critic_loss = 0
        
        ### other setting ###
        if args.seed > 0:
            self.seed(args.seed)
        self.writer = SummaryWriter(args.logdir)
        self.is_training = True
        self.save_threshold = args.save_threshold
        self.date = args.date


    def train(self, num_episodes, checkpoint_path, debug):
        
        epi_idx = None #training時因為episode是random選date，並非照time_order，所以在此設成none沒關係。
        
        self.agent.is_training = True
        step = episode_steps = trajectory_steps = 0
        episode_reward = 0.
        state0 = None
        ewma_reward = 0
        episode = 1
        train_epi_reward = []
        train_ewma_reward = []
        train_actor_loss = []
        train_bc_loss = []
        train_bcQf_loss = []
        train_totPolicy_loss = []
        train_critic_loss = []
        demoN_ratio_batch = []
        
        while episode <= num_episodes:
            episode_steps = 1
            while episode_steps <= self.max_episode_length:
                
                if self.is_demo_warmup:
                #################### warmup adopt expert policy (Dual Thrust Strategy) ####################
                    if episode <= self.warmup:  # Note: warmup generate demonstrations, so here use demo_env
                        if state0 is None:
                            self.agent.reset() 
                            state0 = deepcopy(self.demo_env.reset(epi_idx))  #training時的env reset是random選date，所以此時epi_idx=None沒關係。
                            state0 = state0.values
                        
                        ### Note the following action will be from demonstration, not random.
                        action, epsilon = self.agent.random_action() #其實改成demo後，此action到demo_env也會變成demo的action，而非random
                        action_bc, next_state, reward, done, infos = self.demo_env.step(np.argmax(action))
                        
                    else:  # normal training, without demonstration, so her use training env
                        if state0 is None:
                            self.agent.reset() 
                            state0 = deepcopy(self.env.reset(epi_idx))  #training時的env reset是random選date，所以此時epi_idx=None沒關係。
                            state0 = state0.values
                            
                        state0_cuda = to_tensor(np.array([state0])).cuda()
                        # state0_cuda = to_tensor(state0).cuda()
                        action, epsilon = self.agent.select_action(state0_cuda)
                        action_bc, next_state, reward, done, infos = self.env.step(np.argmax(action))
                ######################################################################
                
                else:
                ############## original random warmup experiences ##############
                    #### reset if it is the start of episode
                    if state0 is None:
                        self.agent.reset()
                        state0 = deepcopy(self.env.reset(epi_idx))  #training時的env reset是random選date，所以此時epi_idx=None沒關係。
                        state0 = state0.values
                    state0_cuda = to_tensor(np.array([state0])).cuda()
                    
                    if episode <= self.warmup:
                        action, epsilon = self.agent.random_action()
                    else: # 正式training，新append experience為agent data
                        action, epsilon = self.agent.select_action(state0_cuda)
                    
                    act = action
                    action_bc, next_state, reward, done, infos = self.env.step(np.argmax(act))
                ###############################################################
                
                next_state = next_state.values
                next_state = deepcopy(next_state)


                ###### agent observe and update policy #####

                #if episode <= self.warmup or episode-self.warmup > self.pretrain:
                if self.is_PER_replay == False:
                    self.memory.append(action_bc, state0, action, reward, done)
                else:
                    self.memory.add((torch.from_numpy(state0).float(),
                                           torch.from_numpy(action).float(),
                                           torch.from_numpy(action_bc).float(),
                                           torch.tensor([reward]).float(),
                                           torch.from_numpy(next_state).float(),
                                           torch.tensor([0.95]),
                                           int(episode<=self.warmup)),
                                           trajectory_steps)

                # update 
                step += 1
                episode_steps += 1
                trajectory_steps += 1
                episode_reward += reward
                state0 = deepcopy(next_state)

                ##### 此exp_traj_len，目前設定成設定每16steps update一次 ##### (但每次episode還是跑到done才會結束)
                if trajectory_steps >= self.exp_traj_len:
                    ### 以下設定是為了讓hidden_state繼續往下一個step傳遞 ###
                    self.agent.reset_rnn_hidden(done=False)  #注意done對應model.py裡的reset_lstm_hidden_state()
                    trajectory_steps = 0
                    if episode > self.warmup:
                        self.update_policy(done)


                if done: # end of episod
                    # reset
                    state0 = None
                    ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
                    (lr_Critic, lr_Actor, lr_RNN) = (self.critic_scheduler.get_last_lr()[0], self.actor_scheduler.get_last_lr()[0], self.rnn_scheduler.get_last_lr()[0])
                    if debug: prYellow('[Step:{}, Episode:{}, Len:{}] [lr:{}, epsl:{}] [Reward={:.3f}  Ewma_Reward={:.3f}]'.format(step, episode, episode_steps, lr_Critic, np.round(epsilon,3), episode_reward, ewma_reward))
                    if self.is_BClone:
                        print('A_loss={:.3f}\tBC_loss={:.3f}\tTotP_loss={:.3f}\tC_loss={:.3f}'.format(self.actor_loss, self.BC_loss, self.tot_policy_loss, self.critic_loss))
                    else:
                        print('TotP_loss={:.3f}\tC_loss={:.3f}'.format(self.tot_policy_loss, self.critic_loss))
                    
                    ##### log data for Tensorboard #####
                    self.writer.add_scalar('Train/Episode Reward', episode_reward, episode)
                    self.writer.add_scalar('Train/EWMA Reward', ewma_reward, episode)
                    self.writer.add_scalar('Train/Actor Loss', self.actor_loss, episode)
                    self.writer.add_scalar('Train/BC Loss', self.BC_loss, episode)
                    self.writer.add_scalar('Train/Qf Loss', self.BC_loss_Qf, episode)
                    self.writer.add_scalar('Train/tot Policy Loss', self.tot_policy_loss, episode)
                    self.writer.add_scalar('Train/Critic Loss', self.critic_loss, episode)
                    self.writer.add_scalar('Train/Demon ratio in batch', self.demoN_ratio, episode)
                    
                    ##### append training info for plot #####
                    train_epi_reward.append(episode_reward)
                    train_ewma_reward.append(ewma_reward)
                    train_actor_loss.append(self.actor_loss)
                    train_bc_loss.append(self.BC_loss)
                    train_bcQf_loss.append(self.BC_loss_Qf)
                    train_totPolicy_loss.append(self.tot_policy_loss)
                    train_critic_loss.append(self.critic_loss)
                    demoN_ratio_batch.append(self.demoN_ratio)

                    if episode == self.warmup:
                        ite=1
                        if self.is_pretrain:
                            while(ite<=self.pretrain_itrs):
                                self.update_policy(done)
                                if ite%(self.pretrain_itrs/10) == 0:
                                    print('demoN_ratio=',self.demoN_ratio)
                                    print('Pretrain',ite,':','TotP_loss={:.3f}\tC_loss={:.3f}\tDemo_ratio={:.3f}'.format(self.tot_policy_loss, self.critic_loss, self.demoN_ratio))
                                ite+=1

                    episode_reward = 0.
                    episode += 1
                    self.agent.reset_rnn_hidden(done=False)  #注意done對應model.py裡的reset_hidden_state()
                    break
            
            ##### Save models #####
            if (episode-1) >= 150 or (episode-1) % 100 == 0 or ewma_reward > self.save_threshold:
                self.agent.save_model(checkpoint_path, (episode-1), ewma_reward)
            
            ##### Plot Training Curves #####
            if (episode-1) % 100 == 0:
                if self.is_BClone:
                    self.train_plot_bc(episode-1, train_epi_reward, train_ewma_reward,  
                                       train_totPolicy_loss, train_critic_loss,
                                       train_actor_loss, train_bc_loss, train_bcQf_loss)
                    self.train_demoN_ratio(episode-1, demoN_ratio_batch)
                else:
                    self.train_plot(episode-1, train_epi_reward, train_ewma_reward, 
                                    train_totPolicy_loss, train_critic_loss)
                    self.train_demoN_ratio(episode-1, demoN_ratio_batch)
            
            ##### Apply Q-filter to BC loss #####
            if (episode-1) >= (self.warmup+self.use_Qfilt):
                self.is_Qfilt=True

#            if step >= args.warmup and episode > args.bsize:
#                # Update weights
#                agent.update_policy()


    def update_policy(self, done):
        ### Sample batch of trajectories
        t_len = 0
        if self.is_PER_replay == False:
            experiences = self.memory.sample(self.batch_size)
            if len(experiences) == 0: # not enough samples
                return
            t_len = len(experiences)
        else:
            (state0s, actions, action_bcs, rewards, state1s, batch_gammas, batch_flagss), \
                                           weights, idxes = self.memory.sample(self.batch_size)
            t_len = len(state0s)

        actor_loss_total = 0  #actor loss
        BC_loss_total = 0  #BC loss
        BC_loss_Qf_total = 0  #BC loss after Q-filter
        policy_loss_total = 0  #policy loss
        value_loss_total = 0  #critic loss
        demo_cnt = []
        for t in range(t_len): # iterate over episodes
            if self.is_PER_replay == False and t == t_len-1:
                break
            a_target_cx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            a_target_hx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            
            a_cx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            a_hx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            
            
            if self.is_PER_replay == False:
                action_bc = np.stack((trajectory.action_bc for trajectory in experiences[t]))

                state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))          
                action = np.stack((trajectory.action for trajectory in experiences[t]))
                action = to_tensor(action)
                reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
                reward = to_tensor(reward)
                state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))
            
                state0_cuda = to_tensor(state0).cuda()
                state1_cuda = to_tensor(state1).cuda()
            else:
                state0 = state0s[t]
                action = actions[t]
                reward = rewards[t]
                state1 = state1s[t]
                batch_flags = batch_flagss[t]
                action_bc = action_bcs[t]
                state0_cuda = state0.cuda()
                state1_cuda = state1.cuda()
                
                ##### calculate demonstration ratio in a batch #####
                d_flags = torch.from_numpy(batch_flags)
                demo_select = d_flags == DEMO_flag
                N_act = demo_select.sum().item()
                demo_cnt.append(N_act/self.batch_size)
            
            
            ######################## critic loss calculation ########################
            # with torch.no_grad():
            if self.rnn_mode == 'lstm':
                xh0, _ = self.agent.rnn(state0_cuda, (a_hx, a_cx))
                current_q = self.agent.critic([xh0, action.cuda()])
                
                with torch.no_grad():
                    xh1, _ = self.agent.rnn_target(state0_cuda, (a_hx, a_cx))
                    target_action = self.agent.actor_target(xh1)
                    target_action = target_action.detach()
                    next_q_value = self.agent.critic_target([xh1, target_action])
                
                
            elif self.rnn_mode == 'gru':
                xh0, _ = self.agent.rnn(state0_cuda, a_hx)
                current_q = self.agent.critic([xh0, action.cuda()])
                
                with torch.no_grad():
                    xh1, _ = self.agent.rnn_target(state1_cuda, a_hx)
                    target_action = self.agent.actor_target(xh1)
                    target_action = target_action.detach()
                    next_q_value = self.agent.critic_target([xh1, target_action])
                    
            target_q = reward + (1-done) * self.discount * next_q_value.cpu()
            
            value_loss = 0
            if self.is_PER_replay == False:
                value_loss = F.smooth_l1_loss(current_q, target_q.cuda())
            else:
                value_loss = (F.smooth_l1_loss(current_q, target_q.cuda()) * torch.tensor(weights).cuda()).mean()
            value_loss /= t_len # divide by experience length
            value_loss_total += value_loss 
            ####### update Critic per step ####### 
            self.agent.rnn.zero_grad()
            self.agent.actor.zero_grad()
            self.agent.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            self.rnn_optim.step()  
            
            
            ########################## Actor loss calculation & Update Actor ##########################
            ################## Preliminary for loss calculation ##################:
            if t % self.a_update_freq ==0: # update Actor per 3-steps 
                if self.rnn_mode == 'lstm':
                    xh_b0, _ = self.agent.rnn(state0_cuda, (a_hx, a_cx))
                    behavior_action = self.agent.actor(xh_b0)
                    ### Estimate actor action ###
                    q_action = self.agent.critic([xh_b0, action.cuda()])  #這邊的critic是behavior_critic，有別於target_critic
                    
                    ### Calculate Actor loss based on Q-value ###
                    actor_loss = -self.agent.critic([xh_b0, behavior_action])
                    
                    ### calculate q_actor_loss for priority
                    q_actor_loss = self.agent.critic([xh_b0, behavior_action])
                    
                    ##### Behavior Cloning Loss #####
                    if self.is_BClone:
                        ### Estimate prophetic action ###
                        q_action_bc = self.agent.critic([xh_b0, action_bc.cuda()])
                        
                        ### Q_filter & BC_loss ###
                        BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
                        BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
                        
                        Q_filter = torch.gt(q_action_bc, q_action)
                        BC_loss_Qf = BC_loss * (Q_filter.detach())
                        if self.is_Qfilt:
                            ### modified Policy loss ###
                            policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf) 
                        else:
                            ### modified Policy loss ###
                            policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                            
                    else:  ### Original Policy loss ###
                        policy_loss = actor_loss
                    
                elif self.rnn_mode == 'gru':
                    xh_b0, _ = self.agent.rnn(state0_cuda, a_hx)
                    behavior_action = self.agent.actor(xh_b0)
                    ### Estimate actor action ###
                    q_action = self.agent.critic([xh_b0, action.cuda()])  #這邊的critic是behavior_critic，有別於target_critic
                    
                    ### Calculate Actor loss based on Q-value ###
                    behavior_action = self.agent.actor(xh_b0)
                    actor_loss = -self.agent.critic([xh_b0, behavior_action])
                    
                    ### calculate q_actor_loss for priority
                    q_actor_loss = self.agent.critic([xh_b0, behavior_action])
                    
                    ##### Behavior Cloning Loss #####
                    if self.is_BClone:
                        ### Estimate prophetic action ###
                        q_action_bc = self.agent.critic([xh_b0, action_bc.cuda()])
                        
                        ### Q_filter & BC_loss ###
                        BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
                        BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
                        
                        Q_filter = torch.gt(q_action_bc, q_action)
                        BC_loss_Qf = BC_loss * (Q_filter.detach())
                        if self.is_Qfilt:
                            ### modified Policy loss ###
                            policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf)
                        else:
                            ### modified Policy loss ###
                            policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                            
                    else:  ### Original Policy loss ###
                        policy_loss = actor_loss
                
                ################## Actor loss calculation ##################
                if self.is_BClone:
                    BC_loss /= t_len
                    BC_loss_total +=  BC_loss.mean()  #BC loss
                    BC_loss_Qf  /= t_len
                    BC_loss_Qf_total += BC_loss_Qf.mean()
                    actor_loss /= t_len
                    actor_loss_total += actor_loss.mean()   #actor loss
                else:
                    BC_loss_total = torch.zeros(1)
                    BC_loss_Qf_total = torch.zeros(1)
                    actor_loss_total = torch.zeros(1)
                
                policy_loss /= t_len # divide by experience length
                policy_loss_total += policy_loss.mean()
    
                ####### Update Actor ###########
                self.agent.rnn.zero_grad()
                self.agent.actor.zero_grad()
                self.agent.critic.zero_grad()
                policy_loss = policy_loss.mean()
                policy_loss.backward()
                self.actor_optim.step()
                self.rnn_optim.step()  
                
        ##### Learning rate Scheduling #####
        self.rnn_scheduler.step()
        self.critic_scheduler.step()
        self.actor_scheduler.step()
        

        ###Update priority###
        if self.is_PER_replay == True:
            TDerror_square = (target_q.cpu() - current_q.cpu()).pow(2)
            loss2actor_square = q_actor_loss.cpu().pow(2)
            
            self.priority = (TDerror_square + self.lambda_balance*loss2actor_square).detach().numpy().ravel() + self.small_const
            self.priority[batch_flags == DEMO_flag] += self.priority_const   
            
            self.memory.update_priorities(idxes, self.priority)
            self.demoN_ratio = np.sum(demo_cnt) / t_len
            
        ########### Record all losses ############
        self.actor_loss = actor_loss_total.item()
        self.BC_loss = BC_loss_total.item()
        self.BC_loss_Qf = BC_loss_Qf_total.item()
        self.tot_policy_loss = policy_loss_total.item()
        self.critic_loss = value_loss_total.item()
        ##### 以下的batch期望值是整條segmt_traj，但因為有照time_order後，會不會就不符i.i.d.了？ #####
        ##### update once after experience (segmt_traj) ##### (雖有問題，但也值得try看看)
#        policy_loss_total /= self.batch_size # divide by number of trajectories
#        value_loss_total /= self.batch_size # divide by number of trajectories
#
#        self.agent.critic.zero_grad()
#        value_loss_total.backward()
#        self.critic_optim.step()
#
#        self.agent.actor.zero_grad()
#        policy_loss_total.backward()
#        self.actor_optim.step()

        ##### Target_Net update #####
        soft_update(self.agent.rnn_target, self.agent.rnn, self.tau)
        soft_update(self.agent.actor_target, self.agent.actor, self.tau)
        soft_update(self.agent.critic_target, self.agent.critic, self.tau)

    
    def test(self, model_path, model_fn, description, lackM=True, debug=False):
        
        if self.agent.load_weights(model_path, model_fn) == False:
            prRed("model path not found")
            return

        self.agent.is_training = False
        self.agent.eval()

        with torch.no_grad():
            test_mean_reward = self.evaluate(self.env, self.agent, description, lackM=lackM, debug=debug, save=False)
            if debug: prYellow('[Evaluate]: mean_reward:{}'.format(test_mean_reward))

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
    
    def train_plot_bc(self, episode, train_epi_reward, train_ewma_reward, 
                      train_totPolicy_loss, train_critic_loss,
                      train_actor_loss, train_bc_loss, train_bcQf_loss):
        font_size = 16
        plt.figure(num=1, figsize=(12, 6))
        
        plt.subplot(321)
        plt.title('Episode Reward', fontsize=font_size)
        plt.plot(train_epi_reward)
        
        plt.subplot(322)
        plt.title('EWMA Reward', fontsize=font_size)
        plt.plot(train_ewma_reward)
        
        plt.subplot(323)
        plt.title('total Policy Loss', fontsize=font_size)
        plt.plot(train_totPolicy_loss)
        
        plt.subplot(324)
        plt.title('Critic Loss', fontsize=font_size)
        plt.plot(train_critic_loss)
        
        plt.subplot(325)
        plt.title('Actor Loss', fontsize=font_size)
        plt.plot(train_actor_loss)
        
        plt.subplot(326)
        plt.title('BC Loss', fontsize=font_size)
        bc, = plt.plot(train_bc_loss, label='BC')
        Qf, = plt.plot(train_bcQf_loss, label='Qf')
        plt.legend(handles=[bc, Qf], loc='upper center', fontsize=10)
        # plt.legend([bc, Qf], ["BC", "Qf"], loc='upper left') #, facecolor='blue')
        # plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
        
        plt.tight_layout()
        train_his_fn = 'lamBC_' +str(np.round(self.lambda_BC,3)) +'_' +self.rnn_mode +'_' +str(self.date)
        plt.savefig('results/TrainCurve_epi' +str(episode) +'_' +train_his_fn +'.jpg')
        plt.close()
        
        ##### Save Training History to csv file #####
        # epi_r = np.array(train_epi_reward)
        # ewma_r = np.array(train_ewma_reward)
        # totP_loss = np.array(train_totPolicy_loss)
        # c_loss = np.array(train_critic_loss)
        # a_loss = np.array(train_actor_loss) 
        # bc_loss = np.array(train_bc_loss)
        # qf_loss = np.array(train_bcQf_loss) 
        
        dic = {'Episode Reward':train_epi_reward,
               'EWMA Reward':train_ewma_reward,
               'total Policy Loss':train_totPolicy_loss,
               'Critic Loss':train_critic_loss,
               'Actor Loss':train_actor_loss,
               'BC Loss':train_bc_loss,
               'BC_Qf Loss':train_bcQf_loss
              }
        train_history = pd.DataFrame(dic)
        train_history.to_csv('results/TrainHis_' +train_his_fn +'.csv',  index=False)
         
    def train_plot(self, episode, train_epi_reward, train_ewma_reward, train_totPolicy_loss, train_critic_loss):
        font_size = 16
        plt.figure(num=1, figsize=(12, 7))
        
        ax1 = plt.subplot(221)
        ax1.set_title('Episode Reward', fontsize=font_size)
        plt.plot(train_epi_reward)
        
        plt.subplot(222)
        plt.title('EWMA Reward', fontsize=font_size)
        plt.plot(train_ewma_reward)
        
        plt.subplot(223)
        plt.title('total Policy Loss', fontsize=font_size)
        plt.plot(train_totPolicy_loss)
        
        plt.subplot(224)
        plt.title('Critic Loss', fontsize=font_size)
        plt.plot(train_critic_loss)
        
        plt.tight_layout()
        train_his_fn = '_' +self.rnn_mode +'_' +str(self.date)
        plt.savefig('results/TrainCurve_epi' +str(episode) +train_his_fn +'.jpg')
        plt.close()
        
        ##### Save Training History to csv file #####
        dic = {'Episode Reward':train_epi_reward,
               'EWMA Reward':train_ewma_reward,
               'total Policy Loss':train_totPolicy_loss,
               'Critic Loss':train_critic_loss,
              }
        train_history = pd.DataFrame(dic)
        train_history.to_csv('results/TrainHis_' +train_his_fn +'.csv',  index=False)
        
    def train_demoN_ratio(self, episode, demoN_ratio_batch):
        plt.xlabel('# of episode')
        plt.ylabel('batch_demoN_ratio')
        plt.plot(demoN_ratio_batch)
        fig_fn = 'batch_demoN_ratio_ep' +'.jpg'
        plt.savefig(fig_fn)
        plt.close()
        
        

