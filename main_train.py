import numpy as np
import argparse
from copy import deepcopy
import random
import torch
from timeit import default_timer as timer

from evaluator import Evaluator
from rdpg import RDPG
from util import *
from environment import environment

torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Financial trading--iRDPG algorithm')
    
    ##### Model Setting #####
    # parser.add_argument('--rnn_mode', default='lstm', type=str, help='RNN mode: LSTM/GRU')
    parser.add_argument('--rnn_mode', default='gru', type=str, help='RNN mode: LSTM/GRU')
    parser.add_argument('--input_size', default=14, type=int, help='num of features for input state')
    parser.add_argument('--seq_len', default=15, type=int, help='sequence length of input state')
    parser.add_argument('--num_rnn_layer', default=2, type=int, help='num of rnn layer')
    parser.add_argument('--hidden_rnn', default=128, type=int, help='hidden num of lstm layer')
    parser.add_argument('--hidden_fc1', default=256, type=int, help='hidden num of 1st-fc layer')
    parser.add_argument('--hidden_fc2', default=64, type=int, help='hidden num of 2nd-fc layer')
    parser.add_argument('--hidden_fc3', default=32, type=int, help='hidden num of 3rd-fc layer')
    parser.add_argument('--init_w', default=0.005, type=float, help='initialize model weights') 
    
    ##### Learning Setting #####
    parser.add_argument('--r_rate', default=0.0001, type=float, help='gru layer learning rate')  
    parser.add_argument('--c_rate', default=0.0001, type=float, help='critic net learning rate') 
    parser.add_argument('--a_rate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--beta1', default=0.3, type=float, help='mometum beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.9, type=float, help='mometum beta2 for Adam optimizer')
    parser.add_argument('--sch_step_size', default=16*150, type=float, help='LR_scheduler: step_size')
    parser.add_argument('--sch_gamma', default=0.5, type=float, help='LR_scheduler: gamma')
    parser.add_argument('--bsize', default=100, type=int, help='minibatch size')
    
    ##### RL Setting #####
    parser.add_argument('--warmup', default=100, type=int, help='only filling the replay memory without training')
    parser.add_argument('--discount', default=0.95, type=float, help='future rewards discount rate')
    parser.add_argument('--a_update_freq', default=3, type=int, help='actor update frequecy (per N steps)')
    parser.add_argument('--Reward_max_clip', default=15., type=float, help='max DSR reward for clipping')
    parser.add_argument('--tau', default=0.002, type=float, help='moving average for target network')
    ##### original Replay Buffer Setting #####
    parser.add_argument('--rmsize', default=12000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')  
    ##### Exploration Setting #####
    parser.add_argument('--ou_theta', default=0.18, type=float, help='noise theta of Ornstein Uhlenbeck Process')
    parser.add_argument('--ou_sigma', default=0.3, type=float, help='noise sigma of Ornstein Uhlenbeck Process') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu of Ornstein Uhlenbeck Process') 
    parser.add_argument('--epsilon_decay', default=100000, type=int, help='linear decay of exploration policy')
    
    ##### Training Trajectory Setting #####
    parser.add_argument('--exp_traj_len', default=16, type=int, help='segmented experiece trajectory length')  
    parser.add_argument('--train_num_episodes', default=2000, type=int, help='train iters each episode')  
    ### Also use in Test (Evaluator) Setting ###
    parser.add_argument('--max_episode_length', default=240, type=int, help='the max episode length is 240 minites in one day')  
    parser.add_argument('--test_episodes', default=243, type=int, help='how many episode to perform during testing periods')
    
    ##### PER Demostration Buffer #####
    parser.add_argument('--is_PER_replay', default=True, help='conduct PER momery or not')
    parser.add_argument('--is_pretrain', default=True, action='store_true', help='conduct pretrain or not')
    parser.add_argument('--Pretrain_itrs', default=100, type=int, help='number of pretrain iterations')
    parser.add_argument('--is_demo_warmup', default=True, action='store_true', help='Execute demonstration buffer')
    parser.add_argument('--PER_size', default=40000, type=int, help='memory size for PER')
    parser.add_argument('--p_alpha', default=0.3, type=int, help='the power of priority for each experience')
    parser.add_argument('--lambda_balance', default=50, type=int, help='priority coeffient for weighting the gradient term')
    parser.add_argument('--priority_const', default=0.1, type=int, help='priority constant for demonstration experiences')
    parser.add_argument('--small_const', default=0.001, type=int, help='priority constant for agent experiences')
    
    ##### Behavior Cloning #####
    parser.add_argument('--is_BClone', default=True, action='store_true', help='conduct behavior cloning or not')
    parser.add_argument('--is_Qfilt', default=False, action='store_true', help='conduct Q-filter or not')
    parser.add_argument('--use_Qfilt', default=100, type=int, help='set the episode after warmup to use Q-filter')
    parser.add_argument('--lambda_Policy', default=0.7, type=int, help='The weight for actor loss')
    # parser.add_argument('--lambda_BC', default=0.5, type=int, help='The weight for BC loss after Q-filter, default is equal to (1-lambda_Policy)')
    
    ##### Other Setting #####
    parser.add_argument('--seed', default=627, type=int, help='seed number')
    parser.add_argument('--date', default=629, type=int, help='date for output file name')
    parser.add_argument('--save_threshold', default=20, type=int, help='lack margin stop ratio')
    parser.add_argument('--lackM_ratio', default=0.7, type=int, help='lack margin stop ratio')
    parser.add_argument('--debug', default=True, dest='debug', action='store_true')
    parser.add_argument('--checkpoint', default="checkpoints", type=str, help='Checkpoint path')
    parser.add_argument('--logdir', default='log')
    # parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    
    
    args = parser.parse_args()
    #######################################################################################################

    ####################################################################################################
    '''##### Run Task #####'''
    if args.seed > 0:
        np.random.seed(args.seed)
        random.seed(args.seed)

    is_lack_margin = True
    # is_lack_margin = False
    
    ##### Demonstration Setting #####
    if args.is_demo_warmup:
        data_fn = "data_preprocess/IF_tech_oriDT.csv"
        demo_env = environment(data_fn=data_fn, data_mode='random', duration='train', is_demo=True, 
                               is_intraday=True, is_lack_margin=is_lack_margin, args=args)
    else:
        demo_env = None
        
        
    ##### Run Training #####
    start_time = timer()
    if args.mode == 'train':
        print('##### Run Training #####')
        ### train_env setting ###
        data_mode = 'random'  # random select a day for a trading episode (240 minutes)
        duration = 'train'  # training period from 2016/1/1 to 2018/5/8
        
        data_fn = "data_preprocess/IF_prophetic.csv"
        train_env = environment(data_fn=data_fn, data_mode=data_mode, duration=duration, is_demo=False, 
                                is_intraday=True, is_lack_margin=is_lack_margin, args=args)
        
        ### Run training ###
        rdpg = RDPG(demo_env, train_env, args)
        rdpg.train(args.train_num_episodes, args.checkpoint, args.debug)
        
        end_time = timer()
        minutes, seconds = (end_time - start_time)//60, (end_time - start_time)%60
        print(f"\nTraining time taken: {minutes} minutes {seconds:.1f} seconds")

        
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


