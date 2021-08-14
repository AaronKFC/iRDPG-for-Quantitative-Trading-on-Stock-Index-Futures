
import numpy as np
import torch
from model import (RNN, Actor, Critic)
from random_process import OrnsteinUhlenbeckProcess
from util import *
from scipy.special import softmax


class Agent(object):
    def __init__(self, args):
        nb_actions = 2
        self.date = args.date
        # if args.seed > 0:
        #     self.seed(args.seed)
        
        ##### Create RNN Layer #####
        self.rnn = RNN(args)
        self.rnn_target = RNN(args)
        ##### Create Actor Network #####
        self.actor = Actor(args)
        self.actor_target = Actor(args)
        ##### Create Critic Network #####
        self.critic = Critic(args)
        self.critic_target = Critic(args)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Hyper-parameters
        self.is_training = True
        self.is_BClone = args.is_BClone
        self.rnn_mode = args.rnn_mode
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon_decay
        self.epsilon = 1.0
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        if torch.cuda.is_available() : 
            self.cuda()
            print('USE CUDA')
        

    def eval(self):
        self.rnn.eval
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self):
        action = np.random.random(size=(2))
        action = softmax(action)
        return action, self.epsilon

    def select_action(self, state, noise_enable=True, decay_epsilon=True):
        xh, _ = self.rnn(state)
        action = self.actor(xh)
        
        action = to_numpy(action.cpu()).squeeze(0)
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()
            action = softmax(action)
            
        # print(action)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return action, self.epsilon
    
    def reset_rnn_hidden(self, done=True):
        self.rnn.reset_hidden_state(done)

    def reset(self):
        self.random_process.reset_states()

    def cuda(self):
        #device = torch.device('cuda:0')
        self.rnn.cuda()
        self.rnn_target.cuda()
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def load_weights(self, checkpoint_path, model_fn):
        if checkpoint_path is None: return False
        
        model_path = checkpoint_path +'/test_case/' +model_fn
        model = torch.load(model_path)
        self.rnn.load_state_dict(model['rnn'])
        self.actor.load_state_dict(model['actor'])
        self.critic.load_state_dict(model['critic'])

        return True

    def save_model(self, checkpoint_path, episode, ewma_reward):
        e_reward = int(np.round(ewma_reward)) #(ewma_reward,2)
        description = '_' +self.rnn_mode +'_' +'ep' +str(episode) +'_' +'rd' +str(e_reward) +'_' +str(self.date) +'.pkl'
        if self.is_BClone:
            description = '_BC' +description
        model_path = checkpoint_path +'/' +description
        torch.save({'rnn': self.rnn.state_dict(),
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    # 'actor_target': self.actor_target.state_dict(),
                    # 'critic_target': self.critic_target.state_dict(),
                    # 'rnn_opt': self.rnn_optim.state_dict(),
                    # 'actor_opt': self.actor_optim.state_dict(),
                    # 'critic_opt': self.critic_optim.state_dict(),
                    }, model_path)


