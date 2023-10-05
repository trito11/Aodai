import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from environment import *
from mix_state_env import MixStateEnv
from config import *
import copy
from MyGlobal import MyGlobals
from itertools import count
from torch.distributions import Categorical
import random
import math
from collections import namedtuple, deque
import pickle

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
result_train2=list()
result_test2=list()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

state_size = NUM_STATE
action_size = NUM_ACTION
lr = 0.001
eps_start=0.9
eps_end=0.05
eps_decay=1000


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
    def push(self, *args):
        self.memory.append(self.Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
replay = ReplayMemory(10000)
class DQNnet(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQNnet,self).__init__()
        self.layer1=nn.Linear(n_observations,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,n_actions)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self) :
        self.env=MixStateEnv()
        self.optimize=0
        self.env.seed(123)
        self.batch_size=128
        self.eps_start=0.9
        self.eps_end=0.05
        self.eps_decay=1000
        self.tau_start=0.5
        self.tau_end=0.01
        self.tau_decay=1000
        self.tau1=0.05
        self.tau2=0.
        self.lr=1e-4
        self.n_actions=NUM_ACTION
        self.n_observations=NUM_STATE
        self.policy_net=DQNnet(self.n_observations,self.n_actions).to(device)
        self.target_net=DQNnet(self.n_observations,self.n_actions).to(device)
        self.save_net=DQNnet(self.n_observations,self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.save_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer=optim.AdamW(self.policy_net.parameters(),lr=self.lr,amsgrad=True)
        self.memory=ReplayMemory(10000)
        self.stepdone=0
        self.stepdone1=0
    def select_action(self,state):
            sample=random.random()
            eps_threshold=self.eps_end+(self.eps_start-self.eps_end)*math.exp(-1.*self.stepdone/self.eps_decay)
            self.stepdone+=0.05
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(-1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randint(0,NUM_ACTION-1)]], device=device, dtype=torch.long)    

    def optimize_model(self,gamma):
            # self.start_time=time.time()
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            batch_state=[]
            for i in batch.state :
                batch_state.append(i.view(1,15))
            state_batch = torch.cat(batch_state)
            action_batch = torch.cat(batch.action)

           
            reward_batch = torch.cat(batch.reward)

            
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.batch_size, device=device)
            with torch.no_grad():
                 next_state_values[non_final_mask] =self.target_net(non_final_next_states).max(1)[0]
                # next_state_values[non_final_mask] =self.policy_net(non_final_next_states).gather(1,self.target_net(non_final_next_states).max(1)[1].unsqueeze(0))
                
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * gamma) + reward_batch
            

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            self.optimize+=1
            # self.end_time=time.time()
            # print(f"optimze_time:{self.end_time-self.start_time},times{self.optimize}")
    def train(self,num_iters,num_episodes,duration,gamma):
    
            for iter in range(num_iters):
                self.env.replay()
                for episode in range(num_episodes):
                    state = self.env.reset()
                    done = False
                    while not done:
                        for i in count():
                            state = torch.FloatTensor(state).to(device)
                            action = self.select_action(state)
                            action1=action.item()
                            next_state, reward, done, _= self.env.step(np.array(action1))
                            reward = torch.tensor([reward], device=device)
                            if done:
                                if (self.env.old_avg_reward < -1500):
                                    return
                                next_state = None
                                print('Episode: {}, Score: {}'.format(
                                    episode, self.env.old_avg_reward))
                                result_train2.append(self.env.old_avg_reward*-800)

                                break
                            
                                
                            if (i > duration):
                                break
                            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                            self.memory.push(state, action, next_state, reward)
                            state = next_state

                            self.optimize_model(gamma)
                            self.tau=self.tau_end+(self.tau_start-self.tau_end)*math.exp(-1.*self.stepdone1/self.tau_decay)
                            self.stepdone1+=0.005
             
                            
                            target_net_state_dict = self.target_net.state_dict()
                            policy_net_state_dict = self.policy_net.state_dict()
                            save_net_state_dict = self.save_net.state_dict()
                            
                            for key in policy_net_state_dict:
                                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau1 + target_net_state_dict[key]*(1-self.tau1)
                                save_net_state_dict[key] = policy_net_state_dict[key]*self.tau2 + save_net_state_dict[key]*(1-self.tau2)

                            self.target_net.load_state_dict(target_net_state_dict)
                            self.save_net.load_state_dict(save_net_state_dict)
                            self.policy_net.load_state_dict(save_net_state_dict)

    def test(self,num_episodes):
            
            if (self.env.old_avg_reward < -1500):
                return
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False

                while not done:
                    state = torch.FloatTensor(state).to(device)
                    action=action = self.select_action(state)
                    action1=action.item()
                    next_state, reward, done, _= self.env.step(np.array(action1))

                    state = next_state

                print('Test Episode: {}, Score: {}'.format(episode, self.env.old_avg_reward))
                result_test2.append(self.env.old_avg_reward*-800)
    def runAC(self,i, dur,gamma):
    # MyGlobals.folder_name = "Actor_Critic_800_30s/dur" + str(dur) + "/" + str(i) +'/'
        MyGlobals.folder_name = f"test/gamma{gamma}/dur{dur}/{i}/"
        self.train(num_iters=9, num_episodes=121,
            duration=dur,gamma=gamma )
        self.test( num_episodes=60, )

Agent=DQNAgent()
Agent.runAC(1,10,0.99)