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
import matplotlib
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class ReplayMemory(object):
    def __init__(self,capacity) -> None:
        self.memory=deque([],maxlen=capacity)
    def push(self,state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
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
        self.env.seed(123)
        self.batch_size=128
        self.eps_start=0.9
        self.eps_end=0.05
        self.eps_decay=1000
        self.tau=0.05
        self.lr=1e-4
        self.n_actions=NUM_ACTION
        self.n_observations=NUM_STATE
        self.model1=MultiOutputRegressor(RandomForestClassifier(n_estimators=500, random_state=42))
        self.model2=MultiOutputRegressor(RandomForestClassifier(n_estimators=500, random_state=42))
        self.isFit=False
        self.memory=ReplayMemory(10000)
        self.episode_durations = []
        self.stepdone=0
        self.gamma=0.001
    def select_action(self,state):
            sample=random.random()
            eps_threshold=self.eps_end+(self.eps_start-self.eps_end)*math.exp(-1.*self.stepdone/self.eps_decay)
            self.stepdone+=0.05
            if sample > eps_threshold:
                with torch.no_grad():
                    if self.isFit==True:
                        q_value= self.model1.predict(state)
                    else: 
                        q_value=np.zeros(self.n_actions)
                    return np.argmax(q_value[0])
            else:
                return random.randrange(self.n_actions)   

    def optimize_model(self,train=True):
            
            if len(self.memory) < self.batch_size:
                return
            batch = self.memory.sample(self.batch_size)
            X = []
            targets = []
            for state, action, next_state, reward,terminal in batch:
                q_update = reward
                if  not terminal :
                    if self.isFit:
                        q_update = (reward + self.gamma * np.amax(self.model2.predict(next_state)[0]))
                    else:
                        q_update = reward
                if self.isFit:
                    q_values = self.model1.predict(state)
                else:
                    q_values = np.zeros(self.n_actions).reshape(1, -1)
                q_values[0][action] = q_update
                
                X.append(list(state[0]))
                targets.append(q_values[0])
            self.model1.fit(X, targets)
            if train:
                self.model2.fit(X,targets)
            self.isFit = True

            
    def train(self,num_iters,num_episodes,duration):
    
            for iter in range(num_iters):
                self.env.replay()
                for episode in range(num_episodes):
                    state = self.env.reset()
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    done = False
                    count_exploit = [0] * NUM_ACTION
                    while not done:
                        for i in count():
                            action = self.select_action(state)
                            next_state, reward, done, _= self.env.step(np.array(action))
                            reward = torch.tensor([reward], device=device)
                            if done:
                                if (self.env.old_avg_reward < -1500):
                                    return
                                next_state = None
                                print('Episode: {}, Score: {}'.format(
                                    episode, self.env.old_avg_reward))

                                break        
                            if (i > duration):
                                break
                            next_state = np.reshape(next_state, [1, Agent.n_observations])
                            self.memory.push(state, action, next_state, reward,done)
                            state = next_state

                            self.optimize_model(train=True)

    def test(self,num_episodes):
            
            if (self.env.old_avg_reward < -1500):
                return
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False

                while not done:
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action=action = self.select_action(state)
                    next_state, reward, done, _= self.env.step(np.array(action))

                    state = next_state

                print('Test Episode: {}, Score: {}'.format(episode, self.env.old_avg_reward))
    def runAC(self,i, dur):
    # MyGlobals.folder_name = "Actor_Critic_800_30s/dur" + str(dur) + "/" + str(i) +'/'
        self.train(num_iters=9, num_episodes=121,
            duration=dur )
        self.test( num_episodes=31, )

Agent=DQNAgent()
Agent.runAC(1,30)
        