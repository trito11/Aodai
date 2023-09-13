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
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from getmodel import get_model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

state_size = NUM_STATE
action_size = NUM_ACTION
lr = 0.001
eps_start=0.9
eps_end=0.05
eps_decay=1000


class DQNAgent:
    def __init__(self) :
        self.env=MixStateEnv()
    def train(self,num_iters,num_episodes,duration,gamma):
        self.model=get_model("actor_critic",self.env)
        for iter in range(num_iters):
            self.env.replay()
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False
                self.model.count()
                while not done:
                    self.model.make()
                    for i in count():
                        self.model.move(state)
                        done=self.model.done
                        if done:
                            if (self.model.env.old_avg_reward < -1500):
                                return
                            self.model.next_state = None
                            print('Episode: {}, Score: {}'.format(
                                episode, self.model.env.old_avg_reward))

                                # print(dist.probs)
                                #print('Iteration: {}, Score: {}'.format(episode, i))
                            break
                        if (i > duration):
                            break
                        self.model.optimize(gamma)
                tempstr = ','.join([str(elem) for elem in self.model.count_exploit])
                self.model.exploit_rate_files.write(tempstr+"\n")
                # exploit_rate_files.write('{}\n'.format(count_exploit))
                print(tempstr)
        self.model.exploit_rate_files.close()
    def test(self,num_episodes):
            if (self.model.env.old_avg_reward < -1500):
                return
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False
                while not done:
                    state = torch.FloatTensor(state).to(device)
                    self.model.move(state)
                    done=self.model.done

                print('Test Episode: {}, Score: {}'.format(episode, self.env.old_avg_reward))
    def runAC(self,i, dur,gamma):
    # MyGlobals.folder_name = "Actor_Critic_800_30s/dur" + str(dur) + "/" + str(i) +'/'
        MyGlobals.folder_name = f"test/gamma{gamma}/dur{dur}/{i}/"
        self.train(num_iters=9, num_episodes=121,
            duration=dur,gamma=gamma )
        self.test( num_episodes=31, )

Agent=DQNAgent()
Agent.runAC(1,30,0.99)
      