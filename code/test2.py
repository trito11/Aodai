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


class DQNnet(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQNnet,self).__init__()
        self.layer1=nn.Linear(n_observations,128)
        self.layer2=nn.Linear(128,256)
        self.layer3=nn.Linear(256,128)
        self.layer4=nn.Linear(128,n_actions)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        return self.layer4(x)
policy_net=DQNnet(14,7)
optimizer=optim.AdamW(filter(lambda p: p.requires_grad, policy_net.parameters()),lr=1e-4,amsgrad=True)
for param in policy_net.layer_to_freeze.parameters():
    print(param)
    param.requires_grad = False