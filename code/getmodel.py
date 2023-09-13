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

device = torch.device("cpu") 
torch.autograd.set_detect_anomaly(True)
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

class agent_actor_critic:
    def __init__(self,env):
        self.env=env
        self.state_size = NUM_STATE
        self.action_size = NUM_ACTION
        self.lr = 0.001
        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.critic = Critic(self.state_size, self.action_size).to(device)
        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerC = optim.Adam(self.critic.parameters())
        self.exploit_rate_files = open(
            RESULT_DIR + MyGlobals.folder_name + "exploit_rate.csv", "w")
        self.exploit_rate_files.write('1')
        for i in range(2, NUM_ACTION + 1):
            self.exploit_rate_files.write(',' + str(i))
        self.exploit_rate_files.write('\n')

    def compute_returns(self,next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    def make(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
    def count(self):
        self.count_exploit = [0] * NUM_ACTION
    def take_action(self):
        return self.dist.sample()
    def move(self,state):
        self.state = torch.FloatTensor(state).to(device)
        self.dist, self.value = self.actor(self.state), self.critic(self.state)

        self.action = self.dist.sample()
        self.prob_dist = self.dist.probs
        # print(type(prob_dist))
        # print(prob_dist)
        # print(action)
        # print(torch.topk(prob_dist.flatten(), NUM_ACTION).indices.tolist())
        # print(torch.topk(prob_dist.flatten(), NUM_ACTION).indices.tolist().index(action))
        # print(torch.topk(
        #     prob_dist.flatten(), NUM_ACTION))
        # assert 2 == 3
        self.count_exploit[torch.topk(
            self.prob_dist.flatten(), NUM_ACTION).indices.tolist().index(self.action)] += 1
        # if action == dist.probs.argmax():
        #     count_exploit += 1
        self.next_state, self.reward, self.done, _ = self.env.step(
            self.action.cpu().numpy())
        self.state = self.next_state
        

    def optimize(self,gamma):
        self.log_prob = self.dist.log_prob(self.action).unsqueeze(0)
        self.log_probs.append(self.log_prob)
        self.values.append(self.value)
        self.rewards.append(torch.tensor(
            [self.reward], dtype=torch.float, device=device))
        self.masks.append(torch.tensor(
            [1-self.done], dtype=torch.float, device=device))
        self.next_state = torch.FloatTensor(self.next_state).to(device)    
        
        self.next_value = self.critic(self.next_state)
        self.returns = self.compute_returns(
            self.next_value, self.rewards, self.masks, gamma=gamma)

        self.log_probs_cat = torch.cat(self.log_probs)
        self.returns_cat = torch.cat(self.returns).detach()
        self.values_cat = torch.cat(self.values)

        self.advantage = self.returns_cat - self.values_cat

        self.actor_loss = -(self.log_probs_cat * self.advantage.detach()).mean()
        self.critic_loss = self.advantage.pow(2).mean()

        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        self.critic_loss.backward(retain_graph=True)
        self.actor_loss.backward(retain_graph=True)
        self.optimizerA.step()
        self.optimizerC.step()
        


def get_model(name,env):
    if name=="actor_critic":
        return agent_actor_critic(env)