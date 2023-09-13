import numpy as np
np.float = float
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier,iSOUPTreeRegressor
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
from xgboost import XGBRegressor
import xgboost as xgb
import joblib
import time
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
file = open("myfile.txt", "w")
def huber_loss(y_true, y_pred, delta):
    residual = y_true - y_pred
    condition = np.abs(residual) <= delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * (np.abs(residual) - 0.5 * delta)
    return np.where(condition, squared_loss, linear_loss)

def huber_loss_gradient(y_true, y_pred, delta):
    residual = y_true - y_pred
    condition = np.abs(residual) <= delta
    gradient = residual
    gradient[~condition] = delta * np.sign(residual[~condition])
    return gradient

def huber_loss_hessian(y_true, y_pred, delta):
    residual = y_true - y_pred
    condition = np.abs(residual) <= delta
    hessian = np.ones_like(residual)
    hessian[~condition] = 0
    return hessian
def custom_obj(y_true, y_pred):
    obj=lambda y_true, y_pred: huber_loss(y_true, y_pred, delta),
    grad=lambda y_true, y_pred: huber_loss_gradient(y_true, y_pred, delta),
    hess=lambda y_true, y_pred: huber_loss_hessian(y_true, y_pred, delta)
# Create the custom loss function object
delta = 1.0  # Threshold parameter


# Configure the XGBoost model with the custom Huber loss function



plt.ion()
# def is_max_tree_reached(multioutput_model):
#     # Get the maximum number of trees allowed
#     max_trees = multioutput_model.estimators_[0].get_params()['n_estimators']
#     # Check if the current number of trees in each output regressor is equal to the maximum
#     for estimator in multioutput_model.estimators_:
#         current_trees = estimator.get_booster().best_ntree_limit
#         if current_trees != max_trees:
#             return False
    
    # return True
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
class DQNAgent:
    def __init__(self) :
        self.optimize=0
        self.max=False
        self.env=MixStateEnv()
        self.env.seed(123)
        self.batch_size=2000
        self.eps_start=0.9
        self.eps_end=0.05
        self.eps_decay=1000
        self.tau=0.05
        self.lr=1e-4
        self.n_actions=NUM_ACTION
        self.n_observations=NUM_STATE
        self.isFit=False
        self.memory=ReplayMemory(10000)
        self.episode_durations = []
        self.stepdone=0
        self.delta = 1.0  # Threshold parameter
    def huber_loss(self,y_true, y_pred, delta):
        residual = self.targets-self.values
        if abs(residual) <= delta:
            return 0.5 * residual**2
        else:
            return delta * (abs(residual) - 0.5 * delta)
    def huber_loss_gradient(self,y_true, y_pred):
        residual = self.targets-self.values
        if abs(residual) <= self.delta:
            return residual
        else:
            return delta * (residual / abs(residual))

    def huber_loss_hessian(self,y_true, y_pred):
        residual = self.targets-self.values
        if abs(residual) <= self.delta:
            return 1.0
        else:
            return delta / abs(residual)
    def custom_obj(self,y_true, y_pred):
        obj=lambda y_true, y_pred: self.huber_loss(y_true, y_pred),
        grad=lambda y_true, y_pred: self.huber_loss_gradient(y_true, y_pred),
        hess=lambda y_true, y_pred: self.huber_loss_hessian(y_true, y_pred)
# Create the custom loss function object
 
    def select_action(self,state):
            sample=random.random()
            eps_threshold=self.eps_end+(self.eps_start-self.eps_end)*math.exp(-1.*self.stepdone/self.eps_decay)
            self.stepdone+=0.005
            if sample > eps_threshold:
                with torch.no_grad():
                    if self.isFit==True:
                        q_value= self.model1.predict(state)
                    else: 
                        q_value=np.zeros(self.n_actions)
                    return np.argmax(q_value[0])
            else:
                return random.randrange(self.n_actions)
    def optimize_model(self,upgrade):
            if len(self.memory) < self.batch_size:
                return
            batch = self.memory.sample(self.batch_size)
            X = []
            self.targets = []
            self.values=[]
            for state, action, next_state, reward, terminal in batch:
                # q_update = reward
                # if  not terminal :
                if self.isFit:
                    self.q_update = (reward + self.gamma * np.amax(self.model2.predict(next_state)[0]))
                else:
                    self.q_update = reward
                if self.isFit:
                    self.q_values = self.model1.predict(state)
                else:
                    self.q_values = np.zeros(self.n_actions).reshape(1, -1)
                self.values.append(self.q_values[0])
                self.q_values[0][action] = self.q_update
                
                X.append(list(state[0]))
                self.targets.append(self.q_values[0])
            X=np.array(X)
            self.targets=np.array(self.targets)
            self.calculate=time.time()
            if upgrade:
                joblib.dump(self.model1, 'initial_model.pkl')
                self.model2=joblib.load('initial_model.pkl')
                # self.model1=MultiOutputRegressor(XGBRegressor(learning_rate=self.learning_rate,n_estimators=200,max_depth=10,n_jobs=1,tree_method='gpu_hist'))
                # self.model2=MultiOutputRegressor(XGBRegressor(learning_rate=self.learning_rate,n_estimators=200,max_depth=10,n_jobs=1,tree_method='gpu_hist'))
            if self.isFit:
                self.model1.partial_fit(X, self.targets)
            else:
                self.model1.fit(X, self.targets)
                self.model2.fit(X, self.targets)
            self.optimize+=1
            self.isFit = True   
            self.end_time=time.time()
            # print(f"optimze_time:{self.end_time-self.start_time},calculate:{self.calculate-self.start_time};times:{self.optimize}")
    def train(self,num_iters,num_episodes,duration,tree,learning_rate,gamma):
            self.learning_rate=learning_rate
            self.gamma=gamma
            self.tree=tree
            self.num_iters=num_iters
            self.num_episodes=num_episodes
            self.model1=MultiOutputRegressor(HoeffdingAdaptiveTreeRegressor())
            self.model2=MultiOutputRegressor(HoeffdingAdaptiveTreeRegressor())
            self.time_start=time.time()
            for iter in range(self.num_iters):
                self.env.replay()
                for episode in range(self.num_episodes):
                    state = self.env.reset()
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    done = False
                    while not done:
                        self.buffer=list()
                        for i in count():
                            action = self.select_action(state)
                            action1=np.array(action)
                            next_state, reward, done, _= self.env.step(action1)
                            reward = torch.tensor([reward], device=device)
                            next_state = np.reshape(next_state, [1, Agent.n_observations])
                            self.memory.push(state, action, next_state, reward, done)
                            state = next_state
                            if done:
                                if (self.env.old_avg_reward < -1500):
                                    return
                                print('Episode: {}, Score: {}'.format(
                                    episode, self.env.old_avg_reward))
                                break        
                            if (i > duration):
                                break
                            
                        
                            # for (a,b,c,d,e) in self.buffer:
                            #     self.memory.push(a,b,c,d,e)
                    if done:
                        if episode==0: 
                            self.optimize_model(upgrade=False)
                        else:  
                            if  done and episode %2==0:
                            # if self.isFit and (not self.is_max_tree_reached1() or not self.is_max_tree_reached2()):
                                self.optimize_model(upgrade=True)
                            else:
                                self.optimize_model(upgrade=False)
                    # if len(self.memory) < 2000:
                    #     self.batch_size=len(self.memory)//3
                    # else:
                    #     self.batch_size=2000
            # self.time_end=time.time()
            # print(f"time:{self.time_end-self.time_start}")
            
    def test(self,num_episodes):
        if (self.env.old_avg_reward < -1500):
            return
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = self.select_action(state)
                next_state, reward, done, _= self.env.step(np.array(action))
                state = next_state

            print('Test Episode: {}, Score: {}'.format(episode, self.env.old_avg_reward))
            file.write('\nTest Episode: {}, Score: {}'.format(episode, self.env.old_avg_reward))
        print(self.num_iters,self.num_episodes,self.tree,self.learning_rate,self.gamma)
        file.write('\niter: {}, episode: {}, tree{},learning_rate:{},gamma:{}'.format(self.num_iters,self.num_episodes,self.tree,self.learning_rate,self.gamma))
    def runAC(self, iters,episode,dur,tree,learning_rate,gamma):
        self.train(num_iters=iters, num_episodes=episode,
            duration=dur,tree= tree,learning_rate=learning_rate,gamma=gamma)
        self.test( num_episodes=31, )
Agent=DQNAgent()
Agent.runAC(iters=7,episode=120,dur=31,tree=20,learning_rate=0.05,gamma=0.99)

    