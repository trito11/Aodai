import numpy as np
np.float = float
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier,iSOUPTreeRegressor
import torch
from matplotlib import pyplot as plt
import copy
from environment import *
from mix_state_env import MixStateEnv
from config import *
import copy
from MyGlobal import MyGlobals
from itertools import count
import random
import math
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
from skmultiflow.meta import MultiOutputLearner
import joblib
import time
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
file = open("myfile.txt", "w")


# Configure the XGBoost model with the custom Huber loss function
def huber_loss(y_true, y_pred, delta):
    residual = torch.from_numpy(y_true).gather(1, Agent.action_batch)-torch.from_numpy(y_pred).gather(1, Agent.action_batch)
    condition = np.abs(residual) <= delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * (np.abs(residual) - 0.5 * delta)
    return np.where(condition, squared_loss, linear_loss)

def huber_loss_gradient(y_true, y_pred, delta):
    residual = torch.from_numpy(y_true).gather(1, Agent.action_batch)-torch.from_numpy(y_pred).gather(1, Agent.action_batch)
    condition = np.abs(residual) <= delta
    gradient = residual
    gradient[~condition] = delta * np.sign(residual[~condition])
    return gradient

def huber_loss_hessian(y_true, y_pred, delta):
    residual = torch.from_numpy(y_true).gather(1, Agent.action_batch)-torch.from_numpy(y_pred).gather(1, Agent.action_batch)
    condition = np.abs(residual) <= delta
    hessian = np.ones_like(residual)
    hessian[~condition] = 0
    return hessian
def custom_obj(y_true, y_pred):
    obj=lambda y_true, y_pred: huber_loss(y_true, y_pred, delta),
    grad=lambda y_true, y_pred: huber_loss_gradient(y_true, y_pred, delta),
    hess=lambda y_true, y_pred: huber_loss_hessian(y_true, y_pred, delta)
delta=1

device = device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self,capacity) -> None:
        self.memory=deque([],maxlen=capacity)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
    def push(self, *args):
        self.memory.append(self.Transition(*args))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)
class DQNAgent:
    def __init__(self) :
        self.optimize=0
        self.max=False
        self.env=MixStateEnv()
        self.env.seed(125)
        self.batch_size=800
        self.eps_start=0.9
        self.eps_end=0.05
        self.eps_decay=1000
        self.tau=0.1
        self.lr=1e-4
        self.n_actions=NUM_ACTION
        self.n_observations=NUM_STATE
        self.isFit=False
        self.memory1=ReplayMemory(10000)
        self.memory2=ReplayMemory(800)
        self.episode_durations = []
        self.stepdone=0
        self.delta = 1  # Threshold parameter
        self.ok=False
# Create the custom loss function object
 
    def select_action(self,state):
            sample=random.random()
            eps_threshold=self.eps_end+(self.eps_start-self.eps_end)*math.exp(-1.*self.stepdone/self.eps_decay)
            self.stepdone+=0.005
            if sample > eps_threshold:
                with torch.no_grad():
                    if self.isFit==True:
                        q_value= self.model1.predict(state.reshape(1,-1))
        
                    else: 
                        q_value=np.zeros(self.n_actions)
                    return torch.tensor(np.argmax(q_value[0])).view(1, 1)
            else:
                return torch.tensor([[random.randint(0,NUM_ACTION-1)]], device=device, dtype=torch.long)
    def optimize_model(self,upgrade):
            ok=self.ok
            if len(self.memory2) < self.batch_size:
                return
            transitions1 = self.memory1.sample(0)
            transitions2 = self.memory2.sample(800)
            batch = Transition(*zip(*transitions1+transitions2))
            assert isinstance(self.model1, MultiOutputLearner)
            assert isinstance(self.model1, MultiOutputLearner)
            
            # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            #                                     batch.next_state)), device=device, dtype=torch.bool)
            # non_final_next_states = torch.cat([s for s in batch.next_state
            #                                             if s is not None])
            batch_state=[]
            for i in batch.state :
                batch_state.append(i.view(1,14))
            batch_next_state=[]
            for i in batch.next_state :
                batch_next_state.append(i.view(1,14))
            next_state_batch = torch.cat(batch_next_state)
            state_batch = torch.cat(batch_state)
            self.action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            index_tensor = torch.arange(self.action_batch.size(0)).reshape(-1, 1)
            if self.isFit:
                state_value1=torch.from_numpy(self.model1.predict(state_batch))
                
            else:
                state_value1=state_value2=torch.zeros((self.batch_size,self.n_actions))
                state_value2=state_value2.float()
            # state_action_values = self.model1(state_batch).gather(1, action_batch)
            # values=torch.zeros(self.batch_size,self.n_actions, device=device)
            # values[index][action_batch]=state_value[index][action_batch]
            state_value1=state_value1.float()
            
            indices=torch.cat((index_tensor,self.action_batch),dim=1)
            if self.isFit:
                # q1=torch.from_numpy(np.array(self.model1.predict(next_state_batch)))
                # q2=torch.from_numpy(np.array(self.model2.predict(next_state_batch)))
               
                # q2=q2.max(1)[1]
                # next_state_values1 =torch.from_numpy(np.array(q1.gather(1,q2.unsqueeze(0) ))).squeeze(0)
                next_state_values1 =torch.from_numpy(np.array(self.model2.predict(next_state_batch).max(1)[0]))
            else:  
                next_state_values1=next_state_values2=reward_batch.float()
            # Compute the expected Q values
            expected_state_action_values1 = (next_state_values1 * self.gamma) + reward_batch
            if not self.isFit:
                expected_state_action_values2 = (next_state_values2 * self.gamma) + reward_batch
                torch.index_put_(state_value2, tuple(indices.t()), expected_state_action_values2.float())
            torch.index_put_(state_value1, tuple(indices.t()), expected_state_action_values1.float())
            
            
            if upgrade:
                joblib.dump(self.model1, 'initial_model.pkl')
                self.loaded_xgb_regressor = joblib.load('initial_model.pkl')
                self.model2=MultiOutputLearner(self.loaded_xgb_regressor)
            else:   
              
            
              # self.model2=MultiOutputRegressor(XGBRegressor(learning_rate=self.learning_rate,n_estimators=200,max_depth=10,n_jobs=1,tree_method='gpu_hist'))
                if self.isFit:
                        self.model1.partial_fit(state_batch, state_value1)
                    
                else:
                    self.model1.fit(state_batch.reshape(1, -1), state_value1.reshape(1, -1))
                    self.model2.fit(state_batch.reshape(1, -1), state_value2.reshape(1, -1))
            self.isFit = True   
            
            # print(f"optimze_time:{self.end_time-self.start_time},calculate:{self.calculate-self.start_time};times:{self.optimize}")
    def train(self,num_iters,num_episodes,duration,tree,learning_rate,gamma):
            self.learning_rate=learning_rate
            self.gamma=gamma
            self.tree=tree
            self.num_iters=num_iters
            self.num_episodes=num_episodes
            # Create an instance of the ExtremelyFastDecisionTreeClassifier
            base_regressor = ExtremelyFastDecisionTreeClassifier()

        # Wrap the regressor in MultiOutputRegressor
            self.model1 = MultiOutputLearner(base_regressor)
            self.model2 = MultiOutputLearner(base_regressor)
            self.time_start=time.time()
            for iter in range(self.num_iters):
                self.env.replay()
                for episode in range(self.num_episodes):
                    state = self.env.reset()
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    done=False
                    while not done:
                        for i in count():
                            state = torch.FloatTensor(state).to(device)
                            action = self.select_action(state)
                            action1=action.item()
                            next_state, reward, done, _= self.env.step(np.array(action1))
                            reward = torch.tensor([reward], device=device)
                            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                            self.memory1.push(state, action, next_state, reward)
                            self.memory2.push(state, action, next_state, reward)
                            state = next_state
                            if done:
                                if (self.env.old_avg_reward < -1500):
                                    return
                                print('Episode: {}, Score: {}'.format(
                                    episode, self.env.old_avg_reward))
                                break
                            if (i > duration):
                                break
                   
                    if done:
                        if episode==0: 
                                self.optimize_model(upgrade=False)
                        else:  
                            if  done and episode % 20==0:
                            # if self.isFit and (not self.is_max_tree_reached1() or not self.is_max_tree_reached2()):
                                self.optimize_model(upgrade=True)
                            else:
                                self.optimize_model(upgrade=False)
                            
                    # if done:
                    #     if episode%: 
                    #         self.optimize_model(upgrade=True,reset=False)
                    #     else:  
                    #         self.optimize_model(upgrade=False,reset=False)
                   
                    
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
                action1=action.item()
                next_state, reward, done, _= self.env.step(np.array(action1))
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
Agent.runAC(iters=7,episode=120,dur=100,tree=10,learning_rate=0.03 ,gamma=0.99)

    