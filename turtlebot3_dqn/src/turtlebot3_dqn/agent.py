#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import os
import numpy as np
import random
import time
import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.dqn_model import DQN, DuelingDQN
import torch
import torch.nn.functional as F
import torch.optim as optim

# set the random seed:
torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)


class ReinforceAgent():
    def __init__(self, state_size, action_size, stage, method, mode):
        self.stage = stage
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('src/turtlebot3_dqn', 
                                    'save_model/stage_'+self.stage+'_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 10000
        self.update_freq = 1
        self.target_update = 2000
        self.target_update_freq = 5000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.max_epsilon = 1.0   
        self.min_epsilon = 0.1
        self.epsilon_decay_step = (self.max_epsilon - self.min_epsilon)/100000
        self.batch_size = 64
        self.train_start = 5000
        self.memory = deque(maxlen=1000000)
        self.mode = mode

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize Q network and target Q network
        self.set_weight_init = True
        if method == "dqn":
            self.model =  DQN(self.state_size, self.action_size, 
                                set_init=self.set_weight_init).to(self.device)
            self.target_model = DQN(self.state_size, self.action_size, 
                                set_init=self.set_weight_init).to(self.device)
        elif method == "dueling":
            self.model =  DuelingDQN(self.state_size, self.action_size, 
                                set_init=self.set_weight_init).to(self.device)
            self.target_model = DuelingDQN(self.state_size, self.action_size, 
                                set_init=self.set_weight_init).to(self.device)


        # define loss function and optimizer
        self.loss_fcn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate)

        if self.mode == "test":
            self.load_model = True
        
        if self.load_model:
            if self.stage == '1':
                model_name = 'final_dqn_11-24.pth'
            if self.stage == '2':
                model_name = 'final_dqn_11-24.pth'
            if self.stage == '3':
                # model_name = 'final_dueling_11-30_23:25.pth'
                model_name = 'final_dqn_12-06_21:46.pth'
            if self.stage == '4':
                model_name = 'final_dqn_12-02_21:42.pth'
            elif self.stage == '5':
                model_name = 'final_dqn_12-02_21:42.pth'
                self.dirPath = self.dirPath.replace('save_model/stage_5_', 
                                    'save_model/stage_4_')

            print('loading trained model in: '+ self.dirPath + model_name)
            model_params = torch.load(self.dirPath + model_name)
            self.model.load_state_dict(model_params)

        # update the target model
        self.updateTargetModel()
        
        # saving model and data
        self.save_model_freq = 20
        

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def getAction(self, state):
        # process state
        state_t = torch.Tensor(np.array(state)).to(self.device).unsqueeze(0)

        # obtain q_value from q_net
        self.q_value = self.model(state_t)

        if self.mode == "train":
            # use epsilon-greedy if test is false
            if np.random.rand() < self.epsilon:
                action = random.randrange(self.action_size)

            else:
                # find greedy action
                action = int(torch.argmax(self.q_value))
            
        if self.mode == "test":
            action = int(torch.argmax(self.q_value))

        return action


    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replayBuffer(self):
         # sample from the replay buffer
        states, actions, rewards, next_states, dones = \
                    zip(*random.sample(self.memory, self.batch_size))

        return states, actions, rewards, next_states, dones


    def trainModel(self):
        # update epsilon value: causing it to decay
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay_step

        ## sample random minibatch from buffer
        states, actions, rewards, next_states, done = self.replayBuffer()

        ## process parameters
        states_t = torch.Tensor(np.array(states)).to(self.device)
        actions_t = torch.Tensor(actions).to(self.device)
        actions_t = actions_t.type(torch.int64).unsqueeze(-1)
        next_states_t = torch.Tensor(np.array(next_states)).to(self.device)

        ## get max Q value for state->next_state using q_target_net
        max_q_values = torch.max(self.target_model(next_states_t), dim=1)[0]

        ## check if the episode terminates in next step
        td_target = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if done[i]:
                td_target[i] = rewards[i]
            else:
                td_target[i] = rewards[i] + self.discount_factor*max_q_values[i]

        ## convert td_target to tensor
        td_target_t = torch.Tensor(td_target).to(self.device)

        ## get current_q_values
        curr_q_values = self.model(states_t).gather(1, actions_t).squeeze()

        ## calculate the loss 
        self.loss = self.loss_fcn(curr_q_values, td_target_t)

        ## perform backprop and update weights
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

