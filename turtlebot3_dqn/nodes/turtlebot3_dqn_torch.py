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
import json
import numpy as np
import random
import time
import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.dqn_model import DQN
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.turtlebot3_dqn.agent import ReinforceAgent
from src.turtlebot3_dqn.data_logger import DataLogger
from src.turtlebot3_dqn.environment import Env

# set the random seed:
torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

# initialize tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
tensorboard = SummaryWriter(log_dir=log_dir)
    

# train function
def train(args):
    # parse the arguments:
    stage = args[1]
    method = args[2]
    mode = args[3]
    # set the mode ros_parameter:
    rospy.set_param('mode', mode)
    # initialize node
    rospy.init_node('turtlebot3_dqn_stage_'+str(stage))
    # initialize result publisher
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    # initialize get_action publisher
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    # set varaibles
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # set state and action space sizes
    if stage == "1":
        state_size = 26
        action_size = 5
    else:
        state_size = 28
        action_size = 5

    # initialize environment
    env = Env(action_size)

    # initialize agent
    agent = ReinforceAgent(state_size, action_size, stage, method, mode)

    # set variables
    scores, episodes, total_reward = [], [], []
    global_step = 0
    # set start time
    start_time = time.time()

    # set file storage parameters
    st_time = datetime.datetime.now().strftime("%d-%m-[%H:%M]")
    save_count = 0
    logs_path = 'logs/'
    logs_dir = logs_path + method + '_model_' + st_time 
    os.makedirs(logs_dir)

    EPISODES = 3000

    # main loop: for each episode
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        
        # inner loop: for each episode step
        for t in range(agent.episode_step):
            # get action
            action = agent.getAction(state)

            # take action and return state, reward, status
            next_state, reward, done = env.step(action)

            # append memory to memory buffer
            agent.appendMemory(state, action, reward, next_state, done)

            ## check if replay buffer is ready:
            if len(agent.memory) >= agent.train_start:
                # learn only after every 8 steps:
                if global_step % agent.update_freq == 0:
                    agent.trainModel()

            ## update the target network parameters
            if global_step % agent.target_update_freq == 0:
                # update the target network
                agent.updateTargetModel()

            # increment score and append reward
            score += reward
            total_reward.append(reward)

            # update state
            state = next_state

            # publish get_action
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # save to tensorboard
            num = 30
            tensorboard.add_scalar('step reward', reward, global_step)
            tensorboard.add_scalar('average step reward (over 30 steps)', 
                                    sum(total_reward[-num:])/num, global_step)

            # save model after every N episodes
            if e % agent.save_model_freq == 0 and e != 0:
                torch.save(agent.model.state_dict(), agent.dirPath + str(e) + '.pth')

            # timeout after 1200 steps (robot is just moving in circles or so)
            if t >= 1200: # changed this from 500 to 1200 steps
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, int(torch.argmax(agent.q_value))]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d | score: %.2f | memory: %d | epsilon: %.6f | time: %d:%02d:%02d',
                            e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))

                # add to tensorboard
                k = 10
                tensorboard.add_scalar('episode reward', score, e)
                tensorboard.add_scalar('average episode reward (over 10 episodes)', 
                                    sum(scores[-k:])/k, e)

                # save reward to file
                reward_logs_filename = logs_dir + '/' + method + '-rewards.npy'

                np.save(reward_logs_filename, np.array(total_reward, dtype=np.float))
                np.save(reward_logs_filename, np.array(scores, dtype=np.float))

                break

            global_step += 1


# test function
def test(args):
    # parse the arguments:
    stage = args[1]
    method = args[2]
    mode = args[3]
    # set the mode ros_parameter:
    rospy.set_param('mode', mode)
    # initialize node
    rospy.init_node('turtlebot3_dqn_stage_'+str(stage))
    # initialize result publisher
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    # initialize get_action publisher
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    # set varaibles
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    # initialize data logger
    logger = DataLogger(args)

    # set state and action space sizes
    if stage == '1':
        state_size = 26
        action_size = 5
    else:
        state_size = 28
        action_size = 5

    # initialize environment
    env = Env(action_size)

    rospy.loginfo("state_size is: "+str(state_size))

    # initialize agent
    agent = ReinforceAgent(state_size, action_size, stage, method, mode)

    # set variables
    scores, episodes, total_reward = [], [], []
    global_step = 0
    # set start time
    start_time = time.time()

    TRIALS = 30

    # set the trial length
    if stage == '5':
        TRIAL_LENGTH = 8
    else:
        TRIAL_LENGTH = 5

    # main loop: for each episode
    for e in range(agent.load_episode + 1, TRIALS):        
        # set the trial and goal_count as a ros_parameter
        rospy.set_param('trial_number', e-1)
        goal_count = 0
        rospy.set_param('goal_count', goal_count)

        done = False
        state = env.reset()
        score = 0
        
        # inner loop: for each episode step
        # for t in range(agent.episode_step):
        episode_done = False
        while not episode_done:
            # get action
            action = agent.getAction(state)

            # take action and return state, reward, status
            next_state, reward, done = env.step(action)

            # increment score and append reward
            score += reward
            total_reward.append(reward)

            # update state
            state = next_state

            # publish get_action
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # store states
            logger.store_data()

            # check if goal is reached
            if env.goal_reached:
                #goal_count += 1
                rospy.set_param('goal_count', goal_count)
                rospy.loginfo('Trial: %d | Goal [%d] completed', e, goal_count)
                env.goal_reached = False
                if goal_count == TRIAL_LENGTH:
                    episode_done = True
                    logger.save_data(e)

            if done:
                # if episode/trial fails (i.e. collides)
                logger.save_data(e, done='fail')
                result.data = [score, int(torch.argmax(agent.q_value))]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Failed Trial  |  Starting Trial %d ', e)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))

                break

            global_step += 1



if __name__ == '__main__':

    # get argument passed to node
    args = rospy.myargv(argv=sys.argv)
    
    # get mode argument:
    mode = args[3]

    # run the appropriate mode
    if mode == "train":
        train(args)
    if mode == "test":
        test(args)
    





