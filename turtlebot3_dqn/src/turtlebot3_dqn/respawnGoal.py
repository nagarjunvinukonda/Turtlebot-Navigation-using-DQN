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
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('RoboNav/turtlebot3_dqn/src/turtlebot3_dqn',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        
        # get mode parameter
        mode = rospy.get_param('mode')

        if self.stage != 4 and self.stage != 5:
            if mode == 'train':
                while position_check:
                    goal_x = random.randrange(-12, 13) / 10.0
                    goal_y = random.randrange(-12, 13) / 10.0
                    if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                        position_check = True
                    elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                        position_check = True
                    elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                        position_check = True
                    elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                        position_check = True
                    elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                        position_check = True
                    else:
                        position_check = False

                    if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                        position_check = True

                    self.goal_position.position.x = goal_x
                    self.goal_position.position.y = goal_y
                    
            if mode == 'test':
                trial = rospy.get_param('trial_number')
                goal_count = rospy.get_param('goal_count')

                trial_goals = [[[0.6, 0.0], [0.9, 1.2], [-0.9, 0.0], [-0.1, -0.1], [0.5, 0.1]],
                                [[0.9, 1.2], [-1.1, 0.9], [0.9, 1.2], [1.1, 0.3], [0.1, 1.0]],
                                [[-0.6, -1.2], [1.2, 0.5], [1.2, -0.6], [-0.1, -1.2], [1.2, 1.2]],
                                [[-0.5, -0.1], [-1.2, -1.1], [-0.1, -0.8], [0.8, 1.1], [-0.3, 1.2]],
                                [[1.2, 1.0], [-0.6, 0.0], [1.2, -1.0], [-0.2, -1.2], [0.1, 0.7]],
                                [[-1.1, -0.7], [-0.9, 1.1], [-0.8, 1.2], [-1.2, -0.3], [1.1, -0.1]],
                                [[1.2, 0.7], [-1.2, -0.5], [1.2, -0.8], [-1.2, 0.8], [1.1, 0.7]],
                                [[-1.1, 0.9], [0.0, -1.0], [-1.2, 0.1], [-0.5, -1.2], [0.6, 1.1]],
                                [[-1.1, 0.5], [1.1, 1.0], [1.0, 0.0], [-0.9, 1.2], [0.0, -0.7]],
                                [[-0.1, 1.1], [1.1, 0.3], [-0.8, 1.2], [-1.2, -0.3], [0.0, 0.9]]]

                self.goal_position.position.x = trial_goals[trial][goal_count][0]
                self.goal_position.position.y = trial_goals[trial][goal_count][1]

        elif self.stage == 4:
            if mode == 'train':
                while position_check:
                    goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                    goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                    self.index = random.randrange(0, 13)
                    print(self.index, self.last_index)
                    if self.last_index == self.index:
                        position_check = True
                    else:
                        self.last_index = self.index
                        position_check = False

                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
            
            if mode == 'test':
                trial = rospy.get_param('trial_number')
                goal_count = rospy.get_param('goal_count')

                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.trial_index = [[ 3,  1, 10,  8,  6],
                                    [ 7, 10,  6,  0, 10],
                                    [ 1,  2,  4,  6,  4],
                                    [ 0,  5,  6, 11,  4],
                                    [ 0,  7,  6, 11,  6],
                                    [11,  8,  4,  6, 10],
                                    [ 9,  9,  0,  7,  7],
                                    [ 2,  0,  9,  5,  7],
                                    [ 2,  3,  1,  9,  6],
                                    [ 5,  3,  2, 10,  0]]

                self.goal_position.position.x = goal_x_list[self.trial_index[trial][goal_count]]
                self.goal_position.position.y = goal_y_list[self.trial_index[trial][goal_count]]

        elif self.stage == 5:
            
            if mode == 'test':
                trial = rospy.get_param('trial_number')
                goal_count = rospy.get_param('goal_count')

                # for turtlebot3_house world
                # goal_x_list = [-3.9, -6.3, -6.2, -6.5, -1.2, 2.1, 3.1, 6.7, 4.8]
                # goal_y_list = [4.0, 3.8, 0.3, -3.3, 0.5, 0.4, 2.0, 4.2, 1.6]

                goal_x_list = [-6.2, -6.3, -5.1, -3.9, -0.8, 3.2, 6.3, 5.5]
                goal_y_list = [-2.6, 0.2, 3.2, 0.9, 0.3, 0.7, 0.3, -3.8]

                self.trial_index = [[0, 1, 2, 3, 4, 5, 6, 7],
                                    [0, 1, 2, 3, 4, 5, 6, 7],
                                    [0, 1, 2, 3, 4, 5, 6, 7]]
                

                self.goal_position.position.x = goal_x_list[self.trial_index[trial][goal_count]]
                self.goal_position.position.y = goal_y_list[self.trial_index[trial][goal_count]]


        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
