#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
import numpy as np
import time
import argparse
import sys
import os
from datetime import datetime


"""
data_logger.py

"""

class DataLogger():

    def __init__(self, args):
        # define objects to track
        self.stage = args[1]
        self.method = args[2]

        if self.stage == '1' or self.stage == '2':
            self.object_id = ['turtlebot3_burger']
        if self.stage == '3':
            self.object_id = ['turtlebot3_burger',
                              'obstacle']
        if self.stage == '4':
            self.object_id = ['turtlebot3_burger',
                              'obstacle_1',
                              'obstacle_2']
        if self.stage == '5':
            # self.object_id = ['turtlebot3_burger']
            self.object_id = ['turtlebot3']
            
        # variables
        self.x = [[],[],[]]
        self.y = [[],[],[]]
        self.theta = [[],[],[]]
        self.v = [[],[],[]]  
        self.omega = [[],[],[]] 
        self.min_dist = []

        # define path
        self.directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'+'stage'+self.stage+'/'

    def store_data(self):
        # get one instance of message
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/gazebo/model_states', 
                                ModelStates, timeout=3)
                scan_data = rospy.wait_for_message('scan', LaserScan, timeout=3)
            except:
                pass
        
        for i in range(len(self.object_id)):
            # Find the index of this object_id in the name list:
            idx = data.name.index(self.object_id[i])

            # Retrieve states from data
            self.x[i].append(data.pose[idx].position.x)
            self.y[i].append(data.pose[idx].position.y)
            self.v[i].append(data.twist[idx].linear.x)
            self.omega[i].append(data.twist[idx].angular.z)

        # obtain min obstacle distance
        scan_range = []
        for i in range(len(scan_data.ranges)):
            if scan_data.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan_data.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan_data.ranges[i])

        self.min_dist.append(round(min(scan_range), 2))

    def clear_data(self):
        self.x = [[],[],[]]
        self.y = [[],[],[]]
        self.theta = [[],[],[]]
        self.v = [[],[],[]]  
        self.omega = [[],[],[]] 
        self.min_dist = []


    def save_data(self, trial, done='success'):
        # save state data
        for i in range(len(self.object_id)):
            if self.object_id[i] == 'turtlebot3_burger':
                data = np.array([done, self.x[i], self.y[i], self.v[i], self.omega[i], self.min_dist])
            else:
                data = np.array([self.x[i], self.y[i], self.v[i], self.omega[i]])

            # filename = '/logs/'+self.object_id[i]+'_stage_'+self.stage+'_'+self.method+'_'+str(trial)
            filename = self.object_id[i]+'_stage_'+self.stage+'_'+self.method+'_'+str(trial)

            if os.path.isdir(self.directory):
                np.save(self.directory+filename+".npy", data)
            else:
                os.makedirs(self.directory)
                np.save(self.directory+filename+".npy", data)

        # clear data
        self.clear_data()