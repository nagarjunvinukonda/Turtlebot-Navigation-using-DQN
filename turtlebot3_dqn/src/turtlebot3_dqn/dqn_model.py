#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    This is just a hint. You can build your own structure.
    """

    def __init__(self, state_size=4, action_size=4, set_init=False):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, 
                which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        
        self.in_channels = state_size
        self.action_size = action_size

        # define the network
        self.fc = nn.Sequential(
                  nn.Linear(self.in_channels, 512),
                  nn.ReLU(),
                  nn.Linear(512, self.action_size),
        )

        # initialize the weights using He initialization
        if set_init:
            relu_gain = nn.init.calculate_gain("relu")
            self.fc[0].weight.data.mul_(relu_gain)
            self.fc[2].weight.data.mul_(relu_gain)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################

        # forward through fc layers
        output = self.fc(x).squeeze()

        ###########################
        return output


class DuelingDQN(nn.Module):
    """
    Initialize a dueling deep Q-learning network

    """

    def __init__(self, state_size=4, action_size=4, set_init=False):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, 
                which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DuelingDQN, self).__init__()
        ###########################

        self.in_channels = state_size
        self.action_size = action_size

        # define the network
        self.fc = nn.Sequential(
                  nn.Linear(self.in_channels, 512),
                  nn.ReLU()
        )

        # state-value
        self.fc_value = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

        # advantage
        self.fc_adv = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)
        )

        # initialize the weights using He initialization
        if set_init:
            relu_gain = nn.init.calculate_gain("relu")
            self.fc[0].weight.data.mul_(relu_gain)

            self.fc_value[0].weight.data.mul_(relu_gain)
            self.fc_value[2].weight.data.mul_(relu_gain)

            self.fc_adv[0].weight.data.mul_(relu_gain)
            self.fc_adv[2].weight.data.mul_(relu_gain)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################

        # forward through conv layers
        output = self.fc(x)

        # forward through state-value and advantage layers
        V = self.fc_value(output)
        A = self.fc_adv(output)

        q_values = V + (A - torch.mean(A, 1).unsqueeze(-1))

        ###########################
        return q_values