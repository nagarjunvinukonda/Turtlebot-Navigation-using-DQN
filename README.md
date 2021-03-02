# Turtlebot-Navigation-using-DQN

The aim of this project is to train the turtlebot to reach the goal without hitting the obstcales. This project is to understand DQN and how it is trained ledding turtlebot navigation. 

# Setup to Run:

Set the TURTLEBOT3_MODEL in system environment
* export TURTLEBOT3_MODEL=burger

Launch Gazebo with turtlebot3 in Gazebo
* roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch

Launch the DQN training
* roslaunch turtlebot3_dqn turtlebot3_dqn_torch.launch stage:=3 method:='dueling' mode:='test' move_3:='true'

# References:
* [TurtleBot traning with Machine Learning](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/)
* [Contrsuct TurtleBot Navigation using ML, OpenAI Gym](https://www.theconstructsim.com/machine-learning-openai-gym-ros-development-studio-2/)
* [motion planner using Deep Deterministic Policy Gradient (DDPG) in gazebo](https://github.com/m5823779/MotionPlannerUsingDDPG)
