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
import time
from gazebo_msgs.msg import ModelState, ModelStates

class MoveObstacle():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        state, state2 = 0, 0
        
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle_1':
                    obstacle_1 = ModelState()
                    obstacle_1.model_name = model.name[i]
                    obstacle_1.pose = model.pose[i]
                    if abs(obstacle_1.pose.position.x + 1.3) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                        state = 0

                    if state == 0:
                        obstacle_1.pose.position.x -= 0.015
                        obstacle_1.pose.position.y -= 0.00
                        if abs(obstacle_1.pose.position.x + 2.74) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                            state = 1

                    elif state == 1:
                        obstacle_1.pose.position.x -= 0.015
                        obstacle_1.pose.position.y += 0.000
                        if abs(obstacle_1.pose.position.x + 4.80) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                            state = 2

                    elif state == 2:
                        obstacle_1.pose.position.x -= 0.015
                        obstacle_1.pose.position.y -= 0.000
                        if abs(obstacle_1.pose.position.x + 6.3) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                            state = 3

                    elif state == 3:
                        obstacle_1.pose.position.x += 0.015
                        obstacle_1.pose.position.y += 0.00
                        if abs(obstacle_1.pose.position.x + 5.95) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                            state = 4

                    elif state == 4:
                        obstacle_1.pose.position.x += 0.015
                        obstacle_1.pose.position.y += 0.000
                        # if abs(obstacle_1.pose.position.x + 1.3) < 0.05 and abs(obstacle_1.pose.position.y - 3.2) < 0.05:
                        #     state = 5

                    self.pub_model.publish(obstacle_1)
                    time.sleep(0.1)

                if model.name[i] == 'obstacle_2':
                    obstacle_2 = ModelState()
                    obstacle_2.model_name = model.name[i]
                    obstacle_2.pose = model.pose[i]
                    if abs(obstacle_2.pose.position.x - 6.8) < 0.05 and abs(obstacle_2.pose.position.y - 0.5) < 0.05:
                        state2 = 0

                    if state2 == 0:
                        obstacle_2.pose.position.x -= 0.015
                        obstacle_2.pose.position.y -= 0.00
                        if abs(obstacle_2.pose.position.x - 4.25) < 0.05 and abs(obstacle_2.pose.position.y - 0.5) < 0.05:
                            state2 = 1

                    elif state2 == 1:
                        obstacle_2.pose.position.x -= 0.015
                        obstacle_2.pose.position.y += 0.000
                        if abs(obstacle_2.pose.position.x + 4.1) < 0.05 and abs(obstacle_2.pose.position.y - 0.5) < 0.05:
                            state2 = 2

                    elif state2 == 2:
                        obstacle_2.pose.position.x += 0.015
                        obstacle_2.pose.position.y -= 0.000
                    #     if abs(obstacle_2.pose.position.x - 6.8) < 0.05 and abs(obstacle_2.pose.position.y - 0.5) < 0.05:
                    #         state2 = 3

                    # elif state2 == 3:
                    #     obstacle_2.pose.position.x += 0.015
                    #     obstacle_2.pose.position.y += 0.000

                    self.pub_model.publish(obstacle_2)
                    time.sleep(0.1)

def main():
    rospy.init_node('House_obstacles_node')
    try:
        obstacle_1 = MoveObstacle()
        obstacle_2 = MoveObstacle()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()