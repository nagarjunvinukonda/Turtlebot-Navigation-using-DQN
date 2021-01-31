#!usr/bin/env python
import numpy as np
import math
import pickle
import csv
import statistics
import os


def calculate_dist(x2,y2,x1,y1):
    return math.sqrt ((x2-x1)**2 + (y2-y1)**2)

def calculate_path_length(x_list, y_list):
    path_length = 0.0
    if len(x_list) != len(y_list):
        print("Different Sizes of arrays")
        return
    for i in range (1,len(x_list)):
        path_length += calculate_dist(x_list[i],y_list[i],x_list[i-1],y_list[i-1])
    return path_length

def calculate_average_clearance(dist_list):
    return np.mean(dist_list)

def calculate_computation_time(x_list): # figure out convertion from size to seconds
    return len(x_list)

def calculate_smoothing(x_list, y_list):
    slope_list = []
    for i in range (1,len(x_list)):
        if x_list[i] - x_list[i-1] == 0.0:
            slope_list.append(math.inf)
        else:
            slope_list.append((y_list[i] - y_list[i-1])/(x_list[i] - x_list[i-1]))
    return statistics.stdev(slope_list)


# # define path
# directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'+'stage'+stage+'/'
log_directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'


for i in range(1,5): # iterating through stage folders

    with open(log_directory+"stage_"+str(i)+'_summary.csv', mode='a') as output:
        dw = csv.DictWriter(output, delimiter='\t', fieldnames=['Success','Path Length (in meters)', 'Average Clearance (in meters)', 'Computation_time', 'Smoothness Factor'])
        dw.writeheader()

        for trial_no in range(1,11): #iterating through trail numbers
            a = np.load(log_directory+"stage"+str(i) + "/turtlebot3_burger" + "_stage_" + str(i) + "_dqn_" + str(trial_no) + ".npy", allow_pickle=True)
            if a[0] == 'success':
                success = 1
            else:
                success = 0

            x_data = a[1]
            y_data = a[2]
            clearance_data = a[5]

            path_length = calculate_path_length(x_data,y_data)
            avg_clearance = calculate_average_clearance(clearance_data)
            computation_time = calculate_computation_time(x_data)
            smoothness = calculate_smoothing(x_data,y_data)
            # with open('output.csv', mode='a') as output:
            
            csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ success, path_length,avg_clearance,computation_time, smoothness])