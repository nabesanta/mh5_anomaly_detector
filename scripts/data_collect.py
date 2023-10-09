#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import csv
import sys

POSITIONS_LIST = []
LABEL = None
PREV_IMU = None
PREV_ODOM = None

def control_command_imu(val):
    global POSITIONS_LIST
    global LABEL
    global PREV_IMU

    # Add your condition here based on the received IMU data
    # For example, if you want to check if linear acceleration is non-zero:
    if val.linear_acceleration.x != 0.0 or val.linear_acceleration.y != 0.0 or val.linear_acceleration.z != 0.0:
        if LABEL:
            if PREV_IMU != [val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z]:
                POSITIONS_LIST.append([val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z, LABEL])
                PREV_IMU = [val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z]
        else:
            if PREV_IMU != [val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z]:
                POSITIONS_LIST.append([val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z])
                PREV_IMU = [val.linear_acceleration.x, val.linear_acceleration.y, val.linear_acceleration.z]

def control_command_odom(val):
    global POSITIONS_LIST
    global LABEL
    global PREV_ODOM

    # Add your condition here based on the received Odom data
    # For example, if you want to check if position has changed:
    if val.pose.pose.position.x != 0.0 or val.pose.pose.position.y != 0.0 or val.pose.pose.position.z != 0.0:
        if LABEL:
            if PREV_ODOM != [val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z]:
                POSITIONS_LIST.append([val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z, LABEL])
                PREV_ODOM = [val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z]
        else:
            if PREV_ODOM != [val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z]:
                POSITIONS_LIST.append([val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z])
                PREV_ODOM = [val.pose.pose.position.x, val.pose.pose.position.y, val.pose.pose.position.z]

def collect(out):
    while POSITIONS_LIST:
        out.writerow(POSITIONS_LIST[0])
        POSITIONS_LIST.pop(0)

if __name__ == '__main__':
    print('collect - ', sys.argv)

    rospy.init_node('data_collector', anonymous=True)
    rospy.Subscriber('/imu', Imu, control_command_imu)
    rospy.Subscriber('/odom', Odometry, control_command_odom)

    cwd = sys.argv[1]

    if len(sys.argv) > 2:
        LABEL = sys.argv[2]

    sleep(3)

    with open(cwd, "w+") as out_file:
        out = csv.writer(out_file, delimiter=',')
        print('Writing to', cwd)
        while not rospy.is_shutdown():
            collect(out)
