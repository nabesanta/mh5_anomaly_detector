#!/usr/bin/env python

import sys
import rospy

from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import subprocess
import os

SCALER = None
REDUCE = None
MODEL = None
THRESHOLD = None

PREV = None
GOOD = 0
BAD = 0

POINTS = []
PREDS = []
POSITIONS_LIST = []


def load_modules(load_file, th):
    '''
        This function NEEDS to get called first, so that global variables can
            be set.
        Once set, they can be used to classify incoming data.
        Sets global variables, so they can be used later.
    '''

    global SCALER
    global REDUCE
    global MODEL
    global THRESHOLD

    THRESHOLD = th

    # FIXME: TODO: Fix relative paths
    SCALER = joblib.load("../data/scale.pkl")
    REDUCE = joblib.load("../data/reduce.pkl")

    load_file = '../data/'+load_file
    MODEL = joblib.load(load_file)

    print('Modules loaded. Over.')


def command(val):
    '''
        Callback function for when specified message is published.
        Adds data to global list to be processed later.
    '''
    global PREV
    poses = list(val.position)

    if PREV != poses:
        POSITIONS_LIST.append(poses)
        PREV = poses


def check():

    global SCALER
    global REDUCE
    global MODEL
    global THRESHOLD

    global GOOD
    global BAD

    global POINTS
    global PREDS

    while POSITIONS_LIST:

        data = np.array([POSITIONS_LIST[0]])
        POSITIONS_LIST.pop(0)

        scaled = SCALER.transform(data)

        pca_ed = REDUCE.transform(scaled)
        POINTS.append(pca_ed.tolist()[0])

        prediction = MODEL.predict(pca_ed)
        PREDS.append(prediction[0])

        if prediction == 1:
            GOOD += 1
            print('Good: ', GOOD/float(GOOD+BAD), 'Bad: ', BAD/float(GOOD+BAD), 'Count: ', GOOD+BAD)
        else:
            BAD += 1
            print('Good: ', GOOD/float(GOOD+BAD), 'Bad: ', BAD/float(GOOD+BAD), 'Count: ', GOOD+BAD)
            rospy.logwarn("WARNING: Robot state classified as an anomaly.\n" + str(data))

        if (BAD/float(BAD+GOOD) > float(THRESHOLD)) and (BAD+GOOD > 60):
            rospy.logerr("ERR: Robot state classified as an anomaly.\n" + str(data))


def graph_3d():

    global POINTS
    global PREDS
    global GOOD
    global BAD

    percentgood = GOOD/float(GOOD+BAD)
    percentbad = BAD/float(GOOD+BAD)
    print(GOOD, BAD, percentgood, percentbad)

    pts0 = [row[0] for row in POINTS]
    pts1 = [row[1] for row in POINTS]
    pts2 = [row[2] for row in POINTS]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(pts0, pts1, pts2, c=PREDS, cmap='bwr', s=10)

    plt.show()


if __name__ == '__main__':

    rospy.init_node('monitor', anonymous=True)

    load_modules(sys.argv[2], sys.argv[3])

    # Subscribe to the appropriate topic
    rospy.Subscriber(sys.argv[1],
                    JointState,
                    command)

    rospy.on_shutdown(graph_3d)

    while not rospy.is_shutdown():
        check()
