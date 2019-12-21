# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    return start[0]+length*math.cos(math.radians(angle)), start[1]-length*math.sin(math.radians(angle))

def get_distance(p10, p11, p20, p21):
    return math.sqrt((p10-p20)**2 + (p11-p21)**2)

def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """    
    for line in armPos:
        A = line[1][1]-line[0][1]   #(y2-y1)
        B = -line[1][0]+line[0][0]   #-(x2-x1)
        C = -(line[0][0]*(line[1][1]-line[0][1]) - line[0][1]*(line[1][0]-line[0][0])) #-(y1/(y2-y1) - x1/(x2-x1)) 
        for obstacle in obstacles:
            distance = 1.0*abs(A*obstacle[0]+B*obstacle[1]+C)/math.sqrt(A**2+B**2)
            if(distance <= obstacle[2]):
                point_x = (B**2*obstacle[0]-A*B*obstacle[1]-A*C)/(A**2+B**2)
                point_y = (A**2*obstacle[1]-A*B*obstacle[0]-B*C)/(A**2+B**2)
                if(min(line[0][0], line[1][0]) <= point_x and point_x <= max(line[0][0], line[1][0]) and min(line[0][1], line[1][1]) <= point_y and point_y <= max(line[0][1], line[1][1])):
                    return True
                if(get_distance(obstacle[0],obstacle[1],line[0][0],line[0][1])<=obstacle[2] or get_distance(obstacle[0],obstacle[1],line[1][0],line[1][1])<=obstacle[2]):
                    return True
    return False



def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    for goal in goals:
        distance = math.sqrt((armEnd[0] - goal[0])**2+(armEnd[1] - goal[1])**2)
        if(distance <= goal[2]):
            return True
    return False

def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """
    for point in armPos:
        if (point[0][0] < 0 or point[0][0] > window[0] or point[0][1] < 0 or point[0][1] > window[1] or point[1][0] < 0 or point[1][0] > window[0] or point[1][1] < 0 or point[1][1] > window[1]):
            return False
    return True
