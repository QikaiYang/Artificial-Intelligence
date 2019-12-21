
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.
    """
    limits_arm = arm.getArmLimit()
    rows = int(((limits_arm[0][1] - limits_arm[0][0])*1.0/granularity) + 1)
    cols = int(((limits_arm[1][1] - limits_arm[1][0])*1.0/granularity) + 1)
    start = arm.getArmAngle()
    goal_ob = obstacles.copy() + goals.copy()
    start_ = (int((start[0]-limits_arm[0][0])/granularity), int((start[1]-limits_arm[1][0])/granularity))
    final_maze = [[" " for i in range(cols)] for j in range(rows)]
    for row in range(len(final_maze)):
        for col in range(len(final_maze[0])):
            angles = idxToAngle((row,col), (limits_arm[0][0], limits_arm[1][0]), granularity)
            arm.setArmAngle(angles)
            position = arm.getArmPos()
            if(doesArmTouchGoals(arm.getEnd(), goals)):
                final_maze[row][col] = OBJECTIVE_CHAR 
            elif(doesArmTouchObstacles(position, goal_ob) == True):
                final_maze[row][col] = WALL_CHAR 
            elif(isArmWithinWindow(position, window) == False):
                final_maze[row][col] = WALL_CHAR
    print(np.shape(final_maze))
    final_maze[start_[0]][start_[1]] = START_CHAR
    return Maze(final_maze, [limits_arm[0][0], limits_arm[1][0]], granularity)