# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze
import numpy as np
from util import *

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # TODO: Write your code here 
    record_path = [[(-1,-1) for i in range(len(maze.get_map()))] for j in range(len((maze.get_map())[0]))]
    record_visit = [[0 for i in range(len(maze.get_map()))] for j in range(len((maze.get_map())[0]))]
    begin = maze.getStart()
    queue = [begin]
    curr = queue[0]
    goals = maze.getObjectives()
    while(queue != [] and curr not in goals):
        temp = maze.getNeighbors(curr[0],curr[1])
        queue.pop(0)
        transfer_curr = angleToIdx((curr[0], curr[1]), maze.offsets, maze.granularity)
        #print(transfer_curr[0],transfer_curr[1], curr)
        #print(record_visit)
        record_visit[transfer_curr[1]][transfer_curr[0]] = 1
        for spot in temp:
            transfer_spot = angleToIdx((spot[0], spot[1]), maze.offsets, maze.granularity)
            if(record_visit[transfer_spot[1]][transfer_spot[0]] != 1):
                record_visit[transfer_spot[1]][transfer_spot[0]] = 1
                queue.append(spot)
                record_path[transfer_spot[1]][transfer_spot[0]] = (transfer_curr[1],transfer_curr[0])
        if (queue != []):
            curr = queue[0]
    #generate path---------------------------------------------------
    result = []
    if(curr not in goals and queue == []):
        return [], 0
    idx_begin = (angleToIdx((begin[0], begin[1]), maze.offsets, maze.granularity)[1], angleToIdx((begin[0], begin[1]), maze.offsets, maze.granularity)[0])
    transfer_curr = (angleToIdx((curr[0], curr[1]), maze.offsets, maze.granularity)[1], angleToIdx((curr[0], curr[1]), maze.offsets, maze.granularity)[0])
    while(transfer_curr != idx_begin):
        result.append(transfer_curr)
        transfer_curr = record_path[transfer_curr[0]][transfer_curr[1]]
    result.append(transfer_curr)
    for i in range(len(result)):
        result[i] = idxToAngle(((result[i])[1], (result[i])[0]), maze.offsets, maze.granularity)
    print(result)
    return result[::-1], len(result)

def dfs(maze):
    # TODO: Write your code here    
    return [], 0

def greedy(maze):
    # TODO: Write your code here    
    return [], 0

def astar(maze):
    # TODO: Write your code here    
    return [], 0