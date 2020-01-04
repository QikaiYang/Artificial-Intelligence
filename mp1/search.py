# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

#def reorder(queue,destination):
import heapq as hq
from itertools import permutations

def fun(point, dest):
    return (abs(point[0]-dest[0])+abs(point[1]-dest[1]))

def check_complete(record_path,objects): #objects are the unvisited points and we should +1 because the path include the start 
    length = len(objects)
    for i in range(len(record_path)):
        if(len(record_path[i]) == length):
            return 1
    return 0

def get_tree_score(matrix, unvisited_objects, current):      #(distance_matrix, list of coordinates, 0)
    result = 0
    unvisited = unvisited_objects.copy()
    visited = [unvisited.pop(0)]
    while(unvisited != []):
        temp_len = 10000000000000000000
        temp_record = -1
        for i in range(len(visited)):
            for j in range(len(unvisited)):
                if(matrix[visited[i]][unvisited[j]] <= temp_len):
                    temp_len = matrix[visited[i]][unvisited[j]]
                    temp_record = j
        result += temp_len
        visited.append(unvisited.pop(temp_record))
    #------------------------add the distance from current point to the nearest point-------------------------
    temp_len = 10000000000000000000
    temp_record = -1
    for i in range(len(visited)):
        if(matrix[current][visited[i]] <= temp_len):
            temp_len = matrix[current][visited[i]]
            temp_record = visited[i]
    result += temp_len ###
    return result, temp_record      #which point to visit next

def get_unvisit(objects, visited):                 #get the list!!!!
    result = list(set(objects) - set(visited))
    return result

def get_mini_point(matrix, objects, visited, cost):#distance matrix, all dots, a row of visited dots, a row of cost array
    unvisited = get_unvisit(objects, visited)
    f = 100000000000000
    record = -1
    for i in range(len(visited)):
        store_score, store_index = get_tree_score(matrix, unvisited, visited[i])
        if(cost[i] + store_score <= f):
            f = cost[i] + store_score
            record = i
    return f, visited[record], store_index            #record is current point, store_index is the next point to visit

def astar_help(maze,begin,end):
    result = []
    record_path = [[(-1,-1) for i in range(maze.cols)] for j in range(maze.rows)] #store path
    visited = [[0 for i in range(maze.cols)] for j in range(maze.rows)]           #store visited
    dic = {begin:0}     #relace the function of priority queue : key is the coordinate(x,y) and value is the "f"
    dis = {begin:0}     #store distance
    curr = min(dic, key=dic.get)
    visited[begin[0]][begin[1]] = 1
    while(curr != end and len(dic) != 0):
        dic.pop(min(dic, key=dic.get))
        visited[curr[0]][curr[1]] = 1
        temp = maze.getNeighbors(curr[0],curr[1])
        for spot in temp:
            if(visited[spot[0]][spot[1]] != 1):
                if(dic.__contains__(spot)==True):
                    if(dic[spot] >= dis[curr] + 1 + fun(spot, end)):
                        dic[spot] = dis[curr] + 1 + fun(spot, end)
                        dis[spot] = dis[curr] + 1
                        record_path[spot[0]][spot[1]] = curr
                else:
                    dic[spot] = dis[curr] + 1 + fun(spot, end)
                    dis[spot] = dis[curr] + 1
                    record_path[spot[0]][spot[1]] = curr
        if(len(dic) != 0):
            curr = min(dic, key=dic.get)
    if(len(dic)==0):
        return []
    while(curr != begin):
        result.append(curr)
        curr = record_path[curr[0]][curr[1]]
    result.append(curr)
    return result[::-1]

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "astar": astar,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    record_path = [[(-1,-1) for i in range(maze.cols)] for j in range(maze.rows)]
    record_visit = [[0 for i in range(maze.cols)] for j in range(maze.rows)]
    begin = maze.getStart()
    end = maze.getObjectives()[0]
    queue = [begin]
    curr = queue[0]
    while(queue != [] and curr != end):
        temp = maze.getNeighbors(curr[0],curr[1])
        queue.pop(0)
        record_visit[curr[0]][curr[1]] = 1
        for spot in temp:
            if(record_visit[spot[0]][spot[1]] != 1):
                record_visit[spot[0]][spot[1]] = 1
                queue.append(spot)
                record_path[spot[0]][spot[1]] = (curr[0],curr[1])
        if (queue != []):
            curr = queue[0]
    #generate path---------------------------------------------------
    result = []
    if(curr != end and queue == []):
        return []
    while(curr != begin):
        result.append(curr)
        curr = record_path[curr[0]][curr[1]]
    result.append(curr)
    return result[::-1]


def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    record_path = [[(-1,-1) for i in range(maze.cols)] for j in range(maze.rows)]
    record_visit = [[0 for i in range(maze.cols)] for j in range(maze.rows)]
    stack = [maze.getStart()]
    curr = stack[len(stack)-1]
    while(stack != [] and curr != maze.getObjectives()[0]):
        temp = maze.getNeighbors(curr[0],curr[1])
        stack.pop()
        record_visit[curr[0]][curr[1]] = 1
        for spot in temp:
            if(record_visit[spot[0]][spot[1]] != 1):
                stack.append(spot)
                record_path[spot[0]][spot[1]] = (curr[0],curr[1])
        if (stack != []):
            curr = stack[len(stack)-1]
    #generate path---------------------------------------------------
    result = []
    if(curr != maze.getObjectives()[0] and stack == []):
        return []
    while(curr != maze.getStart()):
        result.append(curr)
        curr = record_path[curr[0]][curr[1]]
    result.append(curr)
    return result[::-1]


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_help(maze,maze.getStart(),maze.getObjectives()[0])

def astar_multi(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    #------------------get matrix first---------------------------------------
    start = maze.getStart()
    destinations = maze.getObjectives()
    all = destinations.copy()
    all.insert(0,start)
    distance_matrix = [[0 for i in range(len(all))]for j in range(len(all))]
    path_matrix = [[[] for i in range(len(all))]for j in range(len(all))]
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if(i!=j):
                path_matrix[i][j] = astar_help(maze, all[i], all[j])
                path_matrix[j][i] = (path_matrix[i][j])[::-1]
                distance_matrix[i][j] = len(path_matrix[i][j])
                distance_matrix[j][i] = distance_matrix[i][j]
            else:
                distance_matrix[i][j] = -1
                distance_matrix[j][i] = -1
    print(distance_matrix)
    #print(distance_matrix)
    #print(path_matrix)
    #-------------------use A* to solve "TSP"---------------------------------
    #distance_matrix  -- follow the order of "all"
    #all  --  all dots
    #matrix = [["A", "B", "C"], \
    #          ["A", "E"], \
    #          ["A", "F", "D", "E"]]
    '''result = []
    #input1 distance_matrix
    #input2 path_matrix
    coo_dic = dict(zip(all, list(range(len(all))))) #used for coordinates transfer
    restrict = [[0] for i in range(len(all))]
    record_path=[[0]]               #record path
    cost = [[0]]                    #cost
    all_ = list(range(len(all)))
    while(check_complete(record_path, all_) != 1):
        print(record_path)
        print(all_)
        which_row = -1
        which_item = -1
        next_item = -1
        least_f = 10000000000
        for i in range(len(record_path)): # check each row
            f, curr, next = get_mini_point(distance_matrix, all_, record_path[i], cost[i])
            if(f <= least_f):
                least_f = f
                which_row = i
                which_item = curr
                next_item = next
        print(record_path[which_row].index(which_item))
        if(record_path[which_row].index(which_item) == len(record_path[which_row]) - 1):
            record_path[which_row].append(next_item)
            cost[which_row].append(cost[which_row][record_path[which_row].index(which_item)] + distance_matrix[which_item][next_item])
            
        else:
            insert_array = []
            insert_cost = []
            for j in range(record_path[which_row].index(which_item)+1):
                insert_array.append(record_path[which_row][j])
                insert_cost.append(cost[which_row][j])
            insert_array.append(next_item)
            insert_cost.append(insert_cost[len(insert_cost)-1] + distance_matrix[insert_array[len(insert_array)-2]][next_item])
            record_path.append(insert_array)
            cost.append(insert_cost)
    print("flag!!!!!!!!!!")'''
    result = []
    order = [0]
    unvisited = list(range(1,len(all)))
    while(unvisited != []):
        temp_len = 1000000000000000000
        temp_item = -1
        for item in unvisited:
            if(distance_matrix[order[len(order)-1]][item] <= temp_len):
                temp_len = distance_matrix[order[len(order)-1]][item]
                temp_item = item
        order.append(unvisited.pop(unvisited.index(temp_item)))
    result = []
    print(order)
    for i in range(len(order)-1):
        for item in path_matrix[order[i]][order[i+1]]:
            result.append(item)
    print(result)
    return result

#-----------------------------------------------------------------------------------------------------------

def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    destinations = maze.getObjectives()
    all = destinations.copy()
    all.insert(0, start)
    distance_matrix = [[0 for i in range(len(all))]for j in range(len(all))]
    path_matrix = [[[] for i in range(len(all))]for j in range(len(all))]
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if(i!=j):
                path_matrix[i][j] = astar_help(maze, all[i], all[j])
                path_matrix[j][i] = (path_matrix[i][j])[::-1]
                distance_matrix[i][j] = len(path_matrix[i][j])
                distance_matrix[j][i] = distance_matrix[i][j]
            else:
                distance_matrix[i][j] = -1
                distance_matrix[j][i] = -1
    print(distance_matrix)
    #-------------------------------------------------------------------------
    result = []
    order = [0]
    unvisited = list(range(1,len(all)))
    while(unvisited != []):
        temp_len = 1000000000000000000
        temp_item = -1
        for item in unvisited:
            if(distance_matrix[order[len(order)-1]][item] <= temp_len):
                temp_len = distance_matrix[order[len(order)-1]][item]
                temp_item = item
        order.append(unvisited.pop(unvisited.index(temp_item)))
    result = []
    print(order)
    for i in range(len(order)-1):
        for item in path_matrix[order[i]][order[i+1]]:
            result.append(item)
    print(result)
    return result
