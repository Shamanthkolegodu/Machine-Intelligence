#PES1UG19CS444

from queue import PriorityQueue
def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = [start_point]
    closed_list = []
    open_list = PriorityQueue()
    open_list.put((heuristic[start_point], (path, start_point, 0)))
    while(not open_list.empty()):
        calculated_cost, (path, cur_node, current_cost) = open_list.get()          
        if (cur_node not in closed_list):
            closed_list.append(cur_node)
            if (cur_node in goals):
                return path
            next_states = [i for i in range (1, len(cost)) if (cost[cur_node][i]>0 and (i not in closed_list)) ]
            for next in next_states:
                next_cost = current_cost + cost[cur_node][next] + heuristic[next]
                from copy import deepcopy	
                next_path = deepcopy(path)
                next_path.append(next)
                open_list.put((next_cost, (next_path, next, current_cost + cost[cur_node][next])))
        if (path[-1] not in goals):
            path = []
    
    return path		
				
def DFS_Traversal(cost, start_point, goals):
	"""
	Perform DFS Traversal and find the optimal path 
	cost: cost matrix (list of floats/int)
	start_point: Staring node (int)
	goals: Goal states (list of ints)
	Returns:
	path: path to goal state obtained from DFS(list of ints)
	"""
	path=[]
	goals_dict={}

	parent_child_path_dictionary={}
	for i in goals:
		goals_dict[i]=1;
	visited=[False for j in range(len(cost))]
	stack=[start_point]
	goal=0
	parent_child_path_dictionary[start_point]=0
	while(len(stack)):
		ele=stack.pop()
		visited[ele] = True
		if(goals_dict.get(ele)):
			goal=ele
			break
		for node in range(len(cost)-1,0,-1):
			if(not visited[node] and cost[ele][node]>0):
				stack.append(node)
				parent_child_path_dictionary[node]=ele
	path.append(goal)
	if(goal==0):
		return []
	while(goal!=start_point):
		path.append(parent_child_path_dictionary.get(goal))
		goal=parent_child_path_dictionary.get(goal)
	return path[::-1]

