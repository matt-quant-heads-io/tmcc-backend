import yaml
from collections import defaultdict


class Graph:
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices
        
    def add_edge(self,u,v):
        self.graph[u].append(v)
 
    def topological_sort_util(self,v,visited,stack):
        visited[v] = True
        
        for i in self.graph[v]:
            if visited[i] == False:
                self.topological_sort_util(i,visited,stack)
                
        stack.insert(0,v)
 
    def topological_sort(self):
        visited = [False]*self.V
        stack =[]
        
        for i in range(self.V):
            if visited[i] == False:
                self.topological_sort_util(i,visited,stack)
                
        print(stack)
 



# NOTE: TESTING SECTION
execution_plan = None
with open('/home/ubuntu/tmcc-backend/test_execution_pipeline.yaml', 'r') as f:
    execution_plan = yaml.load(f, Loader=yaml.SafeLoader)


def build_graph(plan):
    tasks = plan['execution_plan']['tasks']
    g = Graph(len(tasks)+1)
    for task in tasks:
        if task['dependencies']:
            for d in task['dependencies']:
                g.add_edge(d, task['task_id'])

    return g

     

g = build_graph(execution_plan)
g.topological_sort()