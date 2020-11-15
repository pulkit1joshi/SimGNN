import networkx as nx
import random
import json
from tqdm import tqdm
import os.path

# Max index of graph to generate. Number og graphs generated = 1001 - index
MAX_GRAPHS = 1001

def transfomer(graph_1, graph_2, vals, index):

    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2)) 
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    nodes_2 = graph_2.nodes()
    random.shuffle(list(nodes_2))
    mapper = {node:i for i, node in enumerate(nodes_2)}
    edges_2 = [[mapper[edge[0]], mapper[edge[1]]] for edge in graph_2.edges()]

    graph_1 = nx.from_edgelist(edges_1)
    graph_2 = nx.from_edgelist(edges_2)
    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2))
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    edges_2 = [[edge[0], edge[1]] for edge in graph_2.edges()]
    data = dict()
    data["graph_1"] = edges_1
    data["graph_2"] = edges_2
    data["labels_1"] = [str(graph_1.degree(node)) for node in graph_1]
    data["labels_2"] = [str(graph_2.degree(node))  for node in graph_2]
    nx.set_node_attributes(graph_1, 'Labels', 1)
    nx.set_node_attributes(graph_2, 'Labels', 2)
    print(nx.get_node_attributes(graph_1, "Labels"))
    for x in range(0, len(data["labels_1"])):
        graph_1.nodes[x]["Labels"] = data["labels_1"][x]
    for x in range(0, len(data["labels_2"])):
        graph_2.nodes[x]["Labels"] = data["labels_2"][x]

    print(nx.get_node_attributes(graph_1, "Labels"))
    print(nx.get_node_attributes(graph_2, "Labels"))
    max2=0
    # Finding approximate GED
    for v in nx.optimize_graph_edit_distance(graph_1, graph_2):
        max2 = v
        break
    data["ged"] = max2
    print("Graph Edit distance is:")
    print( data["ged"] )
    if len(data["labels_1"]) == len(nx.nodes(graph_1)) and len(data["labels_2"]) == len(nx.nodes(graph_2)) :
        p=index
        while(os.path.isfile(str(p)+".json")):
            p+=1
            print("exists")
        if len(nx.nodes(graph_1)) == max(graph_1.nodes())+1 and len(nx.nodes(graph_2)) == max(graph_2.nodes())+1:
            with open("../dataset/test/"+str(p)+".json",'w+') as f:
                json.dump(data,f)
                print("Saved:")
                print(str(p)+".json")
                f.close
            z = index + 1
    else:
        z = index
    return z





# Starting index (Used to keep the initialial dataset intact. (Overlapping the original dataset may give label errors.
index = 51
while index <MAX_GRAPHS:
    graph = nx.erdos_renyi_graph(int(random.uniform(5,16)),random.uniform(0.4,0.7))
    error=0
    nodes = graph.nodes()
    clone = nx.from_edgelist(graph.edges())
    counter = 0
    #We want connected graphs for GED calculation
    if nx.is_connected(graph):
        vals = int(abs(random.uniform(5,35)))
        while counter < vals:
            #Randomly add/remove edge and nodes.
            x = random.uniform(0, 1)
            if x>0.5:
                if len(list(clone.edges)) == 0:
                    error=1
                    break
                else :
                    node_1 ,node_2 = random.choice(list(clone.edges))
                    counter = counter + 1
                    if graph.has_edge(node_1,node_2):
                        clone.remove_edge(node_1,node_2)
            else:
                node_1 = random.choice(clone.nodes())
                node_2 = random.choice(clone.nodes())
                if node_1!=node_2 and not clone.has_edge(node_1,node_2) and not graph.has_edge(node_1,node_2):
                    clone.add_edge(node_1,node_2)
                    counter = counter + 1
    if error == 0:
        #try:
            isolate = nx.isolates(clone)
            print(nx.number_of_isolates(clone))
            print("Added")
            if len(clone) == 0 or len(graph) == 0:
                continue
            if nx.is_connected(clone) and nx.is_connected(graph):
                index = transfomer(graph, clone, vals, index)
        #except:
            print("Error")
            continue
