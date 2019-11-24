import os
from datetime import datetime
from torch_geometric.data import Data
import torch
from torch_geometric.utils.convert import to_networkx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import is_strongly_connected, connected_components

def printParOnFile(test_name, log_dir, par_list):

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")

def reconstruct_graphs_from_batch(batch):
    current_graph=0
    batch_x=[]
    batch_reco=[]

    for batch_index,x in zip(batch.batch,batch.x):

        if batch_index != current_graph:
            batch_reco.append((batch_x, batch.y[batch_index-1].item()))
            current_graph +=1
            batch_x=[]

        batch_x.append(x.tolist())
    batch_reco.append((batch_x, batch.y[batch_index - 1].item()))
    #def edge_index
    current_graph = 0
    current_graph_edge_0 = []
    current_graph_edge_1 = []
    edge_index_reco=[]
    zero_graph_edge=0

    for edge_0,edge_1 in zip(batch.edge_index[0],batch.edge_index[1]):
        if batch.batch[edge_0] != current_graph: #essendo tutti disgiunti basta controllare che uno dei due edge sia nel grafo
            edge_index_reco.append([current_graph_edge_0,current_graph_edge_1])
            current_graph+=1
            current_graph_edge_0 = []
            current_graph_edge_1 = []
            zero_graph_edge=min(edge_0,edge_1)
        current_graph_edge_0.append(edge_0.item()-zero_graph_edge)
        current_graph_edge_1.append(edge_1.item()-zero_graph_edge)

    edge_index_reco.append([current_graph_edge_0, current_graph_edge_1])
    graphs=[]
    for graph, graph_edge in zip(batch_reco,edge_index_reco):

        graphs.append(Data(x=torch.tensor(graph[0]),edge_index=torch.tensor(graph_edge),y=torch.tensor([graph[1]])))
    return graphs

def get_graph_diameter(data):
    networkx_graph = to_networkx(data).to_undirected()

    sub_graph_list = [networkx_graph.subgraph(c) for c in connected_components(networkx_graph)]
    sub_graph_diam = []
    for sub_g in sub_graph_list:
        sub_graph_diam.append(diameter(sub_g))
    data.diameter=max(sub_graph_diam)
    return data

def get_diameter(graph):
    networkx_graph = to_networkx(graph).to_undirected()

    sub_graph_list = [networkx_graph.subgraph(c) for c in connected_components(networkx_graph)]
    sub_graph_diam = []
    for sub_g in sub_graph_list:
        sub_graph_diam.append(diameter(sub_g))
    return max(sub_graph_diam)