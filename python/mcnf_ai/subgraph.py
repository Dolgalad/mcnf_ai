import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools


graph_directory = "/home/aschulz/Documents/huawei_project/subgraph"

def load_full_graph():
    # initialize empty graph
    G = nx.DiGraph()
    
    node_data_path = os.path.join(graph_directory, "out_node.csv")
    node_data = np.loadtxt(node_data_path, delimiter=",", skiprows=1, dtype=object)

    for node in node_data:
        node_id = int(node[0])
        if not G.has_node(node_id):
            G.add_node(int(node[0]), layer=int(node[3]))
    
    link_data_path = os.path.join(graph_directory, "out_link.csv")
    link_data = np.loadtxt(link_data_path, delimiter=",", skiprows=1, dtype=object)
    
    for link in link_data:
        edge_ids = (int(link[2]), int(link[4]))
        if not G.has_edge(*edge_ids):
            G.add_edge(*edge_ids, cost=int(link[7]), bandwidth=int(link[8]))
    subgraph_files = [os.path.join(graph_directory, f) for f in os.listdir(graph_directory) if f.startswith("subGraph")]
    for subgraph_f in subgraph_files:
        fn = os.path.basename(subgraph_f)
        subgraph_layer = int(fn.replace(".csv","").split("_")[1])
        subgraph_id = int(fn.replace(".csv","").split("_")[2])

        subgraph_data = np.loadtxt(subgraph_f, delimiter=",", skiprows=1, dtype=object)
        nx.set_node_attributes(G, {int(n): subgraph_id for n in subgraph_data[:,0]}, name="subgraph_id")
        nx.set_node_attributes(G, {int(n): subgraph_layer for n in subgraph_data[:,0]}, name="subgraph_level")
    
    return G

def load_full_demands():
    # demand file
    demand_data_path = os.path.join(graph_directory, "demand.csv")
    demand_data = np.loadtxt(demand_data_path, delimiter=",", skiprows=1, dtype=int)
    return [(x[1], x[2], x[3]) for x in demand_data]

def load_subgraph(subgraph_level, subgraph_id, G=None):
    return get_subgraph(subgraph_level, subgraph_id, G)
def get_subgraph(subgraph_level, subgraph_id, G=None):
    if G is None:
        G = load_full_graph()
    subgraph_data_path = os.path.join(graph_directory, f"subGraph_{subgraph_level}_{subgraph_id}.csv")
    subgraph_data = np.loadtxt(subgraph_data_path, delimiter=",", dtype=object, skiprows=1)
    subgraph_nodes = [int(n) for n in subgraph_data[:,0]]
    #print("subgraph_nodes: ", subgraph_nodes)
    #print(G.has_edge(165, 166))
    #subgraph_nodes = [int(n) for n in G.nodes if check_node(n)]
    return G.subgraph(subgraph_nodes)

def iter_subgraph():
    for f in os.listdir(graph_directory):
        if f.startswith("subGraph") and f.endswith(".csv"):
            [lvl, _id] = f.replace(".csv","").split("_")[1:]
            yield get_subgraph(lvl, _id)

def get_subgraph_network():
    fullG = load_full_graph()
    G = nx.DiGraph()
    subgraphs = {}
    for f in os.listdir(graph_directory):
        if not f.startswith("subGraph"):
            continue
        fdata = f.replace(".csv","").split("_")
        lvl,ind = int(fdata[1]), int(fdata[2])
        g = get_subgraph(lvl,ind, fullG)
        subgraphs[lvl,ind] = g
        G.add_node((lvl,ind))
    for k in subgraphs:
        for l in subgraphs:
            if k!=l:
                k_nodes = list(subgraphs[k].nodes)
                l_nodes = list(subgraphs[l].nodes)
                for [a,b] in itertools.product(k_nodes,l_nodes):
                    if fullG.has_edge(a,b):
                        G.add_edge(k,l)
                        break
    return G
                    
        

if __name__=="__main__":
    # plot the subgraph network
    sG = get_subgraph_network()
    layer_nodes = {} 
    for node in sG.nodes:
        if node[0] in layer_nodes:
            layer_nodes[node[0]].append(node)
        else:
            layer_nodes[node[0]] = [node]
    nlist = list(layer_nodes.values())
    nlist.sort(key = lambda l: len(l))
   
    pos = nx.drawing.layout.shell_layout(sG, nlist)

    plt.figure(figsize=(15,15))
    nx.draw_networkx(sG, pos=pos, with_labels=True, node_size=500)
    plt.show()
    exit()


    G = load_full_graph()
    print("Graph: ", G)
    K = load_full_demands()
    print("Demands: ", len(K))
    
    # get a subgraph
    graph_img_dir = "graph_images"
    os.makedirs(graph_img_dir, exist_ok=True)
    
    for f in os.listdir(graph_directory):
        if not f.startswith("subGraph"):
            continue
        fdata = f.replace(".csv", "").split("_")
        img_filename = os.path.join(graph_img_dir, f"subGraph_{fdata[1]}_{fdata[2]}.png")
        #if os.path.exists(img_filename):
        #    continue
        print(f"Getting {f}...", flush=True)
        g = get_subgraph(int(fdata[1]), int(fdata[2]), G)
        print("done", flush=True)
        #if len(g.nodes) > 100:
        #    continue
        
        layer_nodes = {} 
        for node in g.nodes:
            nodedata=G.nodes[node]
            if nodedata["layer"] in layer_nodes:
                layer_nodes[nodedata["layer"]].append(node)
            else:
                layer_nodes[nodedata["layer"]] = [node]
        nlist = list(layer_nodes.values())
        nlist.sort(key = lambda l: len(l))
    
        pos = nx.drawing.layout.shell_layout(G, nlist)
        print(f, g)
        plt.figure(figsize=(15,15))
        plt.title(f"{f}\n{str(g)}")
        nx.draw_networkx(g, pos=pos, with_labels=True, node_size=500)
        plt.savefig(img_filename)
        plt.close("all")
