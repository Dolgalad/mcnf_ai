"""Graph contraction class
"""
import numpy as np
import networkx as nx
from networkx.algorithms.components.weakly_connected import number_weakly_connected_components, weakly_connected_components


from mcnf_ai.mcnf import MCNF
from mcnf_ai.path import Path, node_sequence_to_edge_sequence

class GraphExpander:
    def __init__(self, original_mcnf):
        self.original_mcnf = original_mcnf
    def repair_path(self,path, graph):
        nedges = []
        for e in path.edges:
            chain = graph.edges[e].get("chain", None)
            if chain:
                nedges += chain
            else:
                nedges += [e]
        return Path(edges=nedges)
    def apply(self, solution: np.array, mcnf: MCNF) -> np.array:
        """Expand the solution
        """
        # compute paths and flows used to route demands on contracted graph
        paths, flows = mcnf.solution_paths(solution)
        
        npaths, nflows = {},{}
        for k, demand in enumerate(self.original_mcnf.demands):
            left_extr,right_extr=[],[]
            if demand[0]!=mcnf.demands[k,0]:
                left_extr = nx.algorithms.shortest_path(self.original_mcnf.G, demand[0], mcnf.demands[k,0], weight="cost")
                left_extr = node_sequence_to_edge_sequence(left_extr)
            if demand[1]!=mcnf.demands[k,1]:
                right_extr = nx.algorithms.shortest_path(self.original_mcnf.G, mcnf.demands[k,1], demand[1], weight="cost")
                right_extr = node_sequence_to_edge_sequence(right_extr)
            npathk, nflowk = [],[]
            for path,flow in zip(paths[k], flows[k]):
                npath = left_extr + self.repair_path(path, mcnf.G).edges + right_extr
                npathk.append(Path(npath))
                nflowk.append(flow)
            npaths[k] = npathk
            nflows[k] = nflowk
        return npaths, nflows


     
        
        
    def __call__(self, mcnf: MCNF) -> MCNF:
        return self.apply(mcnf)

class GraphContractor:
    def node_sequence_to_edge_sequence(self, seq):
        a = []
        for i in range(1, len(seq)):
            a.append((seq[i-1], seq[i], 0))
        return a
    def component_node_features(self, component, node, G, forward=True):
        """Compute the features given to the arc (component_node, node) if forward, (node, component) if not forward
        """
        features = []
        for c in component:
            if G.has_edge(node,c) and not forward:
                features.append([G.edges[node,c]["cost"], G.edges[node,c]["bandwidth"]])
            if G.has_edge(c,node) and forward:
                features.append([G.edges[c,node]["cost"], G.edges[c,node]["bandwidth"]])
        if len(features)==1:
            return features[0]
        return np.min(features, axis=0)
    
    def component_node_is_extremity(self,node, component, G):
        #print("in compontnent extremities ")
        node_neighbors = list(G.predecessors(node)) + list(G.successors(node))
        node_neighbors = set(node_neighbors)
        flags = [n in component for n in node_neighbors]
        #print(node_neighbors)
        #print(flags)
        #print(all(flags))
        return not all(flags)
    
    def get_component_extremities(self, component, G):
        c = []
        for node in component:
            if self.component_node_is_extremity(node, component, G):
                c.append(node)
        return c
    def get_component_neighbors(self, component, G):
        extremities = self.get_component_extremities(component, G)
        c = []
        for extr in extremities:
            ns = list(G.predecessors(extr)) + list(G.successors(extr))
            c += [n for n in ns if not n in component]
        return list(set(c))
 
    def contract_graph(self, G, demands, exclude_nodes=[]):
        """Constructs a new graph H that is defined as G in which all chains have been replaced by 
        a new node.
        """
        # new node increment 
        new_node_inc = max(G.nodes) + 1
        # temporary graph with all excluded nodes removed
        tG = G.subgraph([n for n in G.nodes if not n in exclude_nodes])
        # find all nodes that have degree greater then 2 (4 if graph is directed)
        removable_nodes = [n for n in G.nodes if G.degree(n)>4] + exclude_nodes
        # subgraph with those nodes removed
        subG = G.subgraph([n for n in G.nodes if not n in removable_nodes])
        # mapping of the original nodes to new nodes
        node_map = {n:n for n in removable_nodes}
        components = list(weakly_connected_components(subG))
    
        cG = nx.MultiDiGraph(G.subgraph(removable_nodes))
        for component in components:
            component = list(component)
            extr = list(self.get_component_extremities(component, G))
            neighbs = list(self.get_component_neighbors(component, G))
            if len(neighbs)==1:
                # only 1 neighbor
                # remove the component from the graph and map its nodes to the only neighbor
                for n in component:
                    node_map[n] = neighbs[0]
            else:
                chain = nx.Graph(G.subgraph(component + neighbs))
                if chain.has_edge(neighbs[0], neighbs[1]):
                    chain.remove_edge(neighbs[0], neighbs[1])
                if chain.has_edge(neighbs[1], neighbs[0]):
                    chain.remove_edge(neighbs[1], neighbs[0])
                # sum up costs on component
                cost_sum = sum([nd["cost"] for _,_,nd in chain.edges(data=True)]+[0.])
                # get min bandwidth
                bw_min = min([nd["bandwidth"] for _,_,nd in chain.edges(data=True)])
                # add edge 
                direct_path = nx.algorithms.shortest_path(chain, source=neighbs[0], target=neighbs[1], weight="cost")
                direct_path = self.node_sequence_to_edge_sequence(direct_path)
                revers_path = nx.algorithms.shortest_path(chain, source=neighbs[1], target=neighbs[0], weight="cost")
                revers_path = self.node_sequence_to_edge_sequence(revers_path)
    
                cG.add_edge(neighbs[0], neighbs[1], cost=cost_sum, bandwidth=bw_min, type="chain", chain=direct_path)
                cG.add_edge(neighbs[1], neighbs[0], cost=cost_sum, bandwidth=bw_min, type="chain", chain=revers_path)
                for n in component:
                    node_map[n] = (neighbs[0], neighbs[1])
    
        cdemands = []
        for demand in demands:
            cdemands.append([node_map[demand[0]], node_map[demand[1]], demand[2]])
        cdemands = np.array(cdemands)

        
        cmcnf = MCNF(cG, cdemands)
        expander = GraphExpander(MCNF(G, demands))
     
        return cmcnf, expander


    def apply(self, mcnf: MCNF) -> (MCNF, GraphExpander): 
        """Take a MCNF instance and return a new instance defined on the contracted graph
        """
        return self.contract_graph(mcnf.G, mcnf.demands)
    def __call__(self, mcnf: MCNF) -> (MCNF, GraphExpander):
        return self.apply(mcnf)

