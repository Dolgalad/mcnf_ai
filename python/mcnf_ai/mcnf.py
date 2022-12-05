import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from . import graph_to_mps
from mcnf_ai.path import Path
       

class MCNF:
    def __init__(self, G, demands):
        self.G = nx.MultiDiGraph(G)
        self.demands = np.array(demands)
        self.edge_index = {e:i for i,e in enumerate(G.edges)}
    def num_demands(self):
        return self.demands.shape[0]
    def num_variables(self):
        return self.num_demands() * len(self.G.edges)
    def get_cplex_solver(self):
        return graph_to_mps.get_mcnf_solver(self.G, self.demands)
    def solve_with_cplex(self, **kwargs):
        cpx = graph_to_mps.get_mcnf_solver(self.G, self.demands, **kwargs)
        t0 = cpx.get_time()
        cpx.solve()
        t0 = cpx.get_time() - t0
        status_str = cpx.solution.get_status_string()
        if status_str=="optimal":
            sol = np.array(cpx.solution.get_values())
            return status_str, sol, t0
        return status_str, None, t0
    def edge_demand_variable_index(self, edge, k):
        """Returns the index of the variable corresponding to the edge and demand index.
        """
        return self.edge_index[edge] * self.num_demands() + k
    def flow_on_path(self, path, k, solution):
        """Returns the amount of flow using the given path, takes the minimum flow value of edge-demand variables
        """
        flow_values = [solution[self.edge_demand_variable_index(e,k)] for e in path.edges]
        return min(flow_values)
    def path_to_solution(self, k, path):
        """Encode a path for demand k into the solution space
        """
        s = np.zeros((len(self.G.edges), self.num_demands()))
        for edge in path.edges:
            s[self.edge_index[edge], k] = 1.
        return s.flatten()
    def paths_to_solution(self, paths, flows):
        s = np.zeros((len(self.G.edges), self.num_demands())).flatten()

        for k in paths:
            for path,flow in zip(paths[k], flows[k]):
                s += self.path_to_solution(k, path) * flow
        return s
    def solution_paths(self, solution):
        """Returns a dictionary indexed by the demand index and with values that can either be a path or list of paths. One demand can be resolved by multiple paths.

        Returns two dictionaries: demand_paths, demand_flows
        `demand_paths` is a dictionary indexed by the index of the demand and with a list of Path objects as values. `demand_flows` is indexed by the demand index and contains a list of flow values, the length of the list corresponds to the number of paths used by the demand.
        """
        if solution.size != self.num_variables():
            raise Exception(f"Solution has size {solution.size} whereas there are {self.num_variables()} variables in this instance.")
        demand_paths = {}
        demand_flows = {}
        edges = list(self.G.edges)
        m = len(edges)
        n_k = self.demands.shape[0]
        for k in range(len(self.demands)):
            # get the path used for current demand
            # get the arcs used for current demand
            destdict = {}
            kpos = k + np.arange(m)*n_k
            used_edges = []
            for j,f in enumerate(solution[kpos]):
                if f != 0.:
                    u,v,t = edges[j]
                    if u in destdict:
                        destdict[u] = [destdict[u], (v,t)]
                    else:
                        destdict[u] = (v,t)
                    used_edges.append(edges[j])

            [s,t,v] = self.demands[k]
            paths = [Path(edges=[],origin=s)]
            ff = any([not p.is_st_path(s,t) for p in paths])
            while ff:
                for i in range(len(paths)):
                    if not paths[i].is_st_path(s,t):
                        last_node = paths[i].target
                        if last_node is None:
                            last_node = paths[i].origin
                        if isinstance(destdict[last_node], list):
                            # path split, add a new path
                            new_edge = tuple([last_node] + list(destdict[last_node][0]))
                            paths[i].add_edge(new_edge)
                            for j in range(1,len(destdict[last_node])):
                                new_path = Path(edges=paths[i].edges[:-1])
                                new_edge = tuple([last_node] + list(destdict[last_node][j]))
                                new_path.add_edge(new_edge)
                                paths.append(new_path)
                        else:
                            new_edge = tuple([last_node] + list(destdict[last_node]))
                            paths[i].add_edge(new_edge)
                ff = any([not p.is_st_path(s,t) for p in paths])
            flows = [self.flow_on_path(path, k, solution) for path in paths]
            demand_flows[k] = flows
            demand_paths[k] = paths
            
        return demand_paths, demand_flows
    def draw(self, **kwargs):
        return graph_to_mps.plot_mcnf_instance(nx.DiGraph(self.G), self.demands, **kwargs)
    def draw_solution(self, solution, **kwargs):
        return graph_to_mps.plot_mcnf_solution(nx.DiGraph(self.G), self.demands, solution)
