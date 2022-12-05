"""Convert a Networkx graph and a set of demands to an MPS file that we can solve with CPLEX
"""
import os
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import cplex
from cplex import Cplex

from mcnf_ai.instance import MIPInstance



def get_random_demands(G, n_demands):
    K = np.zeros((n_k, 3), dtype=int)
    nodes_map = {n:i for i,n in enumerate(G.nodes)}
    for i in range(n_k):
        print(list(G.nodes))
        K[i,:2] = np.random.choice(list(G.nodes), size=2, replace=False)
        K[i,2] = np.random.choice(range(1,3))
    return K

def get_mcnf_solver(G, demands, cost="cost", capacity="capacity"):
    nodes_map = {n:i for i,n in enumerate(G.nodes)}

    # number of edges in the graph
    n_edges = len(G.edges)
    n_nodes = len(G.nodes)
    n_k = len(demands)
    # number of variables in the problem
    n_var = n_k * n_edges
    # objective function coefficients and demand-arc-index map
    demand_arc_index = {}
    c = []
    for arc in G.edges:
        for k in range(n_k):
            demand_arc_index[k,arc[0], arc[1], arc[2]] = len(c)
            c.append(G.edges[arc]["cost"])
    c = np.array(c)
    
    # CPLEX solver instance
    cpx = Cplex()
    cpx.parameters.timelimit.set(60)
    cpx.set_log_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)



    #cpx.set_problem_name("G-MCNF")
    cpx.set_problem_type(cpx.problem_type.LP)
    
    # add variable names
    cpx.variables.add(obj=c.astype(float), lb=np.zeros(n_var), names=[f"x_{i}" for i in range(n_var)])
    #print("Number of variables : ", cpx.variables.get_num())
    
    # add objective
    cpx.objective.set_name("cost")
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    ## add upper bound constraints
    constraint_names = [f"c_eq_{i}" for i in range(n_nodes*n_k)]
    constraint_senses = ["E" for i in range(n_nodes*n_k)]
    constraint_rhs=[]
    constraint_lin_expr = []
    for k in range(n_k):
        for i in G.nodes:
            ind,val=[],[]
            b = 0.
            # incoming flow, set to 1
            for j in G.neighbors(i):
                t = 0
                while G.has_edge(i,j,t):
                    ind.append(demand_arc_index[k,i,j,t])
                    val.append(1)
                    t += 1
            for j in G.predecessors(i):
                t=0
                while G.has_edge(j,i,t):
                    ind.append(demand_arc_index[k,j,i, t])
                    val.append(-1)
                    t += 1
            if i==demands[k,0]:
                b = demands[k,2]
            if i==demands[k,1]:
                b = -demands[k,2]
            #if nodes_map[i]==demands[k,0]:
            #    b = demands[k,2]
            #if nodes_map[i]==demands[k,1]:
            #    b = -demands[k,2]

            #print(f"demand {k}, node {i} : b={b}", demands[k], nodes_map[i]==demands[k,0], nodes_map[i]==demands[k,1])
            constraint_lin_expr.append(cplex.SparsePair(ind=ind,val=val))
            constraint_rhs.append(b)
    
    constraint_names += [f"c_ub_{i}" for i in range(n_edges)]
    constraint_senses += ["L" for i in range(n_edges)]
    for (i,j,edge_idx) in G.edges:
        ind,val=[],[]
        for k in range(n_k):
            ind.append(demand_arc_index[k,i,j,edge_idx])
            val.append(1.)
        #print("edge const ind", ind)
        #print("edge const val ", val)
        #print("rhs", float(G.edges[i,j][capacity]))
        constraint_lin_expr.append(cplex.SparsePair(ind=ind,val=val))
        constraint_rhs.append(float(G.edges[i,j,edge_idx][capacity]))
    
    cpx.linear_constraints.add(lin_expr=constraint_lin_expr, 
                               senses=constraint_senses, 
                               rhs=np.array(constraint_rhs), 
                               names=constraint_names)
    return cpx

def plot_mcnf_instance(graph, demands, node_color="#1f78b4", ax=None, cost_name="cost", capacity_name="bandwidth"):
    # plot the graph and demands
    pos = nx.circular_layout(graph)
    if ax is None:
        fig, ax=plt.subplots(1,1)
    nx.draw_networkx(graph, pos=pos, with_labels=True, node_color=node_color, ax=ax)
    demand_G = nx.MultiDiGraph()
    for k in demands:
        demand_G.add_node(k[0])
        demand_G.add_node(k[1])
        demand_G.add_edge(k[0], k[1])

    nx.draw_networkx(demand_G, pos=pos, edge_color="g", width=4, style="dashed", alpha=.5, arrowsize=16, ax=ax)
    if not graph.is_multigraph():
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels={(u,v):f"{graph.edges[u,v][cost_name]}\n{graph.edges[u,v][capacity_name]}" for u,v in graph.edges},
            font_color='red',
            rotate=False,
            label_pos=0.2,
            ax=ax,
            
        )
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels={(k[0],k[1]):k[2] for k in demands},
            font_color='blue',
            rotate=False,
            label_pos=0.2,
            ax=ax,
            
        )

    return ax

def plot_mcnf_solution(graph, demands, solution):
    sol = np.array(solution)
    n_k = len(demands)
    n_edges = int(len(solution) / n_k)
    pos = nx.circular_layout(graph)
    f, ax = plt.subplots(1,n_k)
    if n_k==1:
        ax = [ax]
    for k,demand in enumerate(demands):
        # plot the graph and demands
        nx.draw_networkx(graph, pos=pos, with_labels=True, ax=ax[k])
        demand_G = nx.DiGraph()
        demand_G.add_node(demand[0])
        demand_G.add_node(demand[1])
        demand_G.add_edge(demand[0], demand[1])

        nx.draw_networkx(demand_G, pos=pos, edge_color="g", width=4, style="dashed", alpha=.5, arrowsize=16, ax=ax[k])
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels={(demand[0],demand[1]):demand[2]},
            font_color='blue',
            rotate=False,
            label_pos=0.2,
            ax=ax[k]
        )
        # get flow on each edge attributed to current demand
        edge_k_flows = {}
        for i,edge in enumerate(graph.edges):
            # all flows through current edge
            tot_flow = np.sum(sol[i*n_k:(i+1)*n_k])
            # flow attributable to demand k
            k_flow = sol[i*n_k+k]
            if k_flow:
                edge_k_flows[edge] = f"{int(k_flow)}/{int(tot_flow)}\n{round(100*k_flow / tot_flow,2)}%"
        nx.draw_networkx_edges(graph, pos, 
            edgelist=[e for e in edge_k_flows if edge_k_flows[e]],
            edge_color="blue",
            alpha=.5,
            width=4,
            arrowsize=10,
            ax=ax[k],
        )
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_k_flows,
            font_color='green',
            rotate=False,
            #label_pos=0.2,
            ax=ax[k],
        )
        plt.title(f"Demand {k}: {demand[0]}->{demand[1]}, d_{k}={demand[2]}")
        #plt.show()
        #plt.close("all")
    return f, ax

   


if __name__=="__main__":
    # number of demands per problem
    n,m = 2, 2
    output_dir = f"grid_{n}x{m}_MCNF"
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    n_k = 2#int(.1 * n * m)
    G = nx.generators.lattice.grid_graph((n,m))
    nodes_map = {n:i for i,n in enumerate(G.nodes)}
    # make the graph directed
    G = nx.DiGraph(G)
    nx.set_edge_attributes(G, 2., "capacity")
    nx.set_edge_attributes(G, 1., "cost")
    G=nx.convert_node_labels_to_integers(G)

    solve_status = []
    objective_values = []
    solve_time = []
    primal_gaps=[]


    var_names = []
    for u,v in G.edges:
        for k in range(n_k):
            var_names.append(f"x_{u},{v},{k}")
    
    for inst in range(1000):
        K = get_random_demands(G, n_k)
       
        
        cpx = get_mcnf_solver(G, K)
        instance_path = os.path.join(output_dir, str(inst)+".mps")
        #print(f"Saving problem to {instance_path}")
        #cpx.write(instance_path)
    
        t0 = cpx.get_time()
        cpx.solve()
        t0 = cpx.get_time() - t0
        #print("Solve status: ", cpx.solution.get_status_string())
        if "optimal" in cpx.solution.get_status_string():
            x = np.array(cpx.solution.get_values())
            #print("Solution objective value: ", cpx.solution.get_objective_value())
            #print("Solution type : ", cpx.solution.get_solution_type())
            #print("Solution feasible: ", cpx.solution.is_primal_feasible())
            solve_status.append(1)
            objective_values.append(cpx.solution.get_objective_value())
            #primal_gaps.append(0.)
            solve_time.append(t0)
            print("Saving optimal solution")
            print("demands : ", K)
            for name,val in zip(var_names, x):
                print("\t", name, val)
            cpx.solution.write(os.path.join(output_dir, f"{inst}.sol"))
            plot_mcnf_instance(G, K)
            plt.show()
 
        elif "infeasible" in cpx.solution.get_status_string():
            solve_status.append(0)
            #primal_gaps.append(cpx.solution.MIP.get_mip_relative_gap())
            solve_time.append(t0)
            objective_values.append(np.nan)
            print("Infeasible problem")
        else:
            print("Solution neither optimal or infeasible")
            solve_status.append(2)
            if cpx.solution.is_primal_feasible():
                objective_values.append(cpx.solution.get_objective_value())
                print("Saving solution")
                cpx.solution.write(os.path.join(output_dir, f"{inst}.sol"))
            else:
                objective_values.append(np.nan)
    
            #primal_gaps.append(np.nan)
            solve_time.append(t0)


    
    print("Feasibles : ", np.sum(solve_status))
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("Objective value")
    plt.hist(objective_values, bins=100)
    plt.subplot(2,1,2)
    plt.hist(solve_time, bins=100)
    plt.title("Solve time")
    #plt.subplot(3,1,3)
    #plt.hist(primal_gaps, bins=100)
    #plt.title("MIP relative gap")
    
    plt.show()
