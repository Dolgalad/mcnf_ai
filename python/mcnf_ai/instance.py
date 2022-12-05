import sys
import numpy as np
import networkx as nx
from pysmps import smps_loader as smps
from cplex import Cplex

import matplotlib.pyplot as plt

class MIPInstance:
    """MIP instance data container
    """
    def __init__(self, name, objective_name, row_names, col_names, variable_types, constraint_types, c, A, rhs_names, rhs, bnd_names, bnd):
        self.name = name
        self.objective_name = objective_name
        self.row_names = row_names
        self.col_names = col_names
        self.variable_types = variable_types
        self.constraint_types = constraint_types
        self.c = c
        self.A = A
        self.rhs_names = rhs_names
        self.rhs = rhs
        self.bnd_names = bnd_names
        self.bnd = bnd

        self.filename = None

        if len(self.rhs_names) == 0:
            self.rhs_names = ["rhs"]
            self.rhs = {"rhs": np.zeros(self.constraint_count())}

    def variable_count(self):
        """Get number of variables
        """
        return len(self.c)
    def constraint_count(self):
        """Get number of constraints
        """
        return self.A.shape[0]
    def __str__(self):
        """String representation
        """
        return f"MIPInstance({self.name}, n_vars={self.variable_count()}, n_constraints={self.constraint_count()})"
    
    def is_standard_form(self):
        """Check if current instance is in standard form
        """
        return all([ct == "L" for ct in self.constraint_types])
    def standard_form(self): # TODO : standard or canonical form ??
        """Returns the current problem in standard form
        """
        new_A = []
        new_constraint_types = []
        new_row_names = []
        for i in range(self.constraint_count()):
            constraint_type = self.constraint_types[i]
            row_name = self.row_names[i]
            if constraint_type == "L":
                new_A.append(self.A[i,:])
                new_constraint_types.append("L")
                new_row_names.append(row_name)
            if constraint_type == "E":
                new_A.append(self.A[i,:])
                new_A.append(-self.A[i,:])
                new_constraint_types += ["L", "L"]
                new_row_names += [f"{row_name}_u", f"{row_name}_l"]
            if constraint_type == "G":
                new_A.append(-self.A[i,:])
                new_constraint_types += ["L"]
                new_row_names.append(f"{row_name}_l")
        new_A = np.array(new_A)
        return MIPInstance(self.name, self.objective_name, new_row_names, self.col_names, self.variable_types, new_constraint_types, self.c, new_A, self.rhs_names, self.rhs, self.bnd_names, self.bnd)
    
    def vcgraph(self):
        return self.bipartite_graph()
    def bipartite_graph(self):
        """Build the bipartite graph representing the problem
        """
        if not self.is_standard_form():
            return self.standard_form().bipartite_graph()

        G = nx.Graph()
        # add a node for every variable
        G.add_nodes_from([(var, {"type":"variable","features":[self.c[i]]}) for i,var in enumerate(self.col_names)])
        # add a node for every constraint
        G.add_nodes_from([(constr, {"type":"constraint","features":[self.rhs[self.rhs_names[0]][i]]}) for i,constr in enumerate(self.row_names)])
        # add variable-constraint edges
        for i in range(self.constraint_count()):
            for j in range(self.variable_count()):
                if self.A[i,j] != 0.:
                    G.add_edge(self.col_names[j], self.row_names[i], coeff=self.A[i,j])
        return G



    @staticmethod
    def load(filename: str):
        """Load data from MPS file
        """
        mip = MIPInstance(*smps.load_mps(filename))
        mip.filename = filename
        return mip

    @staticmethod
    def dump(instance, file):
        pass




if __name__=="__main__":
    if len(sys.argv)<=1:
        print("no mps file provided.")
        exit()

    mps_path = sys.argv[1]

    data = smps.load_mps(mps_path)
    print(data)
    print()

    mps_instance = MIPInstance.load(mps_path)
    print(mps_instance)
    print()

    G = mps_instance.bipartite_graph()
    print(G)

    # drawing the bipartite graph
    pos = nx.bipartite_layout(G, nodes=[n for n, nd in G.nodes(data=True) if nd["type"]=="variable"], center=(0,0))

    node_color = []
    for n, data in G.nodes(data=True):
        if data["type"]=="variable":
            node_color.append("#aaaaff")
        else:
            node_color.append("#aaffaa")
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=node_color, node_size=700)
    #nx.draw_networkx_labels(G, pos=pos)
    plt.show()
