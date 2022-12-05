class Path:
    def __init__(self, edges=[], origin=None, target=None):
        self.edges = edges
        self.origin = origin
        self.target = target
        if len(edges):
            self.origin = edges[0][0]
            self.target = edges[-1][1]
    def startswith(self, u):
        return self.origin == u
    def endswith(self, u):
        return self.target == u
    def is_st_path(self,s,t):
        return self.startswith(s) and self.endswith(t)
    def add_edge(self, e):
        self.edges.append(e)
        if self.origin is None:
            self.origin = e[0]
        self.target = e[1]
    def __len__(self):
        return len(self.edges)
    def __str__(self):
        return f"Path ({self.edges})"
    @staticmethod
    def from_node_sequence(nodes):
        edges=[]
        for i in range(1,len(nodes)):
            edges.append((nodes[i-1], nodes[i]))
        return Path(edges)
    @staticmethod
    def from_edge_sequence(edges):
        return Path(edges)

def node_sequence_to_edge_sequence(seq):
   a = []
   for i in range(1, len(seq)):
       a.append((seq[i-1], seq[i], 0))
   return a

