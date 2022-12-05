# Multi-Commodity Network Flow problem solving with AI
Implementation and experimental results relating to solving MCNF instances using AI. 

## Graphs and demands
A collection of graphs has been given to us, they are contained in the `subgraph` directory. Instantiating a subgraph can be done by
using the `subgraph` submodule.

```python
>>> from mcnf_ai.subgraph import load_subgraph
>>> G = load_subgraph(1, 1)
>>> G
<networkx.classes.digraph.DiGraph object at 0x7f5dae863400>
```

## Instances
The `dataset_generator` module is used to create datasets. Every instance is composed of :
- a networkx graph: edges of the graph must have a `cost` and `bandwidth` parameters
- a list of demands : `numpy.array` with size `(n_demands,3)`, each row is a demands `(origin_node,target_node,volume)`

## Graph convolution networks

## Graph contraction
To help the network in its learning task we can simplify the instances using various methods. One such approach relies on transforming
the graph by removing nodes or edges making it smaller. The smaller instances makes the training and prediction tasks more manageble.
For evaluation the solution needs to be projected back into the original state space before comparing it to the optimal solution.

Contraction methods belong to one of two categories:
- exact contractions: there is a guarantee that the optimal solution to the contracted problem is optimal in the original space after projecting
- approximate contractions: we loose the optimal character of the solution in the original space after projecting.

Contractions are implemented in the `contraction` submodule, the expose a `contract(graph: networkx.graph, demands: numpy.array)` function that applies the contraction to the graph as well as a `expand(solution: numpy.array)` that projects a solution from the contracted
space back into the original space.

```python
pass
```

## Evaluation process
In the best of worlds what we are interested in is the ability of a model to predict an accurate assignment of all variables in the
problem. 

If `x_star` is the optimal solution of an instance `I` and `x_pred` the predicted assignment returned by the model, the following
metrics are used to evaluate the quality of the prediction : 
- optimality gap : `|c(x_star) - c(x_pred)| / c(x_star)`
- prediction quality : non-zero variable prediction accuracy, precision, recall and F1-score

