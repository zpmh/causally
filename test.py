"""from causally.graph import random_graph

graph_generator = random_graph.ErdosRenyi(
   num_nodes=10,
   p_edge=0.4
)
adjacency = graph_generator.get_random_graph()
print(adjacency)
#----------
import numpy as np
from causally.scm.causal_mechanism import PredictionModel

class SumOfSquares(PredictionModel):
   def predict(self, X):
      effect = np.square(X).sum(axis=1)
      return effect

mechanism = SumOfSquares()
parents = np.random.standard_normal(size=(1000, 2)) # 1000 samples, 2 parents
child = mechanism.predict(parents)
#---------

import numpy as np
from torch import nn
from causally.scm.noise import Distribution, MLPNoise, Normal

# Generate sample from a Normal distribution
normal_generator = Normal()
normal_samples = normal_generator.sample((1000, ))


# Generate samples from an Laplace distribution
class Laplace(Distribution):
   def __init__(self, loc: float=1.0, scale:float=2.0):
      self.loc = loc
      self.scale = scale

   def sample(self, size: tuple[int]):
      return np.random.laplace(self.loc, self.scale, size)

laplace_generator = Laplace()
laplace_samples = laplace_generator.sample((1000, ))

# Generate samples from a random distribution
mlp_generator = MLPNoise(
   hidden_dim=100,
   activation=nn.Sigmoid(),
   bias=False,
)
mlp_samples = mlp_generator.sample((1000, ))

print(mlp_samples)"""

import causally.scm.scm as scm
import causally.graph.random_graph as rg
import causally.scm.noise as noise
import causally.scm.causal_mechanism as cm

# Erdos-Renyi graph generator
graph_generator = rg.ErdosRenyi(num_nodes=10, expected_degree=1)

# Generator of the noise terms
noise_generator = noise.MLPNoise()

# Nonlinear causal mechanisms (parametrized with a random neural network)
causal_mechanism = cm.NeuralNetMechanism()

# Generated the data
model = scm.AdditiveNoiseModel(
      num_samples=20000,
      graph_generator=graph_generator,
      noise_generator=noise_generator,
      causal_mechanism=causal_mechanism,
      seed=42
)
dataset, groundtruth = model.sample()

print(dataset.shape)


from pgmpy.estimators import PC
from pgmpy.models import BayesianModel
import pandas as pd

column_names = [f'Var{i}' for i in range(10)]
df = pd.DataFrame(dataset, columns=column_names)
#introduce some dependencies?
df['Var6'] = df['Var1'] - df['Var2']
df['Var4'] += df['Var1']

# Initialize PC algorithm
pc = PC(df)
dag = pc.skeleton_to_pdag(*pc.build_skeleton(ci_test='pearsonr'))

# Estimate the skeleton (undirected edges)
#skeleton = pc.estimate()

# Orient the edges to get a DAG
#dag = pc.skeleton_to_pdag(skeleton)

# Print the DAG
print("Estimated DAG:")
print(dag.edges())






"""visualizing the DAG for groundtruth"""
import networkx as nx
import matplotlib.pyplot as plt

# Convert the adjacency matrix to a directed graph
G = nx.from_numpy_array(groundtruth, create_using=nx.DiGraph)

# Visualize the graph
pos = nx.random_layout(G)  # You can use different layout algorithms
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', arrowsize=20)
plt.show()


#print(dataset)

#print('------------')
#print(groundtruth)

#from pcalg import estimate_skeleton, estimate_cpdag
#from gsq.ci_tests import ci_test_dis, ci_test_bin
from scipy.stats import pearsonr

# implement PC alg to see if and how this works
#columns = [f"Variable_{i+1}" for i in range(10)]
#df = pd.DataFrame(samples, columns=columns)

# Use PC algorithm to estimate the skeleton
#(g, sep_set) = estimate_skeleton(gaussian_ci_test, dataset, alpha=0.01)
#g = estimate_cpdag(skel_graph=g, sep_set=sep_set)

#pos = nx.random_layout(g)  # You can use different layout algorithms
#nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', arrowsize=20)
#plt.show()


# Use BayesianEstimator for further orientation
#estimator = ConstraintBasedEstimator(df)
#model = estimator.estimate(model="bn", method="pc", skeleton=skeleton)

# Print the DAG
#print("Estimated DAG:")
#print(model.to_directed().edges())