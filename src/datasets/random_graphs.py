import igraph as ig
import numpy as np
import networkx as nx
from abc import ABCMeta, abstractmethod


# Base class
class GraphGenerator(metaclass=ABCMeta):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    @abstractmethod
    def get_random_graph(self):
        raise NotImplementedError()
    

    def _make_random_order(self, A):
        # Randomly permute nodes of A to avoid trivial ordering
        n_nodes = A.shape[0]
        order = np.random.permutation(range(n_nodes))
        A = A[order, :]
        A = A[:, order]
        return A


################################################################
# -------- Gaussian Random Partition Graphs Generator -------- #
################################################################
class GaussianRandomPartition(GraphGenerator):
    def __init__(
        self,
        num_nodes,
        p_in,
        p_out,
        n_clusters
    ):
        """
        Generator of Gaussian Random Partition directed acyclic graphs.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes
        p_in : float
            Probability of edge connection with nodes in the cluster
        p_out : float
            Probability of edge connection with nodes in different clusters
        n_clusters : int
            Number of clusters in the graph. 

        Attributes
        ----------
        size_of_clusters : List[int]
            The size of the graph's clusters. This is randomly sampled from a multinomial
            distribution with parameters TODO: which paremeters?
        """
        if num_nodes/n_clusters < 2:
            raise ValueError("Expected ratio between num_nodes and 'n_clusters' must be at least two"\
                             f"Instead got {num_nodes/n_clusters}. Decrease the number of clusters required.")

        super().__init__(num_nodes)
        self.p_in = p_in
        self.p_out = p_out
        self.n_clusters = n_clusters
        self.size_of_clusters = self._sample_cluster_sizes()


    def get_random_graph(self):
        # print(f"size of the clusters: {size_of_clusters}")

        # Initialize with the first cluster and remove it from the list
        A = self._sample_er_cluster(self.size_of_clusters[0])
        size_of_clusters = np.delete(self.size_of_clusters, [0])

        # Join all clusters together
        for c_size in size_of_clusters:
            A = self._disjoint_union(A, c_size)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A


    def _sample_cluster_sizes(self):
        """Sample the size of each cluset.

        The size of the clusters is sampled from a multinomial distribution, 
        and post-processed to ensure at least 3 nodes per cluster
        """
        cluster_sizes = np.random.multinomial(
            self.num_nodes, pvals=[1/self.n_clusters for _ in range(self.n_clusters)]
        )
        # At least 3 elements per cluster
        while np.min(cluster_sizes) < 3:
            argmax = np.argmax(cluster_sizes)
            argmin = np.argmin(cluster_sizes)
            cluster_sizes[argmax] -= 1
            cluster_sizes[argmin] += 1
        return cluster_sizes


    def _sample_er_cluster(self, cluster_size):
        """Sample each cluster of GRP graphs with Erdos-Renyi model
        """
        A = ErdosRenyi(num_nodes=cluster_size, p_edge=self.p_in).get_random_graph()
        return A


    def _disjoint_union(self, A, c_size):
        """
        Merge adjacency A with cluster of size `c_size` nodes into a DAG.
        
        The cluster is sampled from the Erdos-Rényi model. 
        Nodes are labeled with respect to the cluster they belong.

        Parameters
        ----------
        A : np.array
            Current adjacency matrix
        c_size : int 
            Size of the cluster to generate
        """
        # Join the graphs by block matrices
        n = A.shape[0]
        er_cluster = self._sample_er_cluster(cluster_size=c_size)
        er_cluster = np.hstack([np.zeros((c_size, n)), er_cluster])
        A = np.hstack([A, np.zeros((n, c_size))])
        A = np.vstack([A, er_cluster])

        # Add connections among clusters from A to er_cluster
        for i in range(n):
            for j in range(n, i+c_size):
                if np.random.binomial(n=1, p=self.p_out) == 1:
                    # print(f"edge {(i, j)} between clusters!")
                    A[i, j] = 1

        return A

    
################################################################
# --------------- Erdos-Rényi Graphs Generator --------------- #
################################################################
class ErdosRenyi(GraphGenerator):
    def __init__(
        self,
        num_nodes : int,
        expected_degree : int = None,
        p_edge : float = None
    ):
        """
        Generator of Erdos-Renyi directed acyclic graphs.

        This class is a wrapper of `igraph` Erdos-Renyi graph sampler.
        
        Parameters
        ----------
        d : int
            Number of nodes
        expected_degree : int, default is None
            Expected degree of each node.
        p_edge : float, default is None
            Probability of edge between each pair of nodes.
        """
        if expected_degree is not None and p_edge is not None:
            raise ValueError("Only one parameter between 'p' and 'm' can be assigned a value."\
                             f" Got instead m={expected_degree} and p={p_edge}.")
        if expected_degree is None and p_edge is None:
            raise ValueError("Please provide a value for one of argument between 'm' and 'p'.")

        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.p_edge = p_edge


    def get_random_graph(self):
        A = np.zeros((self.num_nodes, self.num_nodes))
        while np.sum(A) < 2:
            if self.p_edge is not None:
                G_und = ig.Graph.Erdos_Renyi(n=self.num_nodes, p=self.p_edge)
            elif self.expected_degree is not None:
                G_und = ig.Graph.Erdos_Renyi(n=self.num_nodes, m=self.expected_degree*self.num_nodes)
            A_und = ig_to_adjmat(G_und)
            A = acyclic_orientation(A_und)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A
    

################################################################
# --------------- Barabasi Albert Graphs Generator --------------- #
################################################################
class BarabasiAlbert(GraphGenerator):
    def __init__(
        self,
        num_nodes : int,
        expected_degree : int,
        preferential_attachment_out: bool = True
    ):
        """
        Generator of Scale Free directed acyclic graphs.

        This class is a wrapper of `igraph` Barabasi graph sampler.
        
        Parameters
        ----------
        d : int
            Number of nodes
        expected_degree : int
            Expected degree of each node.
        preferential_attachment_out: bool, default True
            Select the preferential attachment strategy. If True,
            new nodes tend to have incoming edge from existing nodes with high out-degree.
            Else, new nodes tend to have outcoming edge towards existing nodes with high in-degree.
        """
        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.preferential_attachment_out = preferential_attachment_out

    def get_random_graph(self):
        A = np.zeros((self.num_nodes, self.num_nodes))
        while np.sum(A) < 2:
            G = ig.Graph.Barabasi(n=self.num_nodes, m=self.expected_degree, directed=True)
            A = ig_to_adjmat(G)
            if self.preferential_attachment_out:
                A = A.transpose(1, 0)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A
        


################################################################
# -------------------- Utilities -------------------- #
################################################################

def acyclic_orientation(A):
    return np.triu(A, k=1)

def ig_to_adjmat(G : ig.Graph):
    return np.array(G.get_adjacency().data)

def graph_viz(A : np.array):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nx.draw_networkx(G)
