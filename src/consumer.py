import numpy as np
import networkx as nx

class IslandConsumer:
    """
    Processes islands to perform GCN computations, simulating the data flow
    of an I-GCN accelerator.
    """
    def __init__(self, graph: nx.Graph, node_features: dict):
        """
        Initializes the Island Consumer.

        Args:
            graph (nx.Graph): The full graph structure.
            node_features (dict): A dictionary mapping node IDs to their feature vectors.
        """
        self.graph = graph
        self.initial_features = node_features
        self.updated_features = {}
        if not node_features:
            raise ValueError("Node features cannot be empty.")
        self.feature_dim = len(next(iter(self.initial_features.values())))


    def _aggregate(self, node: int, island_nodes_set: set):
        """
        Aggregates features from a node's neighbors that are within the same island.
        This simulates the on-chip data locality benefit.
        """
        neighbor_features = [
            self.initial_features[neighbor]
            for neighbor in self.graph.neighbors(node)
            if neighbor in island_nodes_set
        ]
        
        if neighbor_features:
            return np.sum(neighbor_features, axis=0)
        else:
            return np.zeros(self.feature_dim)

    def _combine(self, node: int, aggregated_vector: np.ndarray):
        """
        Combines the aggregated vector with the node's own feature and applies ReLU activation.
        """
        combined_vector = self.initial_features[node] + aggregated_vector
        return np.maximum(0, combined_vector) # ReLU activation

    def process_island(self, island: list):
        """
        Simulates the GCN computation for all nodes within a single island.
        """
        print(f"\nConsuming island: {island}...")
        island_nodes_set = set(island)

        for node in island:
            aggregated_vector = self._aggregate(node, island_nodes_set)
            self.updated_features[node] = self._combine(node, aggregated_vector)
            
        print(f"  -> Finished processing island. Updated features for {len(island)} nodes.")
