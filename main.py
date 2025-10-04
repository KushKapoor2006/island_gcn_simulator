import networkx as nx
import numpy as np
# Updated import path to reflect the 'src' directory
from src.locator import IslandLocator
from src.consumer import IslandConsumer

def main():
    """
    Main function to run the I-GCN simulation pipeline.
    """
    # --- 1. Define Simulation Parameters ---
    DEGREE_THRESHOLD = 4      # H_d: Min degree to be considered a hub
    MAX_ISLAND_SIZE = 10      # I_s: Max nodes per island
    FEATURE_DIM = 8           # Dimension of node feature vectors

    # --- 2. Load Graph and Generate Mock Features ---
    print("Loading graph and generating features...")
    G = nx.karate_club_graph()
    node_features = {node: np.random.rand(FEATURE_DIM).round(2) for node in G.nodes()}
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- 3. Run Island Location ---
    locator = IslandLocator(G, degree_threshold=DEGREE_THRESHOLD, max_island_size=MAX_ISLAND_SIZE)
    found_islands = locator.locate_islands()

    if not found_islands:
        print("No islands were found. Exiting.")
        return

    # --- 4. Run Island Consumption ---
    print("\n--- Starting Island Consumption ---")
    consumer = IslandConsumer(G, node_features)
    for island in found_islands:
        consumer.process_island(island)

    print(f"\n--- I-GCN Simulation Complete ---")
    print(f"Total nodes with updated features: {len(consumer.updated_features)}")
    
    # Optional: Print the updated feature for a sample node
    if consumer.updated_features:
        sample_node = list(consumer.updated_features.keys())[0]
        print(f"\nExample update for node {sample_node}:")
        print(f"  Initial feature: {node_features[sample_node]}")
        print(f"  Updated feature: {consumer.updated_features[sample_node].round(2)}")


if __name__ == "__main__":
    main()
