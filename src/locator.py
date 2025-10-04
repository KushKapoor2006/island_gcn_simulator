from collections import deque
import networkx as nx

class IslandLocator:
    """
    A class that encapsulates the entire island-finding pipeline.
    It locates hubs, generates tasks, and uses BFS engines to form islands.
    """
    def __init__(self, graph: nx.Graph, degree_threshold: int, max_island_size: int):
        """
        Initializes the IslandLocator.

        Args:
            graph (nx.Graph): The input graph.
            degree_threshold (int): The minimum degree for a node to be a hub.
            max_island_size (int): The maximum number of nodes allowed in an island.
        """
        self.graph = graph
        self.degree_threshold = degree_threshold
        self.max_island_size = max_island_size
        self.nodes_in_islands = set()

    def _find_hubs(self):
        """Identifies hub nodes in the graph."""
        print(f"Searching for hubs with degree > {self.degree_threshold}...")
        hubs = [node for node in self.graph.nodes() if self.graph.degree(node) > self.degree_threshold]
        print(f"Found {len(hubs)} hubs: {hubs}")
        return hubs

    def _generate_tasks(self, hubs):
        """Generates a task queue from the list of hubs."""
        task_queue = deque(hubs)
        print(f"Task queue initialized with {len(task_queue)} tasks.")
        return task_queue

    def _form_single_island(self, task_queue):
        """Forms one island by processing one task from the queue."""
        if not task_queue:
            return None

        start_node = task_queue.popleft()

        if start_node in self.nodes_in_islands:
            print(f"Hub {start_node} is already in a formed island. Skipping.")
            return None

        print(f"Engine starting BFS from hub: {start_node}")
        island, q, visited = [], deque([start_node]), {start_node}

        while q and len(island) < self.max_island_size:
            node = q.popleft()
            island.append(node)
            for neighbor in self.graph.neighbors(node):
                if len(island) >= self.max_island_size:
                    break
                if neighbor not in visited and neighbor not in self.nodes_in_islands:
                    visited.add(neighbor)
                    q.append(neighbor)
        
        self.nodes_in_islands.update(island)
        print(f"  -> Formed island with {len(island)} nodes: {island}")
        return island

    def locate_islands(self):
        """
        Executes the full island location pipeline.
        """
        print("--- Starting Island Location ---")
        hubs = self._find_hubs()
        task_queue = self._generate_tasks(hubs)
        
        found_islands = []
        while task_queue:
            new_island = self._form_single_island(task_queue)
            if new_island:
                found_islands.append(new_island)
        
        print(f"\n--- Island Location Complete ---")
        print(f"Total islands found: {len(found_islands)}")
        return found_islands
