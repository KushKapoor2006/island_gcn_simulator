#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import dataclasses
import os
import argparse
import csv
import random
import re
from typing import List, Dict

# --- Stable CSV header definition ---
CSV_FIELDS = [
    'config_name','pe_sram_bytes','feature_dim','total_time_ns',
    'total_dram_traffic_bytes','fragmentation_penalty_time_ns',
    'fragmentation_penalty_bytes','islands_created','cut_edges',
    'pe_utilization_percent'
]

# --- Configuration ---
@dataclasses.dataclass
class SystemConfig:
    """Represents the physical hardware constraints of the accelerator."""
    pe_count: int
    pe_sram_bytes: int
    feature_dim: int
    feature_bytes: int = 4

    dram_bandwidth_gbps: float = 25.6  # gigabits / sec
    dram_latency_ns: int = 50
    cycles_per_op: int = 2
    cycle_time_ns: float = 0.5  # ns per cycle (2 GHz)

    @property
    def c_max(self) -> int:
        """Calculates the max number of nodes (c_max) that can fit in one PE's SRAM."""
        per_node_bytes = self.feature_dim * self.feature_bytes
        if per_node_bytes <= 0:
            raise ValueError("feature_dim and feature_bytes must be positive")
        c = self.pe_sram_bytes // per_node_bytes
        return int(max(1, c))

@dataclasses.dataclass
class Metrics:
    """Holds the simulation results for one run."""
    config_name: str
    pe_sram_bytes: int
    feature_dim: int
    total_time_ns: float = 0.0
    total_dram_traffic_bytes: int = 0
    fragmentation_penalty_time_ns: float = 0.0
    fragmentation_penalty_bytes: int = 0
    islands_created: int = 0
    cut_edges: int = 0
    pe_utilization_percent: float = 0.0
    # pe_utilization_log is a map: pe_id -> list of (start_ns, end_ns, label)
    pe_utilization_log: Dict[int, List] = dataclasses.field(default_factory=dict)
    # dram_log will be written into Simulator but kept here for convenience
    # Not serialized in main CSV row - separate CSVs are used for timeline logs.

# --- Hardware Component Models ---
class ProcessingElement:
    """Models a single PE in the Island Consumer."""
    def __init__(self, pe_id: int, sram_capacity_bytes: int):
        self.id = pe_id
        self.sram_capacity_bytes = sram_capacity_bytes
        self.finish_time_ns: float = 0.0
        self.total_busy_time_ns: float = 0.0

# --- Graph Generation ---
def generate_graph_with_controlled_communities(config: SystemConfig, seed: int = 42):
    """Generates a synthetic graph to trigger the c_max bottleneck."""
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    c_max = config.c_max
    sizes = [
        max(1, c_max // 2),
        max(1, c_max // 2),
        max(1, c_max - 10),
        max(1, c_max - 10),
        max(1, int(c_max * 1.5))
    ]
    probs = np.full((len(sizes), len(sizes)), 0.01)
    np.fill_diagonal(probs, 0.3)

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    print(f"Generated graph with {G.number_of_nodes()} nodes.")
    print(f"Ground truth community sizes: {sizes}")
    print(f"Hardware c_max (nodes per PE): {c_max}\n")
    return G

# --- Simulation Core Logic ---
class Simulator:
    """Orchestrates the simulation of an I-GCN architecture."""

    def __init__(self, name: str, config: SystemConfig, graph: nx.Graph):
        self.name = name
        self.config = config
        self.graph = graph
        self.metrics = Metrics(
            config_name=name,
            pe_sram_bytes=config.pe_sram_bytes,
            feature_dim=config.feature_dim,
            pe_utilization_log={i: [] for i in range(config.pe_count)}
        )
        # NOTE: DRAM model is conservative (fully serialized).
        # dram_next_free_ns enforces only one DMA transfer active at a time.
        self.dram_next_free_ns = 0.0

        # DRAM timeline log: list of dicts {start_ns, end_ns, bytes, node_count, reason}
        self.dram_log: List[Dict] = []

    # Chunked DMA model to allow compute overlap (streaming-friendly)
    def _model_dma_transfer_chunked(self, node_count: int, dma_start_ns: float, chunk_nodes: int = 16, reason: str = ""):
        """
        Chunked DMA model. Returns (compute_start_ns, dma_end_ns).
        compute_start_ns is when compute can first begin (after first chunk arrives).
        This function also accumulates total_dram_traffic_bytes and appends a dram_log entry.
        'reason' is a short string describing why the transfer happened (Island id / CutEdge etc).
        """
        if node_count == 0:
            return dma_start_ns, dma_start_ns

        per_node_bytes = self.config.feature_dim * self.config.feature_bytes
        total_bytes = node_count * per_node_bytes
        dram_bw_bytes_s = (self.config.dram_bandwidth_gbps * 1e9) / 8.0

        chunk_nodes = max(1, chunk_nodes)
        chunk_bytes = min(chunk_nodes * per_node_bytes, total_bytes)

        first_chunk_transfer_s = chunk_bytes / dram_bw_bytes_s
        first_chunk_ns = self.config.dram_latency_ns + first_chunk_transfer_s * 1e9

        # DMA can only start when DRAM is free (serialized conservative model)
        actual_start_ns = max(dma_start_ns, self.dram_next_free_ns)

        # compute can begin after first chunk arrives
        compute_start_ns = actual_start_ns + first_chunk_ns

        remaining_bytes = total_bytes - chunk_bytes
        remaining_ns = 0.0
        if remaining_bytes > 0:
            remaining_transfer_s = remaining_bytes / dram_bw_bytes_s
            remaining_ns = remaining_transfer_s * 1e9

        dma_end_ns = actual_start_ns + first_chunk_ns + remaining_ns

        # Update DRAM state and traffic. Note: caller should NOT add these bytes again.
        self.dram_next_free_ns = dma_end_ns
        self.metrics.total_dram_traffic_bytes += total_bytes

        # Record DRAM log entry
        self.dram_log.append({
            'start_ns': actual_start_ns,
            'end_ns': dma_end_ns,
            'bytes': int(total_bytes),
            'node_count': int(node_count),
            'reason': reason or ""
        })

        return compute_start_ns, dma_end_ns

    def _model_cut_edge_penalty(self, cut_edges: int, penalty_start_ns: float):
        """Models extra time and traffic for fetching features of cut edges.
        Uses the same chunked DMA model for each read; chunked DMA increments the
        total_dram_traffic_bytes internally (so do NOT double-add later)."""
        if cut_edges == 0:
            return penalty_start_ns, 0

        penalty_bytes = 0
        current_time = penalty_start_ns
        for i in range(cut_edges):
            # Model each cut edge as a small separate DMA transfer of a single node
            _, dma_end = self._model_dma_transfer_chunked(1, current_time, chunk_nodes=1, reason="CutEdge")
            current_time = dma_end
            penalty_bytes += self.config.feature_dim * self.config.feature_bytes

        return current_time, penalty_bytes

    def _model_pe_computation(self, island: List[int]) -> float:
        """Models the time a PE spends on aggregation/combination for an island."""
        subgraph = self.graph.subgraph(island)
        num_ops = (subgraph.number_of_nodes() + subgraph.number_of_edges()) * self.config.feature_dim
        compute_cycles = num_ops * self.config.cycles_per_op
        return compute_cycles * self.config.cycle_time_ns

    def _compute_cut_edges(self, islands: List[List[int]]) -> int:
        """Counts the number of edges between different islands."""
        node_to_island_map = {node: i for i, island in enumerate(islands) for node in island}
        cut_edges = sum(1 for u, v in self.graph.edges() if node_to_island_map.get(u) != node_to_island_map.get(v))
        return cut_edges

    def run(self):
        """Executes the full simulation pipeline."""
        print(f"--- Running Simulation: {self.name} ---")
        islands = self.locate_islands()
        self.metrics.islands_created = len(islands)
        self.metrics.cut_edges = self._compute_cut_edges(islands)
        print(f"Locator created {len(islands)} islands, resulting in {self.metrics.cut_edges} cut edges.")
        self.consume_islands(islands)
        # Base total time is when all PEs finished
        self.metrics.total_time_ns = max(pe.finish_time_ns for pe in self.pes) if self.pes else 0.0

        # Fragmentation penalty (Baseline only implements this)
        if hasattr(self, '_calculate_fragmentation_penalty'):
            self._calculate_fragmentation_penalty()

        total_busy_time = sum(pe.total_busy_time_ns for pe in self.pes)
        # Guard against division by zero
        if self.metrics.total_time_ns > 0:
            total_available_time = self.config.pe_count * self.metrics.total_time_ns
            self.metrics.pe_utilization_percent = (total_busy_time / total_available_time) * 100
        else:
            self.metrics.pe_utilization_percent = 0.0

        print(f"Simulation finished. Total time: {self.metrics.total_time_ns:.2f} ns")
        print(f"Total DRAM traffic: {self.metrics.total_dram_traffic_bytes / 1024:.2f} KB")
        print(f"PE Utilization: {self.metrics.pe_utilization_percent:.2f} %\n")

    def locate_islands(self) -> List[List[int]]:
        """Finds islands using the partitioning strategy (connected components + partition)."""
        found_islands = [list(c) for c in nx.connected_components(self.graph) if c]
        partitioned = []
        for component in found_islands:
            partitioned.extend(self._partition_component(component))
        # Return islands sorted by size descending
        return sorted(partitioned, key=len, reverse=True)

    # Abstract methods to be implemented by concrete simulators
    def consume_islands(self, islands: List[List[int]]):
        raise NotImplementedError

    def _partition_component(self, component_nodes: List[int]) -> List[List[int]]:
        raise NotImplementedError

# --- Baseline I-GCN Implementation ---
class BaselineSimulator(Simulator):
    """Simulates the original I-GCN with a rigid c_max."""
    def __init__(self, config: SystemConfig, graph: nx.Graph):
        super().__init__("Baseline I-GCN (Rigid c_max)", config, graph)
        self.pes = [ProcessingElement(i, config.pe_sram_bytes) for i in range(config.pe_count)]

    def _partition_component(self, component_nodes: List[int]) -> List[List[int]]:
        c_max = self.config.c_max
        if len(component_nodes) <= c_max:
            return [component_nodes]
        print(f"Baseline Locator: Splitting large community of size {len(component_nodes)} into chunks of {c_max}.")
        return [component_nodes[i:i + c_max] for i in range(0, len(component_nodes), c_max)]

    def _calculate_fragmentation_penalty(self):
        """Calculates and adds the cut edge penalty *after* the main run.
        Note: _model_cut_edge_penalty already increments total_dram_traffic_bytes for each read,
        so we must NOT double-add penalty bytes to metrics.total_dram_traffic_bytes here."""
        print(f"Calculating fragmentation penalty for {self.metrics.cut_edges} cut edges...")
        penalty_start_time = self.metrics.total_time_ns
        penalty_end_time, penalty_bytes = self._model_cut_edge_penalty(self.metrics.cut_edges, penalty_start_time)
        self.metrics.fragmentation_penalty_time_ns = penalty_end_time - penalty_start_time
        self.metrics.fragmentation_penalty_bytes = penalty_bytes
        # Do NOT add penalty_bytes again to total_dram_traffic_bytes; already accounted
        self.metrics.total_time_ns = penalty_end_time
        print(f"Fragmentation added {self.metrics.fragmentation_penalty_time_ns:.2f} ns and {penalty_bytes/1024:.2f} KB")

    def consume_islands(self, islands: List[List[int]]):
        for island in islands:
            self.pes.sort(key=lambda pe: pe.finish_time_ns)
            target_pe = self.pes[0]

            pe_free_time = target_pe.finish_time_ns
            compute_time = self._model_pe_computation(island)

            # Chunked DMA: compute can start after the first chunk; DMA continues streaming
            compute_start_time, dma_end_time = self._model_dma_transfer_chunked(
                len(island), pe_free_time, chunk_nodes=16, reason=f"Island({len(island)}) PE{target_pe.id}"
            )

            # The PE is busy until compute finishes and DMA finishes (conservative)
            end_time = max(dma_end_time, compute_start_time + compute_time)
            target_pe.finish_time_ns = end_time

            task_duration = end_time - pe_free_time
            target_pe.total_busy_time_ns += task_duration

            # record PE activity for gantt
            self.metrics.pe_utilization_log[target_pe.id].append((pe_free_time, end_time, f"Island ({len(island)})"))

# --- Enhanced I-GCN Implementation ---
class EnhancedSimulator(Simulator):
    """Simulates the enhanced I-GCN with adaptive granularity."""
    def __init__(self, config: SystemConfig, graph: nx.Graph):
        super().__init__("Enhanced I-GCN (Adaptive)", config, graph)
        self.pes = [ProcessingElement(i, config.pe_sram_bytes) for i in range(config.pe_count)]
        # Handle odd pe_count: only valid pairs added
        self.pe_pairs = [(i, i + 1) for i in range(0, config.pe_count - 1, 2)]

    def _partition_component(self, component_nodes: List[int]) -> List[List[int]]:
        c_max = self.config.c_max
        max_merged_size = c_max * 2 if self.pe_pairs else c_max
        if len(component_nodes) > max_merged_size:
            print(f"Enhanced Locator: Community {len(component_nodes)} too large, splitting.")
            return [component_nodes[i:i + max_merged_size] for i in range(0, len(component_nodes), max_merged_size)]
        return [component_nodes]

    def consume_islands(self, islands: List[List[int]]):
        for island in islands:
            island_size = len(island)

            if island_size <= self.config.c_max:
                # Single PE
                self.pes.sort(key=lambda pe: pe.finish_time_ns)
                target_pe = self.pes[0]
                pe_free_time = target_pe.finish_time_ns
                compute_time = self._model_pe_computation(island)
                compute_start, dma_end = self._model_dma_transfer_chunked(
                    island_size, pe_free_time, chunk_nodes=16, reason=f"Island({island_size}) PE{target_pe.id}"
                )
                end_time = max(dma_end, compute_start + compute_time)
                target_pe.finish_time_ns = end_time
                target_pe.total_busy_time_ns += end_time - pe_free_time
                self.metrics.pe_utilization_log[target_pe.id].append((pe_free_time, end_time, f"Island ({island_size})"))
            elif self.pe_pairs:
                # Merged pair of PEs
                self.pe_pairs.sort(key=lambda p: max(self.pes[p[0]].finish_time_ns, self.pes[p[1]].finish_time_ns))
                pe_id1, pe_id2 = self.pe_pairs[0]

                pair_free_time = max(self.pes[pe_id1].finish_time_ns, self.pes[pe_id2].finish_time_ns)
                compute_time = self._model_pe_computation(island)
                compute_start, dma_end = self._model_dma_transfer_chunked(
                    island_size, pair_free_time, chunk_nodes=16, reason=f"MERGED Island({island_size}) PEs({pe_id1},{pe_id2})"
                )
                end_time = max(dma_end, compute_start + compute_time)

                self.pes[pe_id1].finish_time_ns = end_time
                self.pes[pe_id2].finish_time_ns = end_time
                self.pes[pe_id1].total_busy_time_ns += end_time - pair_free_time
                self.pes[pe_id2].total_busy_time_ns += end_time - pair_free_time
                self.metrics.pe_utilization_log[pe_id1].append((pair_free_time, end_time, f"MERGED ({island_size})"))
                self.metrics.pe_utilization_log[pe_id2].append((pair_free_time, end_time, f"MERGED ({island_size})"))
            else:
                # No pair available: fallback to single PE processing
                self.pes.sort(key=lambda pe: pe.finish_time_ns)
                target_pe = self.pes[0]
                pe_free_time = target_pe.finish_time_ns
                compute_time = self._model_pe_computation(island)
                compute_start, dma_end = self._model_dma_transfer_chunked(
                    island_size, pe_free_time, chunk_nodes=16, reason=f"Island({island_size}) PE{target_pe.id}"
                )
                end_time = max(dma_end, compute_start + compute_time)
                target_pe.finish_time_ns = end_time
                target_pe.total_busy_time_ns += end_time - pe_free_time
                self.metrics.pe_utilization_log[target_pe.id].append((pe_free_time, end_time, f"Island ({island_size})"))

# --- Visualization & Reporting ---
def plot_results(baseline_metrics: Metrics, enhanced_metrics: Metrics, config: SystemConfig, save_dir: str):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'I-GCN Performance: Baseline vs. Enhanced\n(SRAM: {config.pe_sram_bytes}B, Features: {config.feature_dim}D)', fontsize=16)

    # 1. Total Latency
    axs[0, 0].bar(['Baseline', 'Enhanced'], [baseline_metrics.total_time_ns, enhanced_metrics.total_time_ns], color=['#ff6347', '#4682b4'])
    axs[0, 0].set_title('Total Simulation Time (Latency)')
    axs[0, 0].set_ylabel('Time (nanoseconds)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Total DRAM Traffic
    axs[0, 1].bar(['Baseline', 'Enhanced'], [baseline_metrics.total_dram_traffic_bytes / 1024, enhanced_metrics.total_dram_traffic_bytes / 1024], color=['#ff6347', '#4682b4'])
    axs[0, 1].set_title('Total Off-Chip DRAM Traffic (incl. Penalty)')
    axs[0, 1].set_ylabel('Data Transferred (KB)')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Number of Islands Created
    axs[1, 0].bar(['Baseline', 'Enhanced'], [baseline_metrics.islands_created, enhanced_metrics.islands_created], color=['#ff6347', '#4682b4'])
    axs[1, 0].set_title('Number of Islands Created (Fragmentation)')
    axs[1, 0].set_ylabel('Island Count')
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Cut Edges
    axs[1, 1].bar(['Baseline', 'Enhanced'], [baseline_metrics.cut_edges, enhanced_metrics.cut_edges], color=['#ff6347', '#4682b4'])
    axs[1, 1].set_title('Cut Edges (Inter-Island Connections)')
    axs[1, 1].set_ylabel('Edge Count')
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'simulation_results.png')
    plt.savefig(save_path)
    print(f"\nPlots saved to: {os.path.abspath(save_path)}")

def write_csv_results(metrics: Metrics, csv_path: str):
    """Appends a row of metrics to a CSV file using a stable column order."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        metrics_dict = {k: getattr(metrics, k, None) for k in CSV_FIELDS}
        writer.writerow(metrics_dict)

def sanitize_filename(s: str) -> str:
    """Produce a filesystem-safe short name."""
    s2 = re.sub(r'\s+', '_', s)
    s2 = re.sub(r'[^A-Za-z0-9_\-\.]', '', s2)
    return s2

def write_gantt_logs(sim: Simulator, out_dir: str, seed: int = None):
    """
    Writes two CSV logs that gantt.py can consume:
      - <sim_name>_pe_timeline.csv : pe_id,start_ns,end_ns,label
      - <sim_name>_dram_timeline.csv : start_ns,end_ns,bytes,node_count,reason
    """
    os.makedirs(out_dir, exist_ok=True)
    name_base = sanitize_filename(sim.name)
    if seed is not None:
        name_base = f"{name_base}_seed{seed}"

    pe_csv = os.path.join(out_dir, f"{name_base}_pe_timeline.csv")
    dram_csv = os.path.join(out_dir, f"{name_base}_dram_timeline.csv")

    # write PE utilization log
    with open(pe_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pe_id', 'start_ns', 'end_ns', 'label'])
        for pe_id, entries in sim.metrics.pe_utilization_log.items():
            for entry in entries:
                # entry assumed (start_ns, end_ns, label)
                start_ns, end_ns, label = entry
                writer.writerow([pe_id, float(start_ns), float(end_ns), label])

    # write DRAM timeline log
    with open(dram_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_ns', 'end_ns', 'bytes', 'node_count', 'reason'])
        for entry in sim.dram_log:
            writer.writerow([float(entry['start_ns']), float(entry['end_ns']), int(entry['bytes']), int(entry['node_count']), entry.get('reason', '')])

    print(f"Gantt logs written:\n  PE timeline: {os.path.abspath(pe_csv)}\n  DRAM timeline: {os.path.abspath(dram_csv)}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Simulate I-GCN c_max Bottleneck")
    parser.add_argument('--sram_kb', type=int, default=8, help="On-chip SRAM size per PE in KB")
    parser.add_argument('--feature_dim', type=int, default=128, help="Dimension of node feature vectors")
    parser.add_argument('--pe_count', type=int, default=4, help="Number of Processing Elements")
    parser.add_argument('--output_dir', type=str, default="simulation_results", help="Directory to save plots and CSV")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for graph generation")
    args = parser.parse_args()

    config = SystemConfig(
        pe_count=args.pe_count,
        pe_sram_bytes=args.sram_kb * 1024,
        feature_dim=args.feature_dim
    )

    # Generate graph (seeded for reproducibility)
    graph = generate_graph_with_controlled_communities(config, seed=args.seed)

    # Run baseline
    baseline_sim = BaselineSimulator(config, graph)
    baseline_sim.run()

    # Run enhanced
    enhanced_sim = EnhancedSimulator(config, graph)
    enhanced_sim.run()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Produce plots and CSV
    plot_results(baseline_sim.metrics, enhanced_sim.metrics, config, args.output_dir)
    csv_path = os.path.join(args.output_dir, 'simulation_results.csv')
    write_csv_results(baseline_sim.metrics, csv_path)
    write_csv_results(enhanced_sim.metrics, csv_path)

    # Write the Gantt/DRAM logs (one pair per simulator)
    write_gantt_logs(baseline_sim, args.output_dir, seed=args.seed)
    write_gantt_logs(enhanced_sim, args.output_dir, seed=args.seed)

    print(f"Results appended to: {os.path.abspath(csv_path)}")

if __name__ == '__main__':
    main()
