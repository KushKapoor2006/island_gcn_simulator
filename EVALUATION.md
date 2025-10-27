# EVALUATION — I‑GCN c_max Mitigation Study

This document explains the evaluation methodology, the simulation and RTL parameters used, the metrics collected, representative results (Baseline vs Enhanced), assumptions made for tractability, and other technical details necessary to reproduce and interpret the study.

---

## 1. Goals of the Evaluation

* Quantify the performance and bandwidth impact of **rigid `c_max` partitioning** (Baseline) versus **adaptive pairwise PE merging** (Enhanced).
* Provide time‑line (Gantt) visualizations of per‑PE utilization and DRAM transfers to explain *why* the metrics change.
* Keep the simulation realistic enough to capture the dominant effects (on‑chip capacity, DRAM transfers, compute cost) while remaining simple and fast to run for many configurations.

---

## 2. High‑level approach / pipeline

1. **Graph generation**: produce synthetic graphs (stochastic block model, SBM) whose community sizes intentionally stress `c_max`.
2. **Island locator**: find connected components and apply a locator policy that partitions components into islands.

   * Baseline: rigidly split a component into chunks of size `<= c_max`.
   * Enhanced: keep a component whole if it fits in `2*c_max` (merge two PEs), otherwise split into `<= 2*c_max` chunks.
3. **Island consumer**: schedule islands to PEs using a greedy earliest-free PE model. Model DMA (chunked streaming) and PE compute time. Record timelines.
4. **Postprocessing**: compute aggregate metrics (latency, DRAM bytes, islands created, cut edges), generate CSV timelines, draw Gantt charts.

---

## 3. Script parameters (configurable)

All parameters are exposed in `simulation/main_sim.py` via a `SystemConfig` dataclass or CLI flags. Key parameters below (names as in the simulator):

| Parameter              |              Variable | Typical default | Effect on simulation                                                     |
| ---------------------- | --------------------: | --------------: | ------------------------------------------------------------------------ |
| Number of PEs          |            `pe_count` |               4 | More PEs increases parallelism; enables more merging pairs for Enhanced. |
| On‑chip SRAM per PE    |       `pe_sram_bytes` |     8192 (8 KB) | Determines `c_max`. Larger reduces fragmentation.                        |
| Feature dim            |         `feature_dim` |             128 | Affects per‑node bytes and compute cost (compute ∝ `feature_dim`).       |
| Feature bytes          |       `feature_bytes` |               4 | Bytes per feature element (typically float32).                           |
| DRAM bandwidth (Gb/s)  | `dram_bandwidth_gbps` |            25.6 | Determines transfer time; higher bandwidth reduces DMA overhead.         |
| DRAM latency (ns)      |     `dram_latency_ns` |              50 | Fixed per transfer startup latency.                                      |
| cycles per op          |       `cycles_per_op` |               2 | Simplified compute intensity model.                                      |
| cycle time (ns)        |       `cycle_time_ns` |             0.5 | Clock period, converts cycles → ns.                                      |
| DMA chunk size (nodes) |         `chunk_nodes` |              16 | Streaming granularity: the compute can begin after first chunk arrives.  |
| SBM seed               |                `seed` |              42 | Controls reproducible graph generation.                                  |

**Derived**: `c_max = floor(pe_sram_bytes / (feature_dim * feature_bytes))`.

---

## 4. Graph & island parameters

* **Graph model**: Stochastic Block Model (SBM) created with `networkx.stochastic_block_model`. The example generator uses five communities of sizes chosen relative to `c_max` (e.g. `c_max/2`, `c_max - 10`, `1.5*c_max`) and with high intra‑community connection probability (0.3) and low inter‑community (0.01). This creates dense clusters and sparse cross edges — ideal for testing locality.

* **Island locator**:

  * First, connected components are enumerated (this approximates a BFS‑based island discovery stage).
  * Baseline: component split into chunks of `<= c_max` in simple contiguous slices (arbitrary split to model worst‑case fragmentation).
  * Enhanced: if component size `<= 2*c_max` then it is kept whole and scheduled to a merged pair; else component is split to chunks of `<= 2*c_max`.

* **Cut edges**: computed by counting edges whose endpoints map to different islands. Each cut edge is modeled as a penalty DRAM fetch (one node feature read) during a fragmentation penalty stage.

---

## 5. PE / DMA / Compute models (implementation details)

### Compute model

* For a given island, the simulator computes `num_ops = (#nodes + #internal_edges) * feature_dim`.
* Compute cycles = `num_ops * cycles_per_op`.
* Compute time (ns) = `compute_cycles * cycle_time_ns`.

This is a simplified proportional model capturing both node and neighbor aggregation costs.

### DMA model (chunked streaming)

* Transfers are modeled in bytes: `bytes = nodes * feature_dim * feature_bytes`.
* Bandwidth uses `dram_bandwidth_gbps / 8` to convert to bytes/s; transfer time = bytes / bw.
* Latency overhead `dram_latency_ns` is added once per burst.
* **Chunked model**: we model a first chunk (default `chunk_nodes=16`) whose arrival allows compute to start (streaming overlap). Remaining bytes stream while compute continues; PE must still wait until compute finishes and DMA finishes.
* **DRAM contention**: the simulator conservatively serializes DRAM transfers using a `dram_next_free_ns` scalar. This models a worst‑case single‑controller worst‑case serialization and makes fragmentation costs visible.

### Fragmentation penalty

* After the main run, Baseline invokes `_calculate_fragmentation_penalty()` which issues one small DMA read per cut edge (serialized via the same DRAM model). The penalty time and bytes are added to total time/traffic and to `fragmentation_penalty_*` metrics.

**Important**: cut edge penalty uses the same chunked DMA call; that call updates `total_dram_traffic_bytes` internally. The code avoids double counting by not adding penalty bytes again in the penalty calculator.

---

## 6. Metrics collected (and their interpretation)

* `total_time_ns` — end‑to‑end simulation time in ns (includes compute and serialized DMA and fragmentation penalty). Lower is better.
* `total_dram_traffic_bytes` — total bytes transferred (including penalty reads). Lower is better.
* `islands_created` — number of islands output by the locator. Lower → less fragmentation.
* `cut_edges` — number of graph edges crossing islands: correlates with fragmentation and additional DRAM reads.
* `pe_utilization_percent` — aggregated busy time / (PE_count * total_time). Higher → better PE usage.
* `pe_utilization_log` — per‑PE timeline entries for Gantt visualization.
* `dram_timeline` — ordered transfer entries (start, end, bytes, node_count, reason) for Gantt visualization.

**Quantitative summary (example run)**

* Example defaults produced a representative improvement of **~1.5×** endpoint speedup (Baseline / Enhanced) on the synthetic SBM used.
* DRAM traffic dropped by roughly **2×** for the Enhanced run in the shown example, and island count decreased (less fragmentation).

Exact numbers are saved in `simulation_results/simulation_results.csv` for reproducibility.

---

## 7. Baseline vs Enhanced — why the gains occur (technical intuition)

* **Baseline** splits natural communities arbitrarily at `c_max`, producing many islands. These splits create *cut edges*; every cut edge requires re‑fetching neighbor features from DRAM when updating nodes across islands. Because the simulator models DRAM serially, these small random reads are expensive and cause large wall‑clock penalties.

* **Enhanced** leaves dense communities intact when they fit in `2*c_max` by temporarily pairing PEs. This reduces both the number of islands and cut edges, thereby reducing the number of small serialized DRAM reads and improving data locality. The cost is that merged PEs both block (reduce parallelism) while handling a single large island; however, for dense communities the reduced DRAM overhead typically outweighs this loss.

* The chunked DMA model allows compute to overlap with streaming, so the largest benefit is reducing *many* small latency‑bound DRAM reads (cut edges) rather than raw bandwidth‑bound bulk transfers.

---

## 8. Assumptions & simplifications (explicit)

We make several simplifying assumptions to keep the simulator interpretable and fast:

1. **Serialized DRAM**: `dram_next_free_ns` serializes DMA transfers. This is intentionally conservative; real DRAM has bank/channel parallelism which can reduce the fragmentation penalty. We chose conservative DRAM to make fragmentation effects visible and worst‑case.

2. **No multi‑writer coherency network**: The simulator assumes each island's updates are independent for the sake of timing and DMA modeling. The heavier problem of write‑back coherency for shared hub nodes across concurrently processed islands is not modeled (out of scope). Real hardware must implement reductions/atomics or coordinated serialization.

3. **Simplified compute model**: compute cycles scale linearly with `(nodes + edges) * feature_dim`, which is a coarse but effective proxy for aggregation compute.

4. **PE merging limited to pairs**: Enhanced implements only 2‑way merging. Larger merging (k‑way) or flexible memory pooling is not modeled.

5. **SBM synthetic graphs**: We use synthetic SBM graphs to control community sizes and isolate the c_max effect. Results on real graphs (OGB/SNAP) may vary.

6. **Island partitioning strategy for Baseline**: We intentionally use a simplistic contiguous chunking when splitting a component to model a pessimistic fragmentation case. Smarter software partitioners (METIS, KaHIP) will produce fewer cut edges — our Baseline therefore represents a worst‑case hardware policy.

---

## 9. Reproducibility checklist

To reproduce the evaluations and figures shown in the `simulation_results` folder:

1. Install Python dependencies: `pip install -r requirements.txt`.
2. Run a single simulation: `python simulation/main_sim.py --output_dir simulation_results --seed 42`.
3. Draw Gantt charts: `python simulation/gantt.py --input_dir simulation_results --out_dir simulation_results`.
4. Run the sweep (optional): `python run_sweep.py`.
5. Run RTL TB: `cd hardware && iverilog -g2012 -o igcn_sim.vvp igcn_accelerator.sv tb_igcn_accelerator.sv && vvp igcn_sim.vvp`.

---

## 10. Additional notes & recommended next steps

* **DRAM realism**: Add a more realistic DRAM model with bank/channel parallelism and queuing to better quantify fragmentation in modern multi‑channel systems.
* **Reduction/Coherency**: Design and simulate a write‑back reduction mechanism for shared hub updates (options: atomic accumulation unit, producer–consumer queues with accumulation, or a tree‑reduction network). This addresses the other major bottleneck noted in the I‑GCN paper.
* **Partitioner comparison**: Replace the baseline splitting with METIS/Kernighan‑Lin style partitions and compare hardware‑aware partitioning vs our fast online locator.
* **Real datasets**: Run on OGB / SNAP graphs and measure end‑to‑end GNN training/inference time including forward/backward passes and (optionally) model weights.

---

## 11. Where to look in the repo for data & scripts

* `simulation/main_sim.py` — simulator and CLI options (primary file to modify / extend).
* `simulation/gantt.py` — plotting and labeling heuristics for Gantt charts.
* `simulation_results/` — produced CSVs and PNGs for each run.
* `hardware/igcn_accelerator.sv` & `hardware/tb_igcn_accelerator.sv` — RTL model and testbench.

---
