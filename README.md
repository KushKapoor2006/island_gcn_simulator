# I-GCN Accelerator — Simulation + RTL Prototype (c_max Bottleneck Mitigation)

## TL;DR
### island_gcn_simulator (iGCN) — Island Allocation Strategies (Verilog)
- **What:** Accelerator verification harness and controller for islanded GCN execution strategies; compares baseline vs enhanced allocation policies.
- **Key result (From RTL TB trace):** Baseline run cycles = **403**; Enhanced run cycles = **356** → **1.13× speedup** for the enhanced island allocation policy in the TB scenario.  
- **Notes:** The TB trace shows per-PE island allocations, penalties and latched allocations; use the testbench (`tb_igcn_accelerator.sv`) for trace-driven experiments and strategy validation.



**Short description — what this project achieves**

This repository implements, evaluates, and prototypes a practical mitigation to a real architectural bottleneck found in the I‑GCN accelerator design: the **static granularity mismatch** (commonly referred to as `c_max`) between fixed on‑chip scratchpad capacity and natural community sizes in graphs. The core contribution is a hardware-friendly mitigation — **adaptive PE merging (pairwise)** — together with a quantitative simulation study and a small SystemVerilog prototype demonstrating the idea at cycle granularity.

In plain terms: the original I‑GCN approach rigidly splits communities larger than `c_max`, causing fragmentation, extra DRAM traffic, and poor utilization. This project shows that allowing two PEs to temporarily *merge* their scratchpad capacity (supporting islands up to `2*c_max`) significantly reduces fragmentation and off‑chip traffic, improving end‑to‑end latency for many realistic workloads.

---

## Repository layout

```
├── hardware/
│   ├── igcn_accelerator.sv         # Behavioral SystemVerilog accelerator (DUT)
│   └── tb_igcn_accelerator.sv      # Testbench (Icarus Verilog friendly)
├── simulation/
│   ├── main_sim.py                 # Event-driven simulator (Baseline vs Enhanced)
│   ├── gantt.py                    # Gantt chart generator for PE & DRAM timelines
├── simulation_results/             # Example outputs (CSV + PNG images)
├── requirements.txt                # Python dependencies
└── README.md                       # <-- this file
└── EVALUATION.md                   # Parameters and metrics
└── run_small.sh                    # Helper Script - Executable (Run with chmod +x)
└── run_sweep.py                    # Aggregated speedup plot (mean ± std)
```

---

## Key claim (what I solved)

**I demonstrate and quantify a practical solution to the `c_max` (static granularity) bottleneck described (but not solved) in the I‑GCN paper.**

Specifically:

* I identify how rigid `c_max` partitioning increases cut edges and DRAM reads, and how that harms performance in realistic community‑structured graphs.
* I implement **PE merging** (pair-based adaptive granularity) as a low-complexity mitigation that keeps natural communities intact when they fit within `2*c_max`.
* I quantify the benefit with a simulation pipeline (latency, DRAM traffic, islands created, cut edges) and visualize per‑PE utilization & DRAM behavior via Gantt charts.
* I provide a small behavioral RTL model + testbench (Icarus Verilog friendly) to validate handshake/allocation behavior and cycle counts for the allocation policy.

This work therefore closes an important gap between the I‑GCN high‑level architectural description (parallel PE array + pipeline) and a concrete handling of the `c_max` fragmentation problem — showing a practical, implementable improvement with measurable gains.

---

## What is included

1. **Simulator (Python)** — `simulation/main_sim.py`

   * Models: PE array, on‑chip SRAM `c_max`, chunked DMA streaming, conservative DRAM serialization, compute model proportional to `(nodes + edges) * feature_dim`, and fragmentation penalty for cut edges.
   * Two strategies implemented:

     * **Baseline** — rigid partitioning into chunks sized `<= c_max` (max fragmentation)
     * **Enhanced** — adaptive PE merging (allow pair merges up to `2*c_max`), keeping natural communities intact when possible
   * Produces per‑PE and DRAM timeline CSV files + aggregate metrics (latency, DRAM bytes, islands, cut edges).

2. **Visualization** — `simulation/gantt.py`

   * Draws a two‑row Gantt per simulation run: top row is PE utilization (per PE), bottom row is DRAM transfer timeline (color = bytes, log scale).
   * `simulation/main_sim.py` already saves summary bar plots comparing Baseline vs Enhanced (latency, DRAM, islands, cut edges).

3. **Parameter sweep** — `run_sweep.py`

   * Sweeps SRAM sizes / feature dims / PE counts and aggregates mean speedup ± std. Example output: `sweep_speedup_plot.png`.

4. **Hardware (RTL)** — `hardware/igcn_accelerator.sv` and `hardware/tb_igcn_accelerator.sv`

   * Behavioral SystemVerilog model of N PEs, a controller that latches island requests, allocates single PEs or contiguous PE pairs (for merging), and produces a registered `island_accepted` handshake.
   * The TB partitions islands for Baseline/Enhanced strategies and measures cycle‑level finish time.
   * Intended for Icarus Verilog with `-g2012`.

5. **Simulation results** — example PNGs and CSVs in `simulation_results/` demonstrating the improvements and Gantt charts.

---

## How to run — quick guide

### 1) Install Python deps

```bash
pip install -r requirements.txt
```

`requirements.txt` contains (example):

```
networkx
numpy
matplotlib
pandas
```

### 2) Run the simulator (single experiment)

```bash
# from repo root
python simulation/main_sim.py --output_dir simulation_results --seed 42
```

This produces CSV timelines and PNG summary plots under `simulation_results/` (see filenames printed by the script).

### 3) Create Gantt charts (if you already have CSVs)

```bash
python simulation/gantt.py --input_dir simulation_results --out_dir simulation_results
```

Only PNGs are written by the provided script; adjust the `--show` flag to display interactively.

### 4) Run the sweep

```bash
python run_sweep.py
```

This runs multiple configs and produces an aggregated speedup plot (mean ± std) placed under `simulation_results/sweep_results/`.

### 5) RTL simulation with Icarus Verilog

```bash
cd hardware
iverilog -g2012 -o igcn_sim.vvp igcn_accelerator.sv tb_igcn_accelerator.sv
vvp igcn_sim.vvp
# optional VCD viewing
gtkwave igcn_tb.vcd
```

**Note:** If your Icarus build lacks certain SystemVerilog niceties, you may need to simplify the SV code (the included RTL was adapted for broad -g2012 compatibility). If you see syntax errors referencing `foreach` or `break`, use a modern Icarus version or adjust those constructs to simple loops.

---

## Simulation configuration & important parameters

`main_sim.py` exposes the following important parameters (CLI flags or in the `SystemConfig` dataclass):

* `pe_count` — number of PEs in the array (parallel workers)
* `pe_sram_bytes` — SRAM capacity per PE in bytes (controls `c_max`)
* `feature_dim` — dimension of node feature vectors
* `feature_bytes` — bytes per feature (defaults to 4 for float32)
* `dram_bandwidth_gbps` — DRAM bandwidth (gigabits/sec)
* `dram_latency_ns` — DRAM latency in ns
* `cycles_per_op`, `cycle_time_ns` — simplifed compute model parameters
* `seed` — RNG seed for stochastic block model (SBM) graph generation used in examples

**How `c_max` is computed**

```
c_max = pe_sram_bytes // (feature_dim * feature_bytes)
```

The simulator generates an SBM graph with community sizes selected to stress `c_max` (some communities intentionally slightly larger than `c_max`). The baseline locator rigidly splits any community > `c_max`; the enhanced locator keeps a community intact if it fits in a merged pair (`<= 2*c_max`).

---

## Evaluation methodology & metrics (what was measured)

For each run we capture/compute:

* **Total time (ns)** — wallclock-like end‑to‑end completion time (PE compute + DMA ordering + fragmentation penalty).
* **Total DRAM traffic (bytes)** — bytes moved to/from DRAM (chunked transfers and cut edge penalty reads).
* **Islands created** — fragmentation count (how many islands the locator produced from original communities).
* **Cut edges** — number of graph edges that cross islands (each creates an extra DRAM read modeled as a penalty read).
* **PE utilization logs** — per‑PE task start/end labels for Gantt charts.
* **DRAM timeline** — per‑transfer start/end and size, used to visualize contention & serialization.

**Procedure**

1. Generate a synthetic SBM graph with seeded randomness for reproducibility.
2. Run Baseline: locate islands, partition with rigid `c_max`, dispatch islands to PEs, record timelines.
3. Run Enhanced: same graph, adaptive locator that tries to avoid splitting dense communities (merge two PEs when necessary), record timelines.
4. Derive aggregate metrics and compare (latency, bytes, islands, cut edges). Visualize with bar charts and Gantt charts.

**Key result summary (representative)**

* Example run (default parameters in repo): **~1.55× mean speedup** (Baseline / Enhanced) in end‑to‑end time for the tested synthetic graph set. DRAM traffic dropped by roughly **2×** in the example run, and the number of islands (fragmentation) also decreased.

See `simulation_results/` for the exact CSV values and `sweep_results/` for aggregated statistics across multiple SRAM/PE configurations.

---

## Hardware notes and implications

* The mitigation (pairwise PE merging) is low complexity: it requires the controller to be able to allocate two contiguous PEs to a single island, and a small amount of coordination to treat them as a single larger scratchpad for the duration of the island processing.
* This avoids the much harder problems (not solved here) of multi‑PE write‑back coherency for shared hubs and global atomic reductions. The project explicitly isolates `c_max` fragmentation and treats the reduction/coherency problem as **orthogonal** future work.
* The RTL testbench demonstrates allocation handshake correctness and cycle counts; it is not a full physical implementation and lacks memory coherence or actual feature vector movement logic (that would be part of a full synthesizable design).

---

## Limitations & future work

* **Reduction & write‑back coherency**: This project does not implement a full reduction/atomics mechanism for nodes shared across islands processed in parallel. Solving that requires careful design (reduction network, atomic accumulators, or software‑coordinated serializations) and is on the roadmap.
* **DRAM realism**: The simulator uses a conservative serialized DRAM model to highlight fragmentation costs; real DRAM controllers have bank and channel parallelism — results will vary with a more realistic DRAM microarchitecture.
* **PE merging granularity**: We only model pair merging (2×). Extending to k‑way merging or dynamic on‑chip memory virtualization is possible future work.
* **Real datasets**: Current experiments use SBM synthetic graphs. The next step is to run the simulator on OGB / SNAP graphs and report end‑to‑end GNN accuracy/perf tradeoffs.

---

## Files of interest (quick)

* `simulation/main_sim.py` — the simulation engine and configuration.
* `simulation/gantt.py` — improved labeling heuristics for clarity in Gantt charts.
* `simulation/run_sweep.py` — runs multiple runtime experiments and aggregates speedups.
* `hardware/igcn_accelerator.sv` — SV accelerator (DUT) for allocation handshake and pair merging behavior.
* `hardware/tb_igcn_accelerator.sv` — testbench that partitions islands and measures cycles.

---

## Reproducible example (commands)

```bash
# 1. install deps
pip install -r requirements.txt

# 2. run a single simulation and generate plots
python simulation/main_sim.py --output_dir simulation_results --seed 42
python simulation/gantt.py --input_dir simulation_results --out_dir simulation_results

# 3. run RTL TB (requires iverilog, vvp)
cd hardware
iverilog -g2012 -o igcn_sim.vvp igcn_accelerator.sv tb_igcn_accelerator.sv
vvp igcn_sim.vvp
# optional: gtkwave igcn_tb.vcd
```

---

## Contact / Author

Project owner: *Kush Kapoor* — as part of the research group of Prof. 


---
