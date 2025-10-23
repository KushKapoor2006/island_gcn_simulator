# I-GCN Accelerator (simulation + RTL model)

Compact repo summary and quick-run instructions for the I-GCN accelerator project: a simulation of the I-GCN partitioning/PE scheduling idea, a small performance sweep, and a behavioral RTL model (SystemVerilog) with testbench for cycle-accurate validation.

**Contents**
- `simulation/`
  - `main_sim.py` — event-driven simulator of baseline vs enhanced partitioning (c_max bottleneck + adaptive merging).
  - `gantt.py` — visualization of PE timelines and DRAM transfers (produces PNGs).
  - helper scripts: `run_small.sh`, `run_sweep.py`.
- `simulation_results/` — generated CSVs, PNGs and sweep outputs (example results included).
- `hardware/`
  - `igcn_accelerator.sv` — behavioral SystemVerilog accelerator model (Icarus-compatible, -g2012).
  - `tb_igcn_accelerator.sv` — testbench (drives workload scenarios used in evaluation).
- `requirements.txt` — python packages used for simulation/plots.

---

### Run the simulation (single config)
```bash
# run the small simulation that produces CSV timelines and high-level metrics
./run_small.sh
# or directly:
python simulation/main_sim.py --output_dir simulation_results

### Prerequisites
- Python 3.8+ with packages:
