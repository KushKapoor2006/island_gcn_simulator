#!/usr/bin/env python3
"""
run_sweep.py

Run small parameter sweeps for the I-GCN simulator, collect baseline+enhanced metrics,
compute speedups and DRAM reductions, and write per-run + summary CSVs and a plot.

**IMPORTANT**: Set BASE_OUTPUT_DIR below to a directory where this script may create per-run
subdirectories (or pass via --base_outdir CLI). This script will create `results/<...>` folders.
"""

import os
import subprocess
import csv
import time
import argparse
from statistics import mean, pstdev
import math
import json
import matplotlib.pyplot as plt

# ------------------ PLACEHOLDER ------------------
# Put your base output folder here (where per-run folders will be created).
# Example: BASE_OUTPUT_DIR = "/home/kk/Desktop/Research/work/I-GCN/results"
BASE_OUTPUT_DIR = "simulation_results/sweep_results/"
# -------------------------------------------------

# Default param sets to sweep (you can edit these or pass via CLI)
DEFAULT_PARAM_SETS = [
    # (sram_kb, feature_dim, pe_count)
    (8, 64, 4),
    (16, 128, 4),
    (32, 128, 8),
]

# Output filenames (written into BASE_OUTPUT_DIR)
DETAILED_CSV = "sweep_runs.csv"
SUMMARY_CSV = "sweep_summary.csv"
PLOT_PNG = "sweep_speedup_plot.png"

def run_once(sram_kb, feature_dim, pe_count, seed, per_run_outdir, python_bin="python3"):
    """
    Run main_sim.py once and return a dict with parsed metrics.
    Returns None on failure.
    """
    os.makedirs(per_run_outdir, exist_ok=True)
    log_path = os.path.join(per_run_outdir, "run.log")
    cmd = [
        python_bin, "simulation/main_sim.py",
        "--sram_kb", str(sram_kb),
        "--feature_dim", str(feature_dim),
        "--pe_count", str(pe_count),
        "--output_dir", per_run_outdir,
        "--seed", str(seed)
    ]
    print(f"[RUN ] {' '.join(cmd)}  (logs -> {log_path})")
    with open(log_path, "wb") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[ERROR] Run failed (rc={proc.returncode}). See {log_path}")
        return None

    # Prefer latest_simulation_results.csv (single-run canonical file)
    latest_csv = os.path.join(per_run_outdir, "latest_simulation_results.csv")
    fallback_csv = os.path.join(per_run_outdir, "simulation_results.csv")
    csv_path = latest_csv if os.path.exists(latest_csv) else (fallback_csv if os.path.exists(fallback_csv) else None)
    if not csv_path:
        print(f"[WARN] No results CSV found in {per_run_outdir}")
        return None

    # Parse CSV, expect baseline + enhanced rows or find Enhanced row by name
    try:
        with open(csv_path, "r", newline="") as f:
            reader = list(csv.DictReader(f))
            if len(reader) == 0:
                print(f"[WARN] CSV {csv_path} is empty")
                return None

            # Try to find rows by config_name containing 'Enhanced' or 'Baseline'
            baseline_row = None
            enhanced_row = None
            for r in reader:
                name = (r.get("config_name") or "").lower()
                if "baseline" in name and baseline_row is None:
                    baseline_row = r
                elif "enhanced" in name and enhanced_row is None:
                    enhanced_row = r

            # Fallback: if exactly two rows, assume first=baseline, second=enhanced
            if baseline_row is None or enhanced_row is None:
                if len(reader) >= 2:
                    baseline_row = baseline_row or reader[0]
                    enhanced_row = enhanced_row or reader[1]
                else:
                    # take last row as enhanced, first as baseline if possible
                    baseline_row = baseline_row or reader[0]
                    enhanced_row = enhanced_row or reader[-1]

            # Extract numeric values (guarding against empty/missing)
            def safe_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            baseline_time = safe_float(baseline_row.get("total_time_ns")) if baseline_row else None
            enhanced_time = safe_float(enhanced_row.get("total_time_ns")) if enhanced_row else None
            baseline_dram = safe_float(baseline_row.get("total_dram_traffic_bytes")) if baseline_row else None
            enhanced_dram = safe_float(enhanced_row.get("total_dram_traffic_bytes")) if enhanced_row else None

            return {
                "sram_kb": sram_kb,
                "feature_dim": feature_dim,
                "pe_count": pe_count,
                "seed": seed,
                "per_run_outdir": os.path.abspath(per_run_outdir),
                "baseline_time_ns": baseline_time,
                "enhanced_time_ns": enhanced_time,
                "baseline_dram_bytes": baseline_dram,
                "enhanced_dram_bytes": enhanced_dram,
                "csv_used": os.path.abspath(csv_path)
            }
    except Exception as e:
        print(f"[ERROR] Failed to parse {csv_path}: {e}")
        return None

def summarize_runs(all_runs, outdir_base):
    """
    all_runs: list of per-run dicts (may include None entries)
    Group by (sram_kb, feature_dim, pe_count) and compute mean/std of speedup and DRAM reduction.
    Write detailed and summary CSVs and a plot.
    """
    os.makedirs(outdir_base, exist_ok=True)
    detailed_path = os.path.join(outdir_base, DETAILED_CSV)
    summary_path = os.path.join(outdir_base, SUMMARY_CSV)
    plot_path = os.path.join(outdir_base, PLOT_PNG)

    # Write detailed CSV
    with open(detailed_path, "w", newline="") as f:
        fieldnames = [
            "sram_kb","feature_dim","pe_count","seed","per_run_outdir",
            "baseline_time_ns","enhanced_time_ns","speedup",
            "baseline_dram_bytes","enhanced_dram_bytes","dram_reduction_pct","csv_used"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_runs:
            if r is None:
                continue
            baseline = r.get("baseline_time_ns")
            enhanced = r.get("enhanced_time_ns")
            speedup = None
            if baseline and enhanced and enhanced > 0:
                speedup = baseline / enhanced
            baseline_dram = r.get("baseline_dram_bytes")
            enhanced_dram = r.get("enhanced_dram_bytes")
            dram_reduction = None
            if baseline_dram and enhanced_dram and baseline_dram > 0:
                dram_reduction = (baseline_dram - enhanced_dram) / baseline_dram * 100.0
            row = {
                **{k: r.get(k) for k in ["sram_kb","feature_dim","pe_count","seed","per_run_outdir"]},
                "baseline_time_ns": baseline,
                "enhanced_time_ns": enhanced,
                "speedup": speedup,
                "baseline_dram_bytes": baseline_dram,
                "enhanced_dram_bytes": enhanced_dram,
                "dram_reduction_pct": dram_reduction,
                "csv_used": r.get("csv_used")
            }
            w.writerow(row)
    print(f"[OUT ] Detailed sweep runs written to {detailed_path}")

    # Aggregate into summary
    groups = {}
    for r in all_runs:
        if r is None:
            continue
        key = (r["sram_kb"], r["feature_dim"], r["pe_count"])
        groups.setdefault(key, {"speedups": [], "dram_reductions": []})
        baseline = r.get("baseline_time_ns")
        enhanced = r.get("enhanced_time_ns")
        if baseline is not None and enhanced is not None and enhanced > 0:
            groups[key]["speedups"].append(baseline / enhanced)
        baseline_dram = r.get("baseline_dram_bytes")
        enhanced_dram = r.get("enhanced_dram_bytes")
        if baseline_dram is not None and enhanced_dram is not None and baseline_dram > 0:
            groups[key]["dram_reductions"].append((baseline_dram - enhanced_dram) / baseline_dram * 100.0)

    # Write summary CSV
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["sram_kb","feature_dim","pe_count","mean_speedup","std_speedup","mean_dram_reduction_pct","std_dram_reduction_pct","num_valid_trials"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        summary_rows = []
        for key, vals in groups.items():
            sram_kb, feature_dim, pe_count = key
            sp = vals["speedups"]
            dr = vals["dram_reductions"]
            mean_sp = mean(sp) if len(sp) > 0 else None
            std_sp = pstdev(sp) if len(sp) > 1 else 0.0
            mean_dr = mean(dr) if len(dr) > 0 else None
            std_dr = pstdev(dr) if len(dr) > 1 else 0.0
            num_valid = max(len(sp), len(dr))
            row = {
                "sram_kb": sram_kb,
                "feature_dim": feature_dim,
                "pe_count": pe_count,
                "mean_speedup": mean_sp,
                "std_speedup": std_sp,
                "mean_dram_reduction_pct": mean_dr,
                "std_dram_reduction_pct": std_dr,
                "num_valid_trials": num_valid
            }
            w.writerow(row)
            summary_rows.append((f"{sram_kb}KB-{feature_dim}D-{pe_count}PE", mean_sp, std_sp))

    print(f"[OUT ] Summary written to {summary_path}")

    # Make a quick bar plot of mean speedup with errorbars
    labels = [r[0] for r in summary_rows]
    means = [r[1] if r[1] is not None else float('nan') for r in summary_rows]
    stds = [r[2] if r[2] is not None else 0.0 for r in summary_rows]

    if labels:
        x = range(len(labels))
        plt.figure(figsize=(10, 6))
        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Mean Speedup (baseline / enhanced)")
        plt.title("I-GCN Sweep: Mean Speedup ± STD")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"[OUT ] Plot written to {plot_path}")
    else:
        print("[WARN] No summary rows to plot (all runs failed or no valid speedups).")

def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for I-GCN and aggregate results.")
    parser.add_argument("--base_outdir", type=str, default=None, help="Base directory for per-run outputs (overrides embedded placeholder)")
    parser.add_argument("--trials", type=int, default=3, help="Trials per parameter set (different seeds)")
    parser.add_argument("--python_bin", type=str, default="python3", help="Python binary to use when running main_sim.py")
    args = parser.parse_args()

    base_outdir = args.base_outdir if args.base_outdir else BASE_OUTPUT_DIR
    if base_outdir == "PUT_YOUR_OUTPUT_BASE_DIR_HERE":
        print("[ERROR] Please set BASE_OUTPUT_DIR in this script or pass --base_outdir on the command line.")
        return
    os.makedirs(base_outdir, exist_ok=True)

    param_sets = DEFAULT_PARAM_SETS

    all_runs = []
    for (sram_kb, feature_dim, pe_count) in param_sets:
        for t in range(args.trials):
            seed = 1000 + t
            per_run_dir = os.path.join(base_outdir, f"tmp_{sram_kb}_{feature_dim}_{pe_count}_{t}")
            # run main_sim.py for this configuration
            result = run_once(sram_kb, feature_dim, pe_count, seed, per_run_dir, python_bin=args.python_bin)
            if result is None:
                print(f"[WARN] Run failed for {sram_kb}KB {feature_dim}D {pe_count}PE seed={seed}")
            all_runs.append(result)

    # Summarize and write CSV + plot
    summarize_runs([r for r in all_runs if r is not None], base_outdir)

if __name__ == "__main__":
    main()
