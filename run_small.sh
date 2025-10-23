#!/usr/bin/env bash
# Quick smoke test for I-GCN project
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/simulation_results"
mkdir -p "$OUT"

echo "Running I-GCN smoke test..."

# recommended quick params: small but representative
python3 simulation/main_sim.py --sram_kb 8 --feature_dim 64 --pe_count 4 --output_dir "$OUT" --seed 42

echo "Smoke test finished. Check $OUT for simulation_results.png and simulation_results.csv"
