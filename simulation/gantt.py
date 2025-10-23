#!/usr/bin/env python3
"""
gantt.py (fixed)

Creates readable PE utilization + DRAM transfer Gantt charts from the CSV logs
produced by the simulator.

Usage:
  python gantt.py --input_dir simulation_results
  python gantt.py --input_dir simulation_results --out_dir my_plots --show
"""
from __future__ import annotations
import os
import csv
import glob
import argparse
from typing import List, Dict, Tuple
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

# ---------- I/O helpers ----------
def find_sim_bases(input_dir: str) -> List[str]:
    """Find distinct simulator base names that have a pe_timeline CSV."""
    pattern = os.path.join(input_dir, "*_pe_timeline.csv")
    files = glob.glob(pattern)
    bases = []
    for p in files:
        name = os.path.basename(p)
        if name.endswith("_pe_timeline.csv"):
            bases.append(name.replace("_pe_timeline.csv", ""))
    return sorted(bases)

def read_pe_csv(path: str) -> Dict[int, List[Tuple[float, float, str]]]:
    """Reads pe timeline CSV and returns mapping pe_id -> list of (start_ns, end_ns, label)."""
    result: Dict[int, List[Tuple[float, float, str]]] = {}
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                pe_id = int(row[0])
                start = float(row[1])
                end = float(row[2])
                label = row[3]
            except Exception:
                continue
            result.setdefault(pe_id, []).append((start, end, label))
    for k in result:
        result[k].sort(key=lambda x: x[0])
    return result

def read_dram_csv(path: str) -> List[Dict]:
    """Reads dram timeline CSV and returns list of dicts with keys start_ns,end_ns,bytes,node_count,reason."""
    out = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 5:
                continue
            try:
                start = float(row[0])
                end = float(row[1])
                bytes_ = int(row[2])
                node_count = int(row[3])
                reason = row[4]
            except Exception:
                continue
            out.append({'start_ns': start, 'end_ns': end, 'bytes': bytes_, 'node_count': node_count, 'reason': reason})
    out.sort(key=lambda x: x['start_ns'])
    return out

# ---------- plotting helpers ----------
def choose_time_unit(max_ns: float) -> Tuple[str, float]:
    """Return (unit_label, divisor) for nicer x-axis ticks."""
    if max_ns >= 1e9:
        return 's', 1e9
    if max_ns >= 1e6:
        return 'ms', 1e6
    if max_ns >= 1e3:
        return 'µs', 1e3
    return 'ns', 1.0

def human_bytes(n: int) -> str:
    """Return human-readable byte count."""
    for unit in ['B','KB','MB','GB','TB']:
        if abs(n) < 1024.0 or unit == 'TB':
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.0f}B"

def plot_gantt_for_base(base: str, pe_map: Dict[int, List[Tuple[float, float, str]]],
                        dram_entries: List[Dict], out_dir: str, show: bool = False):
    """Create and save a Gantt chart for a given simulator base name."""
    # collect time range
    all_starts = []
    all_ends = []
    for entries in pe_map.values():
        for s, e, _ in entries:
            all_starts.append(s)
            all_ends.append(e)
    for d in dram_entries:
        all_starts.append(d['start_ns'])
        all_ends.append(d['end_ns'])
    if not all_starts or not all_ends:
        print(f"Skipping {base}: no timeline entries.")
        return

    t_min = min(all_starts)
    t_max = max(all_ends)
    span = max(1.0, t_max - t_min)

    unit_label, divisor = choose_time_unit(t_max)

    def xs(x): return (x - t_min) / divisor

    # Prepare figure height depending on number of PEs
    pe_ids = sorted(pe_map.keys())
    n_pes = len(pe_ids)
    # figure height heuristics
    row_height = 0.6
    dram_height = 1.4
    top_pad = 1.0
    fig_height = max(4.0, n_pes * row_height + dram_height + top_pad)
    fig_width = 14.0
    fig, (ax_pe, ax_dram) = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                         gridspec_kw={'height_ratios': [max(1, n_pes * row_height), dram_height]})

    # Map each pe_id to a y coordinate (PE 0 at top)
    # matplotlib y increases upward, but broken_barh draws bars at given y positions.
    # We'll put PE 0 at the top row visually by mapping to high y values.
    pe_sorted = sorted(pe_ids)
    y_positions = {pe_id: (n_pes - 1 - idx) for idx, pe_id in enumerate(pe_sorted)}
    y_tick_positions = [y_positions[pid] for pid in pe_sorted]  # ordered by pe_id
    y_tick_labels = [str(pid) for pid in pe_sorted]

    # Colors for PE tasks - cycle through tab20
    cmap_pe = plt.get_cmap('tab20')
    color_cycle = [cmap_pe(i) for i in range(cmap_pe.N)]

    # label width threshold (in chosen time units) for drawing labels inside bars
    min_label_width_units = max( (span / divisor) * 0.03, 0.5 )  # heuristic: 3% of span or 0.5 units
    # absolute minimum displayed width for very short-unit spans
    min_label_width_units = float(min_label_width_units)

    # plot PE bars
    for pe_id, entries in pe_map.items():
        y = y_positions[pe_id]
        for idx, (s, e, label) in enumerate(entries):
            start_u = xs(s)
            width_u = (e - s) / divisor
            color = color_cycle[abs(hash(label)) % len(color_cycle)]
            ax_pe.broken_barh([(start_u, width_u)], (y - 0.4, 0.8),
                              facecolors=color, edgecolor='k', linewidth=0.25)
            # draw label only if we have enough horizontal room
            if width_u >= min_label_width_units:
                # font size depends on width
                fs = 8 if width_u > 2*min_label_width_units else 7
                # create short label, avoid extremely long labels
                lab = label if len(label) <= 24 else (label[:21] + '...')
                ax_pe.text(start_u + 0.01 * (span/divisor), y, lab,
                           va='center', ha='left', fontsize=fs, clip_on=True,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.6, edgecolor='none'))

    ax_pe.set_yticks([y_positions[int(pid)] for pid in pe_sorted])
    ax_pe.set_yticklabels([str(pid) for pid in pe_sorted])
    ax_pe.set_ylabel('PE id')
    ax_pe.set_title(f'PE Utilization Gantt: {base}')
    ax_pe.grid(axis='x', linestyle='--', alpha=0.45)
    # set y-limits so bars are nicely centered
    ax_pe.set_ylim(-0.6, n_pes - 0.4)

    # DRAM timeline: color by bytes (log scale to better show spread)
    if dram_entries:
        bytes_list = [d['bytes'] for d in dram_entries]
        b_min = min(bytes_list)
        b_max = max(bytes_list)
        # avoid degenerate range
        if b_min == b_max:
            norm = mcolors.Normalize(vmin=max(1, b_min*0.9), vmax=max(1, b_max*1.1))
        else:
            norm = mcolors.LogNorm(vmin=max(1, b_min), vmax=max(1, b_max))
        cmap = cm.get_cmap('viridis')

        dram_y = 0
        bar_height = 0.6
        # choose a threshold: only label big or wide transfers to avoid clutter
        if bytes_list:
            bytes_thresh = float(np.percentile(bytes_list, 75))
        else:
            bytes_thresh = 1

        for d in dram_entries:
            start_u = xs(d['start_ns'])
            width_u = (d['end_ns'] - d['start_ns']) / divisor
            color = cmap(norm(max(1, d['bytes'])))
            ax_dram.broken_barh([(start_u, width_u)], (dram_y - bar_height/2, bar_height),
                                facecolors=color, edgecolor='k', linewidth=0.25)
            # annotate only if wide enough or bytes large
            if (width_u >= min_label_width_units) or (d['bytes'] >= bytes_thresh):
                # prefer reason + node_count for annotation if there's space, else bytes
                if width_u >= min_label_width_units:
                    ann = f"{d['reason']} ({d['node_count']})"
                else:
                    ann = f"{human_bytes(d['bytes'])}"
                ax_dram.text(start_u + 0.01 * (span/divisor), dram_y, ann,
                              va='center', ha='left', fontsize=8, clip_on=True,
                              bbox=dict(boxstyle="round,pad=0.12", facecolor='white', alpha=0.75, edgecolor='none'))

        # colorbar for bytes
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_dram, orientation='vertical', pad=0.02, fraction=0.035)
        cbar.set_label('DRAM transfer bytes (log scale)')

    ax_dram.set_xlabel(f"Time ({unit_label})")
    ax_dram.set_yticks([])
    ax_dram.set_title('DRAM Transfer Timeline')
    ax_dram.grid(axis='x', linestyle='--', alpha=0.35)

    # Shared x-axis limits (with a small margin)
    margin = max(1e-6, 0.01 * (t_max - t_min)) / divisor
    ax_pe.set_xlim(xs(t_min) - margin, xs(t_max) + margin)
    ax_dram.set_xlim(xs(t_min) - margin, xs(t_max) + margin)

    # Format x-ticks so they are human readable
    # pick ~6 ticks
    ticks = np.linspace(xs(t_min), xs(t_max), num=6)
    tick_labels = [f"{(tick*divisor + t_min)/divisor:.2f}".rstrip('0').rstrip('.') for tick in ticks]
    ax_pe.set_xticks(ticks)
    ax_pe.set_xticklabels(tick_labels)
    ax_dram.set_xticks(ticks)
    ax_dram.set_xticklabels(tick_labels)

    plt.tight_layout()

    # Save figure as PNG only
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"gantt_{base}.png")
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    print(f"Saved: {out_png}")

    if show:
        plt.show()
    plt.close(fig)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Create Gantt charts from simulation_results logs")
    parser.add_argument('--input_dir', type=str, default='simulation_results', help='Directory containing *_pe_timeline.csv and *_dram_timeline.csv')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save plots (defaults to input_dir)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--base', type=str, default=None, help='If provided, only plot this simulator base name (exact match)')
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir or input_dir

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    bases = find_sim_bases(input_dir)
    if not bases:
        raise SystemExit(f"No '*_pe_timeline.csv' files found in {input_dir}")

    if args.base:
        if args.base not in bases:
            raise SystemExit(f"Requested base '{args.base}' not found. Available: {bases}")
        bases = [args.base]

    for base in bases:
        pe_path = os.path.join(input_dir, f"{base}_pe_timeline.csv")
        dram_path = os.path.join(input_dir, f"{base}_dram_timeline.csv")
        print(f"\nProcessing: {base}")
        pe_map = read_pe_csv(pe_path)
        dram_entries = read_dram_csv(dram_path) if os.path.isfile(dram_path) else []
        plot_gantt_for_base(base, pe_map, dram_entries, out_dir, show=args.show)

    print("\nAll done.")

if __name__ == "__main__":
    main()
