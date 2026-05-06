#!/usr/bin/env python3
"""
visualizations/run_all_plots.py
==================
Master script — runs all graph generation scripts in order.
Usage: python visualizations/run_all_plots.py
All PNGs are saved to results/figures/.
"""
import subprocess
import sys
import time
from pathlib import Path

GRAPHS_DIR = Path(__file__).parent  # visualizations/
SCRIPTS = [
    "g01_acceptance_bars.py",
    "g02_ged_distribution.py",
    "g03_roc_curve.py",
    "g04_threshold_sensitivity.py",
    "g05_cumulative_suppression.py",
    "g06_simgnn_speedup.py",
    "g07_loss_convergence.py",
    "g08_consensus_jaccard.py",
    "g09_radar_detection.py",
    "g10_genome_architecture.py",
    "g11_asia_ground_truth.py",
    "g12_ged_score_heatmap.py",
    "g13_rejected_edge_diff.py",
    "g14_defense_summary_table.py",
    "g15_notears_edge_analysis.py",
    "g16_byzantine_tolerance.py",
    "g17_multiround_ged_trend.py",
    # ── Extra 3 (Blueprint G6, G9, G18) ──────────────────────────────────────
    # G18/G20 will auto-run experiments/eval_metrics.py if eval_results.json is missing.
    # G19 shows single-point if per-round consensus files are missing (need re-sim).
    "g18_global_accuracy.py",
    "g19_consensus_jaccard_rounds.py",
    "g20_attack_success_rate.py",
]

DATA_STATUS = {
    "saved_models/asia/ged_scores.json":          "✓ Last-round GED scores",
    "saved_models/asia/simulation_logs.json":     "✓ PoR simulation logs",
    "saved_models/baseline/simulation_logs.json": "✓ Baseline simulation logs",
    "saved_models/asia/consensus_graph.gpickle":  "✓ Final consensus graph",
    "saved_models/asia/rejected_edge_diff.json":  "✓ Rejected edge diff",
    "saved_models/realtime_state.json":           "✓ FedNEAT genome state",
    "saved_models/asia/eval_results.json":
        "⚠ MTA/ASR eval (run: python experiments/eval_metrics.py)",
}

def main():
    passed, failed = [], []
    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"  FYP Graph Generator — Running {len(SCRIPTS)} scripts")
    print(f"{'='*60}\n")

    # Pre-flight: data availability check
    PROJECT = GRAPHS_DIR.parent
    print("  Data availability:")
    for rel_path, desc in DATA_STATUS.items():
        exists = (PROJECT / rel_path).exists()
        status = "✓" if exists else "✗  MISSING"
        print(f"    {status}  {rel_path.split('/')[-1]:40s}  {desc if exists else desc.split('(')[0].strip()}")
    print()

    for script in SCRIPTS:
        path = GRAPHS_DIR / script
        if not path.exists():
            print(f"  SKIP  {script}  (file not found)")
            failed.append((script, "File not found"))
            continue

        print(f"  ▶  {script} ...", end="", flush=True)
        t0  = time.time()
        res = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True
        )
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"  ✓  ({elapsed:.1f}s)")
            passed.append(script)
        else:
            print(f"  ✗  FAILED ({elapsed:.1f}s)")
            print(f"     {res.stderr.strip().splitlines()[-1]}")
            failed.append((script, res.stderr.strip().splitlines()[-1]))

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Done in {total:.1f}s  |  {len(passed)} passed  |  {len(failed)} failed")

    # list generated PNGs in results/figures/
    results_figures = Path(__file__).parent.parent / "results" / "figures"
    pngs = sorted(results_figures.glob("G*.png"))
    print(f"\n  Generated {len(pngs)} graph files:")
    for p in pngs:
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:<48}  {size_kb:>7.1f} KB")

    if failed:
        print(f"\n  Failed scripts:")
        for s, err in failed:
            print(f"    ✗  {s}: {err}")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
