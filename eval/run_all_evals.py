"""
Master evaluation runner — runs all experimental analyses in sequence.

Produces all figures and tables needed for the paper:
  1. Topology Irreducibility validation (Figure 2, Table S1)
  2. GED Score Distributions (Figure 3)
  3. DP Privacy Budget (Figure 4)
  4. Ablation Study (Table 1)
  5. Attention Map Visualizations (Figure 5)

Usage:
    uv run python eval/run_all_evals.py [--skip-slow]

Flags:
    --skip-slow   Skip the DP privacy accountant and attention visualizer
                  (fast run for debugging, ~30 seconds vs ~10 minutes full)
"""
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow evals (ablation, attention maps)")
    args = parser.parse_args()

    os.makedirs("eval", exist_ok=True)
    os.makedirs("eval/attention_maps", exist_ok=True)

    print("\n" + "█"*65)
    print("  Causal PoR Finance — Full Experimental Evaluation Suite")
    print("█"*65 + "\n")

    # ── 1. Topology Irreducibility ─────────────────────────────────────────────
    print("\n[1/5] Topology Irreducibility Theorem Validation")
    print("-"*50)
    from eval.topology_irreducibility import run_irreducibility_analysis
    irred_out = run_irreducibility_analysis()
    if irred_out["theorem_holds"]:
        print(f"  ✅ Theorem HOLDS: ε_min = {irred_out['min_adversary_ged']:.6f}")
    else:
        print(f"  ❌ Theorem FAILED — check adversary generation")

    # ── 2. DP Privacy Budget ──────────────────────────────────────────────────
    print("\n[2/5] DP-SGD Privacy Budget Analysis")
    print("-"*50)
    from eval.dp_privacy_accountant import run_privacy_accounting
    dp_out = run_privacy_accounting()
    final_eps = dp_out["final_guarantee"]["epsilon"]
    print(f"  Final guarantee: (ε={final_eps:.4f}, δ=1e-5)-DP after 20 rounds")

    # ── 3. GED Score Distributions ────────────────────────────────────────────
    print("\n[3/5] GED Score Distribution Analysis")
    print("-"*50)
    from eval.ged_distribution import run_ged_distribution
    ged_out = run_ged_distribution()
    for name, d in ged_out["cohens_d"].items():
        print(f"  Cohen's d (Honest vs {name}): {d}")

    if not args.skip_slow:
        # ── 4. Ablation Study ─────────────────────────────────────────────────
        print("\n[4/5] Ablation Study Runner")
        print("-"*50)
        from eval.ablation_runner import run_ablation
        run_ablation()

        # ── 5. Attention Map Visualizations ───────────────────────────────────
        print("\n[5/5] Attention Map Visualization")
        print("-"*50)
        from eval.attention_visualizer import run_attention_analysis
        run_attention_analysis({
            "Round 0 (Random Init)":   None,
            "Round N (PoR-FL Trained)": "saved_models/finance/global_model.pt",
        })
    else:
        print("\n[4/5] Ablation Study — SKIPPED (--skip-slow)")
        print("[5/5] Attention Maps  — SKIPPED (--skip-slow)")

    print("\n" + "█"*65)
    print("  All evaluations complete. Results saved to eval/")
    print("  Key files:")
    print("    eval/topology_irreducibility.json + _plot.png")
    print("    eval/dp_privacy_budget.json + _plot.png")
    print("    eval/ged_distributions.json + _plot.png")
    print("    eval/ablation_results.json + ablation_table.txt (LaTeX)")
    print("    eval/attention_maps/attention_data.json + *.png")
    print("█"*65 + "\n")


if __name__ == "__main__":
    main()
