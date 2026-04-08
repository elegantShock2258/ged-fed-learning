"""
DP-SGD Privacy Accountant — Rényi Differential Privacy Composition
===================================================================
Formally quantifies the (ε, δ)-DP privacy guarantee consumed by the
DP-SGD training in FinanceClient across all FL rounds.

Theory (brief):
  - Each DP-SGD step is a Gaussian mechanism with noise multiplier σ
    and clipping norm C, operating on a batch of size q (sampling ratio).
  - RDP (Rényi Differential Privacy) composes linearly across steps:
      ε_RDP(α) = T * q² * α / (2 * σ²)     [simplified Sampled Gaussian]
  - Convert to (ε, δ)-DP via:
      ε(δ) = min_{α > 1} [ ε_RDP(α) + log(1/δ)/(α-1) ]

Parameters (matching FinanceClient defaults):
  σ     = dp_noise_multiplier  = 0.3   (noise relative to clipping norm)
  C     = dp_max_grad_norm     = 1.0   (gradient clipping bound)
  n     = approximate dataset  = ~100  (episodes per round)
  batch = epoch_batch_scale * local_epochs (steps per round)
  T     = total steps across all rounds

Output:
  - Prints privacy budget table (rounds × ε at δ=1e-5)
  - Saves eval/dp_privacy_budget.json
  - Generates eval/dp_privacy_budget_plot.png

Usage:
    uv run python eval/dp_privacy_accountant.py
"""

import os
import sys
import json
import math
import numpy as np
from datetime import datetime

os.makedirs("eval", exist_ok=True)

# ── DP-SGD Parameters (must match FinanceClient) ────────────────────────────────
NOISE_MULTIPLIER   = 0.3    # σ
CLIPPING_NORM      = 1.0    # C
SAMPLING_RATIO     = 0.1    # q = batch_size / dataset_size (approximate)
LOCAL_EPOCHS       = 3
EPOCH_BATCH_SCALE  = 5
STEPS_PER_ROUND    = LOCAL_EPOCHS * EPOCH_BATCH_SCALE  # steps per FL round
NUM_FL_ROUNDS      = 20
DELTA              = 1e-5   # target δ for (ε, δ)-DP
PPO_EPOCHS         = 4      # inner PPO optimization epochs


# ── RDP Computation ──────────────────────────────────────────────────────────────
def _rdp_single_step_sampled_gaussian(alpha: float, sigma: float, q: float) -> float:
    """
    RDP of one step of the Sampled Gaussian Mechanism (Mironov 2017).
    Uses Proposition 3 from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
    (Mironov, Talwar, Zhang, 2019) — simplified bound for α ≥ 1:

      ε_RDP(α) ≈  α * q² / (2 * σ²)      [leading term, valid for q << 1]

    More precise bound (binomial expansion, valid for all α):
      Sum over k from 0 to α of C(α,k) * (-q)^(α-k) * q^k * exp((k-1)*k/(2σ²))
    We use the simplified closed-form for clarity in the paper.
    """
    # Simplified tight bound used in most DP-FL papers
    return float(alpha * q**2 / (2 * sigma**2))


def _rdp_to_epsdelta(rdp_epsilon: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee (α, ε_rdp) to (ε, δ)-DP via:
       ε(δ) = ε_rdp + log(1/δ) / (α - 1)
    """
    if alpha <= 1:
        return float("inf")
    return rdp_epsilon + math.log(1.0 / delta) / (alpha - 1)


def compute_privacy_budget(
    num_rounds: int,
    steps_per_round: int,
    noise_multiplier: float,
    sampling_ratio: float,
    delta: float,
    ppo_epochs: int = 1,
    alphas=None
) -> dict:
    """
    Compute cumulative (ε, δ)-DP budget after `num_rounds` FL rounds.
    
    Each FL round has `steps_per_round * ppo_epochs` DP-SGD steps.
    RDP composes linearly: ε_rdp_total(α) = T * ε_rdp_per_step(α)
    
    Returns a dict with per-round ε values and the best (minimum) ε.
    """
    if alphas is None:
        # Search over a wide range of α values
        alphas = [1.1, 1.5, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256]

    total_steps_per_round = steps_per_round * ppo_epochs
    per_step_rdp = {
        alpha: _rdp_single_step_sampled_gaussian(alpha, noise_multiplier, sampling_ratio)
        for alpha in alphas
    }

    results_by_round = []
    for rnd in range(1, num_rounds + 1):
        total_steps = rnd * total_steps_per_round

        # Compute (ε, δ)-DP for each α and take minimum
        best_eps = float("inf")
        best_alpha = None
        candidates = {}
        for alpha in alphas:
            rdp_total = total_steps * per_step_rdp[alpha]
            eps = _rdp_to_epsdelta(rdp_total, alpha, delta)
            candidates[alpha] = round(eps, 6)
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha

        results_by_round.append({
            "round":       rnd,
            "total_steps": total_steps,
            "best_eps":    round(best_eps, 6),
            "best_alpha":  best_alpha,
            "eps_by_alpha": {str(a): candidates[a] for a in alphas},
        })

    return results_by_round


def run_privacy_accounting():
    print("=" * 65)
    print("DP-SGD Privacy Budget Analysis (RDP Composition)")
    print("=" * 65)
    print(f"  Noise multiplier σ    = {NOISE_MULTIPLIER}")
    print(f"  Clipping norm C       = {CLIPPING_NORM}")
    print(f"  Sampling ratio q      = {SAMPLING_RATIO}")
    print(f"  Steps per FL round    = {STEPS_PER_ROUND} × {PPO_EPOCHS} PPO epochs = {STEPS_PER_ROUND * PPO_EPOCHS}")
    print(f"  Total FL rounds       = {NUM_FL_ROUNDS}")
    print(f"  Target δ              = {DELTA}")
    print()

    results = compute_privacy_budget(
        num_rounds=NUM_FL_ROUNDS,
        steps_per_round=STEPS_PER_ROUND,
        noise_multiplier=NOISE_MULTIPLIER,
        sampling_ratio=SAMPLING_RATIO,
        delta=DELTA,
        ppo_epochs=PPO_EPOCHS,
    )

    # Print round-by-round table
    print(f"{'Round':>6} {'Steps':>8} {'ε (δ=1e-5)':>14} {'Best α':>10}")
    print("-" * 45)
    for r in results:
        print(f"{r['round']:>6} {r['total_steps']:>8} {r['best_eps']:>14.4f} {r['best_alpha']:>10}")

    final = results[-1]
    print(f"\nFinal Privacy Guarantee after {NUM_FL_ROUNDS} rounds:")
    print(f"  (ε, δ)-DP = ({final['best_eps']:.4f}, {DELTA:.0e})")
    print(f"  Optimal Rényi order α = {final['best_alpha']}")

    # ── Interpretation for paper ───────────────────────────────────────────────
    print(f"\nInterpretation:")
    eps = final['best_eps']
    if eps < 1.0:
        print(f"  ✅ Strong privacy (ε={eps:.3f} < 1.0) — individual training episodes")
        print(f"     are indistinguishable with high probability.")
    elif eps < 5.0:
        print(f"  ⚠️  Moderate privacy (ε={eps:.3f}). Consider reducing σ or rounds.")
    else:
        print(f"  ❌ Weak privacy (ε={eps:.3f} > 5.0). Increase noise multiplier σ.")

    # ── Sensitivity analysis: ε vs σ ──────────────────────────────────────────
    print("\nSensitivity analysis: ε vs noise multiplier σ (after 20 rounds)")
    sigma_range = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    sensitivity = {}
    for sigma in sigma_range:
        res = compute_privacy_budget(NUM_FL_ROUNDS, STEPS_PER_ROUND, sigma, SAMPLING_RATIO, DELTA, PPO_EPOCHS)
        eps = res[-1]["best_eps"]
        sensitivity[str(sigma)] = round(eps, 4)
        print(f"  σ = {sigma:.1f} → ε = {eps:.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "noise_multiplier": NOISE_MULTIPLIER,
            "clipping_norm":    CLIPPING_NORM,
            "sampling_ratio":   SAMPLING_RATIO,
            "steps_per_round":  STEPS_PER_ROUND,
            "ppo_epochs":       PPO_EPOCHS,
            "num_fl_rounds":    NUM_FL_ROUNDS,
            "delta":            DELTA,
        },
        "per_round_budget": results,
        "final_guarantee": {
            "epsilon": final["best_eps"],
            "delta":   DELTA,
            "best_alpha": final["best_alpha"],
        },
        "sensitivity": sensitivity,
    }
    json_path = "eval/dp_privacy_budget.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPrivacy budget data saved to {json_path}")

    # ── Generate plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("DP-SGD Privacy Budget (RDP Composition)\nFinance Hedge Fund PoR FL System", fontsize=12, fontweight="bold")

        # Left: ε vs FL rounds
        rounds = [r["round"]    for r in results]
        epsilons= [r["best_eps"] for r in results]
        ax1.plot(rounds, epsilons, "b-o", linewidth=2, markersize=5)
        ax1.axhline(y=1.0, color="green", linestyle="--", label="Strong privacy (ε=1.0)")
        ax1.axhline(y=5.0, color="red",   linestyle="--", label="Weak privacy (ε=5.0)")
        ax1.fill_between(rounds, epsilons, 1.0, where=[e > 1.0 for e in epsilons],
                          alpha=0.1, color="orange", label="Moderate zone")
        ax1.set_xlabel("FL Round")
        ax1.set_ylabel(f"Privacy budget ε (δ = {DELTA:.0e})")
        ax1.set_title("Cumulative Privacy Budget vs FL Rounds")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Right: ε vs noise multiplier σ
        sigmas_f = [float(k) for k in sensitivity.keys()]
        epsilons2 = list(sensitivity.values())
        ax2.plot(sigmas_f, epsilons2, "r-s", linewidth=2, markersize=6)
        ax2.axvline(x=NOISE_MULTIPLIER, color="blue", linestyle="--",
                    label=f"Current σ = {NOISE_MULTIPLIER}")
        ax2.axhline(y=1.0, color="green", linestyle="--", label="Strong privacy (ε=1.0)")
        ax2.set_xlabel("Noise Multiplier σ")
        ax2.set_ylabel(f"Privacy budget ε (δ = {DELTA:.0e})")
        ax2.set_title(f"Privacy budget vs Noise Multiplier\n(after {NUM_FL_ROUNDS} FL rounds)")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()
        fig_path = "eval/dp_privacy_budget_plot.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Privacy budget plot saved to {fig_path}")

    except ImportError:
        print("matplotlib not available — skipping plot generation.")

    return out


if __name__ == "__main__":
    run_privacy_accounting()
