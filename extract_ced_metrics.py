"""
CED Gate Validation Metrics Extractor
======================================
Computes per-adversary-type detection rates for the Causal Effect Divergence (CED) gate,
reading adversary identity from params.yaml instead of hardcoding.

Usage:  uv run python extract_ced_metrics.py
"""
import json
import os
import yaml

# Read adversary configuration from params.yaml
with open("params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLIENTS   = int(cfg["simulation"]["num_clients"])
NUM_ADV       = int(cfg["simulation"]["num_false_nodes"])
HONEST_OFFSET = NUM_CLIENTS - NUM_ADV  # client IDs >= HONEST_OFFSET are adversarial

logs_path     = "saved_models/finance/simulation_logs.json"
ged_scores_path = "saved_models/finance/ged_scores.json"

if not os.path.exists(ged_scores_path):
    print("GED scores file not found at", ged_scores_path)
    exit(1)

with open(ged_scores_path, "r") as f:
    ged_scores = json.load(f)

print("--- CED Gate Validation Metrics ---")
print(f"  Config: {NUM_CLIENTS} clients, {NUM_ADV} adversaries, honest IDs 0-{HONEST_OFFSET-1}")

# Per-round and aggregate counters
total_adv_submissions   = 0
total_adv_gate_failures = 0
total_adv_actual_reject = 0
total_hon_submissions   = 0
total_hon_gate_failures = 0
total_hon_actual_reject = 0
per_round = []

for round_data in ged_scores:
    r = round_data.get("round", -1)
    scores = round_data.get("scores", {})
    r_adv_sub, r_adv_gate, r_adv_rej = 0, 0, 0
    r_hon_sub, r_hon_gate, r_hon_rej = 0, 0, 0

    for client_id, score_data in scores.items():
        is_adv = int(client_id) >= HONEST_OFFSET
        status = score_data.get("status", "unknown")

        # Gate failure = any status indicating the logic gate rejected
        gate_failed = status in [
            "rejected", "rejected_grace_bypassed",
            "rejected_ced_gate", "rejected_ced_gate_grace_bypassed",
            "rejected_coverage_gate"
        ]
        # Actual rejection = gate failure NOT in grace period
        actually_rejected = status in ["rejected", "rejected_ced_gate", "rejected_coverage_gate"]

        if is_adv:
            r_adv_sub += 1
            if gate_failed:
                r_adv_gate += 1
            if actually_rejected:
                r_adv_rej += 1
        else:
            r_hon_sub += 1
            if gate_failed:
                r_hon_gate += 1
            if actually_rejected:
                r_hon_rej += 1

    total_adv_submissions   += r_adv_sub
    total_adv_gate_failures += r_adv_gate
    total_adv_actual_reject += r_adv_rej
    total_hon_submissions   += r_hon_sub
    total_hon_gate_failures += r_hon_gate
    total_hon_actual_reject += r_hon_rej

    per_round.append({
        "round": r,
        "adv_detection_rate": round(r_adv_rej / max(r_adv_sub, 1), 4),
        "adv_gate_failure_rate": round(r_adv_gate / max(r_adv_sub, 1), 4),
        "hon_fpr": round(r_hon_rej / max(r_hon_sub, 1), 4),
    })

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'Metric':<45} {'Value':>10}")
print("-" * 58)
print(f"{'Total Adversary Submissions':<45} {total_adv_submissions:>10}")
print(f"{'Total Adversary Gate Failures':<45} {total_adv_gate_failures:>10}")
print(f"{'Total Adversary Actual Rejections':<45} {total_adv_actual_reject:>10}")
print(f"{'Total Honest Submissions':<45} {total_hon_submissions:>10}")
print(f"{'Total Honest Gate Failures':<45} {total_hon_gate_failures:>10}")
print(f"{'Total Honest Actual Rejections':<45} {total_hon_actual_reject:>10}")

if total_adv_submissions > 0:
    tpr_gate = total_adv_gate_failures / total_adv_submissions
    tpr_rej  = total_adv_actual_reject / total_adv_submissions
    print(f"\n{'Adversary Gate Failure Rate (TPR_gate)':<45} {tpr_gate:>10.2%}")
    print(f"{'Adversary Actual Rejection Rate (TPR_rej)':<45} {tpr_rej:>10.2%}")

if total_hon_submissions > 0:
    fpr_gate = total_hon_gate_failures / total_hon_submissions
    fpr_rej  = total_hon_actual_reject / total_hon_submissions
    print(f"{'Honest Gate Failure Rate (FPR_gate)':<45} {fpr_gate:>10.2%}")
    print(f"{'Honest Actual Rejection Rate (FPR_rej)':<45} {fpr_rej:>10.2%}")

# Per-round breakdown
print(f"\n{'Round':>6} {'Adv Gate%':>12} {'Adv Rej%':>12} {'Hon FPR%':>12}")
print("-" * 46)
for r in per_round:
    print(f"{r['round']:>6} {r['adv_gate_failure_rate']:>12.2%} {r['adv_detection_rate']:>12.2%} {r['hon_fpr']:>12.2%}")

# Save structured results
out_path = "saved_models/finance/ced_metrics.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "config": {"num_clients": NUM_CLIENTS, "num_adversaries": NUM_ADV, "honest_offset": HONEST_OFFSET},
        "aggregate": {
            "adv_tpr_gate": round(tpr_gate, 4) if total_adv_submissions > 0 else None,
            "adv_tpr_rej": round(tpr_rej, 4) if total_adv_submissions > 0 else None,
            "hon_fpr_gate": round(fpr_gate, 4) if total_hon_submissions > 0 else None,
            "hon_fpr_rej": round(fpr_rej, 4) if total_hon_submissions > 0 else None,
        },
        "per_round": per_round,
    }, f, indent=2)

print(f"\nResults saved to {out_path}")
