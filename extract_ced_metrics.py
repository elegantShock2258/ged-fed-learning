import json
import os

logs_path = "saved_models/finance/simulation_logs.json"
ged_scores_path = "saved_models/finance/ged_scores.json"

if not os.path.exists(logs_path) or not os.path.exists(ged_scores_path):
    print("Log files not found.")
    exit()

with open(logs_path, "r") as f:
    logs = json.load(f)

with open(ged_scores_path, "r") as f:
    ged_scores = json.load(f)

print("--- CED Gate Validation Metrics ---")

rounds = ged_scores
total_adversary_submissions = 0
total_adversary_rejections = 0
total_honest_submissions = 0
total_honest_rejections = 0

for round_data in rounds:
    r = round_data.get("round")
    scores = round_data.get("scores", {})
    for client_id, score_data in scores.items():
        is_adv = int(client_id) >= 9  # Based on the run_sim_sequential log
        
        status = score_data.get("status")
        # Logic rejection is 'rejected', 'rejected_grace_bypassed', 'rejected_ced_gate', 'rejected_ced_gate_grace_bypassed'
        logic_rejected = status in ["rejected", "rejected_grace_bypassed", "rejected_ced_gate", "rejected_ced_gate_grace_bypassed"]
        
        if is_adv:
            total_adversary_submissions += 1
            if logic_rejected:
                total_adversary_rejections += 1
        else:
            total_honest_submissions += 1
            if logic_rejected:
                total_honest_rejections += 1

print(f"Total Adversary Submissions: {total_adversary_submissions}")
print(f"Total Adversary Rejections: {total_adversary_rejections}")
print(f"Total Honest Submissions: {total_honest_submissions}")
print(f"Total Honest Rejections: {total_honest_rejections}")

if total_adversary_submissions > 0:
    print(f"True Positive Rate (Detection): {total_adversary_rejections/total_adversary_submissions*100:.2f}%")
if total_honest_submissions > 0:
    print(f"False Positive Rate: {total_honest_rejections/total_honest_submissions*100:.2f}%")
