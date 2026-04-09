"""
graphs/collect_per_round_data.py
==================================
Patches fed_neat_strategy.py to ALSO save per-round GED scores and
consensus graph snapshots alongside the existing overwrite.

Run this script ONCE before re-running the simulation:
    python graphs/collect_per_round_data.py
    python federated_sim.py   # re-run simulation

After re-running, the graphs/ directory will have:
    saved_models/asia/ged_scores_round_1.json ... ged_scores_round_N.json
    saved_models/asia/consensus_round_1.gpickle ... consensus_round_N.gpickle

These enable exact G19 (consensus Jaccard per round) and G17 (per-round GED).
"""
import sys
import shutil
from pathlib import Path

PROJECT = Path(__file__).parent.parent
SRC     = PROJECT / "server" / "fed_neat_strategy.py"
BACKUP  = SRC.with_suffix(".py.bak")

# ── Patch 1: per-round GED scores save ────────────────────────────────────────
OLD_GED = '        ged_log_path = os.path.join(self.model_dir, "ged_scores.json")'
NEW_GED = '''\
        ged_log_path = os.path.join(self.model_dir, "ged_scores.json")
        ged_round_path = os.path.join(self.model_dir, f"ged_scores_round_{actual_round}.json")'''

OLD_GED_DUMP = '            json.dump({"round": actual_round, "scores": ged_scores}, f, indent=2)'
NEW_GED_DUMP = '''\
            json.dump({"round": actual_round, "scores": ged_scores}, f, indent=2)
        with open(ged_round_path, "w") as f:
            json.dump({"round": actual_round, "scores": ged_scores}, f, indent=2)'''

# ── Patch 2: per-round consensus graph save ────────────────────────────────────
OLD_CG = '        with open(os.path.join(self.model_dir, "consensus_graph.gpickle"), "wb") as f:'
NEW_CG = '''\
        cg_round_path = os.path.join(self.model_dir, f"consensus_round_{actual_round}.gpickle")
        with open(cg_round_path, "wb") as f:
            pickle.dump(self.global_consensus_graph, f)
        with open(os.path.join(self.model_dir, "consensus_graph.gpickle"), "wb") as f:'''

def apply_patches():
    if not SRC.exists():
        print(f"  ✗  Cannot find {SRC}"); return False

    src = SRC.read_text()

    # Check if already patched
    if "ged_round_path" in src and "cg_round_path" in src:
        print("  ✓  Already patched — no changes made.")
        return True

    # Backup
    shutil.copy(SRC, BACKUP)
    print(f"  ✓  Backup saved: {BACKUP.name}")

    patched = src
    changes = 0

    if OLD_GED in patched:
        patched = patched.replace(OLD_GED, NEW_GED, 1)
        changes += 1; print("  ✓  Patch 1a: per-round GED path")
    else:
        print("  ✗  Patch 1a: target not found (may already be patched)")

    if OLD_GED_DUMP in patched:
        patched = patched.replace(OLD_GED_DUMP, NEW_GED_DUMP, 1)
        changes += 1; print("  ✓  Patch 1b: per-round GED dump")
    else:
        print("  ✗  Patch 1b: target not found")

    if OLD_CG in patched:
        patched = patched.replace(OLD_CG, NEW_CG, 1)
        changes += 1; print("  ✓  Patch 2: per-round consensus save")
    else:
        print("  ✗  Patch 2: target not found")

    if changes > 0:
        SRC.write_text(patched)
        print(f"\n  ✓  Patched {changes}/3 locations in {SRC.name}")
        print("\n  ► Next step: re-run the simulation:")
        print("      python federated_sim.py")
        print("\n  ► Then re-generate graphs:")
        print("      python graphs/run_all.py")
    else:
        print("  ✗  No patches applied — check file manually")
        BACKUP.unlink(missing_ok=True)

    return changes > 0

def revert():
    """Revert to original if backup exists."""
    if BACKUP.exists():
        shutil.copy(BACKUP, SRC)
        BACKUP.unlink()
        print(f"  ✓  Reverted {SRC.name} from backup")
    else:
        print("  ✗  No backup found")

if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert()
    else:
        print("\n=== Patching fed_neat_strategy.py for per-round data collection ===\n")
        apply_patches()
