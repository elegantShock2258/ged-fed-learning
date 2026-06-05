"""
run_sim_sequential.py
=====================
A lightweight sequential federated learning runner that bypasses Ray and
Flower's async actor system entirely.  Each client's fit() is called
synchronously in the main process, then the PoRStrategy aggregates and
validates results exactly as in the full simulation.

Use this on memory-constrained systems where Ray's 0.95 threshold OOM
kills workers before training starts.

Usage:
    uv run python run_sim_sequential.py
"""

import os
import gc
import json
import datetime
import copy
import logging
import yaml
import numpy as np
import torch
import gc

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    FitRes,
    Status,
    Code,
)

from server.logic_validator import LogicValidator
from server.aggregator import PoRStrategy
from client.finance_agent import FinanceClient
from adversary.finance_poisoning import FalseTraderNode
from adversary.finance_adversary_pool import ReversedOrderNode, GradientMimicryNode
from adversary.finance_adaptive_adversary import AdaptiveRLAdversary
from adversary.finance_weight_only_adversary import WeightOnlyAdversary
from adversary.sybil_adversary import SybilLeader, SybilGhost

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    config = yaml.safe_load(f)

SIM   = config["simulation"]
AGENT = config.get("agent_env", {})
ADV   = config.get("adversary", {})

NUM_CLIENTS    = SIM["num_clients"]
NUM_ADV        = SIM["num_false_nodes"]
NUM_HONEST     = NUM_CLIENTS - NUM_ADV
NUM_ROUNDS     = SIM["num_rounds"]
LOCAL_EPOCHS   = SIM["local_epochs"]
device_pref = config.get("hardware", {}).get("device", "auto").lower()
if device_pref == "cpu":
    DEVICE = torch.device('cpu')
else:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
SEED           = config.get("global", {}).get("seed", 42)
DS_NAME        = "finance"
MODEL_DIR      = os.path.join("saved_models", DS_NAME)
ADV_TYPE       = ADV.get("type", "all_three")

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Build client list
# ──────────────────────────────────────────────────────────────────────────────
def make_clients():
    clients = []
    trigger_rate = float(ADV.get("trigger_injection_rate", 0.3))

    for i in range(NUM_CLIENTS):
        cid = str(i)
        if i < NUM_HONEST:
            node = FinanceClient(cid, DEVICE)
            node._load_optimizer_state()
            log.info(f"  Honest Finance Node {cid}")
        else:
            adv_idx = i - NUM_HONEST
            fourth  = max(1, NUM_ADV // 4)
            if ADV_TYPE == "temporal_mimicry_only":
                cls = FalseTraderNode
            elif ADV_TYPE == "reversed_order_only":
                cls = ReversedOrderNode
            elif ADV_TYPE == "gradient_mimicry_only":
                cls = GradientMimicryNode
            elif ADV_TYPE == "adaptive_rl_only":
                cls = AdaptiveRLAdversary
            elif ADV_TYPE == "weight_only":
                cls = WeightOnlyAdversary
            elif ADV_TYPE == "sybil_only":
                cls = SybilLeader if adv_idx == 0 else SybilGhost
            elif ADV_TYPE == "all_three":
                third = max(1, NUM_ADV // 3)
                if adv_idx < third:
                    cls = FalseTraderNode
                elif adv_idx < 2 * third:
                    cls = ReversedOrderNode
                else:
                    cls = GradientMimicryNode
            else:
                all_types = [FalseTraderNode, ReversedOrderNode, GradientMimicryNode,
                             AdaptiveRLAdversary, WeightOnlyAdversary]
                cls = all_types[adv_idx % len(all_types)]
            node = cls(cid, DEVICE)
            node.trigger_rate = trigger_rate
            log.info(f"  {cls.__name__} Adversary {cid}")
        clients.append(node)
    return clients


# ──────────────────────────────────────────────────────────────────────────────
# Parameter helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_global_params(clients):
    """Ask client-0 for initial parameters."""
    return clients[0].get_parameters({})


def broadcast(global_ndarrays, clients):
    """Push global weights into every client."""
    for c in clients:
        c.set_parameters(global_ndarrays)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize validator & strategy
# ──────────────────────────────────────────────────────────────────────────────
validator_path = os.path.join(MODEL_DIR, "simgnn_pretrained.pt")
validator = LogicValidator(
    model_path=validator_path,
    threshold=config["core_logic"].get("finance_validator_threshold", 0.07),
)

strategy = PoRStrategy(
    logic_validator=validator,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    initial_parameters=None,
    on_fit_config_fn=lambda server_round: {"epochs": LOCAL_EPOCHS},
)

# ──────────────────────────────────────────────────────────────────────────────
# Main sequential FL loop
# ──────────────────────────────────────────────────────────────────────────────
def run():
    log.info(f"Sequential FL — {NUM_CLIENTS} clients ({NUM_HONEST} Honest, {NUM_ADV} Adv), {NUM_ROUNDS} rounds")

    clients = make_clients()
    global_weights = get_global_params(clients)

    round_logs = []

    for rnd in range(1, NUM_ROUNDS + 1):
        log.info(f"\n══════════════  ROUND {rnd}  ══════════════")

        # 1. Broadcast current global weights
        broadcast(global_weights, clients)

        # 2. Local training — sequential, one client at a time
        # Dual-Gate PoR: broadcast consensus A_global + B̄ alongside standard config
        base_fit_config = {
            "epochs":       LOCAL_EPOCHS,
            "epsilon":      AGENT.get("epsilon", 0.85),
            "server_round": rnd,
        }
        fit_configs = strategy.get_consensus_fit_config(base_fit_config)
        fit_results = []   # (cid, ndarrays, num_examples, metrics)

        for client in clients:
            try:
                params_in = ndarrays_to_parameters(global_weights)
                fit_ins   = FitIns(parameters=params_in, config=fit_configs)

                # FinanceClient / adversary nodes implement fit() directly
                params_out, num_examples, metrics = client.fit(global_weights, fit_configs)

                fit_results.append((client.cid, params_out, num_examples, metrics))
                log.info(f"  ✓ Client {client.cid} trained ({num_examples} steps)")
            except Exception as e:
                log.warning(f"  ✗ Client {client.cid} fit failed: {e}")
            finally:
                gc.collect()

        if not fit_results:
            log.error("  No clients returned results — skipping aggregation.")
            continue

        # 3. PoR validation + aggregation via strategy
        # Build (FitRes, ClientProxy) tuples that PoRStrategy.aggregate_fit expects
        from flwr.server.client_proxy import ClientProxy

        class _FakeProxy(ClientProxy):
            def __init__(self, cid): 
                super().__init__(cid)
            def get_properties(self, ins, timeout, group_id): ...
            def get_parameters(self, ins, timeout, group_id): ...
            def fit(self, ins, timeout, group_id): ...
            def evaluate(self, ins, timeout, group_id): ...
            def reconnect(self, ins, timeout, group_id): ...

        results_for_strategy = []
        for cid, params_out, num_examples, metrics in fit_results:
            fit_res = FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(params_out),
                num_examples=num_examples,
                metrics=metrics,
            )
            results_for_strategy.append((_FakeProxy(cid), fit_res))

        agg_result = strategy.aggregate_fit(
            server_round=rnd,
            results=results_for_strategy,
            failures=[],
        )

        if agg_result is None or agg_result[0] is None:
            log.warning("  Aggregation returned None — all clients rejected.")
            continue

        agg_params, agg_metrics = agg_result
        global_weights = parameters_to_ndarrays(agg_params)
        log.info(f"  Aggregation done | accepted={agg_metrics.get('accepted_clients', '?')}, rejected={agg_metrics.get('rejected_clients', '?')}")

        round_logs.append({
            "round":    rnd,
            "accepted": agg_metrics.get("accepted_clients", -1),
            "rejected": agg_metrics.get("rejected_clients", -1),
        })

        gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # Persist results
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════  SIMULATION COMPLETE  ══════════════")

    # Save final global model
    # (The FinanceTransformerModel state dict is not directly accessible here;
    #  we rely on clients saving their own state inside PoRStrategy._save)
    # Append simulation summary log
    log_file = os.path.join(MODEL_DIR, "simulation_logs.json")
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                logs = json.load(f)
        except Exception:
            pass

    logs.append({
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runner":          "sequential (no Ray)",
        "num_clients":     NUM_CLIENTS,
        "num_false_nodes": NUM_ADV,
        "num_rounds":      NUM_ROUNDS,
        "round_summary":   round_logs,
    })
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

    log.info(f"Simulation log → {log_file}")
    log.info(f"GED scores     → {MODEL_DIR}/ged_scores.json")


if __name__ == "__main__":
    # Validate configuration before starting
    try:
        from server.config_validator import validate_config
        validate_config()
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)
    run()
