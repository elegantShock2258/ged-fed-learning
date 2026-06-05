"""
server/config_validator.py
===========================
Validates params.yaml configuration at startup to catch misconfigurations
before they cause cryptic runtime errors.

Usage:
    from server.config_validator import validate_config
    validate_config()  # raises ValueError with explanation on failure
"""

import os
import yaml
import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

# Allowed adversary types (must match client factory routing in federated_sim.py)
VALID_ADVERSARY_TYPES = {
    "all_three",           # FalseTrader + Reversed + Gradient (legacy)
    "temporal_mimicry_only",
    "reversed_order_only",
    "gradient_mimicry_only",
    "adaptive_rl_only",
    "weight_only",
    "sybil_only",
}

VALID_DATASET_TYPES = {"cyberdefend", "finance", "asia", "alarm"}

REQUIRED_TOP_KEYS = {
    "core_logic", "server", "hardware", "simulation",
    "agent_env", "global", "adversary",
}

REQUIRED_CORE_LOGIC_KEYS = {
    "causal_edge_threshold", "validator_threshold",
    "consensus_momentum", "finance_validator_threshold",
    "coverage_gate_min_queries",
}


def _load_config(path: str = "params.yaml") -> Dict[str, Any]:
    """Load params.yaml, raising a clear error if it doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"params.yaml not found at {os.path.abspath(path)}. "
            f"Run from the project root directory."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config(path: str = "params.yaml") -> Dict[str, Any]:
    """
    Validate params.yaml and return the config dict.

    Checks performed:
      - File exists and is valid YAML
      - All required top-level keys are present
      - Core logic keys are present with sane values
      - Adversary type is a known value
      - Dataset type is valid
      - Numeric ranges are reasonable

    Raises ValueError with a specific message on failure.
    """
    cfg = _load_config(path)
    errors = []

    # --- Top-level keys ---
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            errors.append(f"Missing top-level key: '{key}'")

    # --- Core logic ---
    cl = cfg.get("core_logic", {})
    for key in REQUIRED_CORE_LOGIC_KEYS:
        if key not in cl:
            errors.append(f"Missing core_logic key: '{key}'")

    # Numeric sanity checks
    def _check_range(section, key, lo, hi, inclusive=True):
        val = cfg.get(section, {}).get(key)
        if val is not None:
            if inclusive and not (lo <= val <= hi):
                errors.append(f"{section}.{key}={val} is outside [{lo}, {hi}]")
            elif not inclusive and not (lo < val < hi):
                errors.append(f"{section}.{key}={val} is outside ({lo}, {hi})")

    _check_range("core_logic", "causal_edge_threshold", 0.0, 1.0)
    _check_range("core_logic", "validator_threshold", 0.0, 1.0)
    _check_range("core_logic", "consensus_momentum", 0.0, 1.0)
    _check_range("core_logic", "finance_validator_threshold", 0.0, 1.0)
    _check_range("core_logic", "coverage_gate_min_queries", 0, 33)
    _check_range("core_logic", "ced_threshold", 0.0, 1.0)
    _check_range("core_logic", "grace_period_rounds", 0, 50)
    _check_range("core_logic", "adaptive_k_sigma", 0.1, 10.0)

    _check_range("simulation", "num_clients", 1, 1000)
    _check_range("simulation", "num_false_nodes", 0, 999)
    _check_range("simulation", "num_rounds", 1, 1000)
    _check_range("simulation", "local_epochs", 1, 100)
    _check_range("simulation", "batch_size", 1, 4096)
    _check_range("simulation", "client_lr", 1e-6, 1.0)

    _check_range("agent_env", "epsilon", 0.0, 1.0)
    _check_range("agent_env", "gamma", 0.0, 1.0)
    _check_range("agent_env", "ent_coef", 0.0, 1.0)
    _check_range("agent_env", "vf_coef", 0.0, 10.0)

    _check_range("adversary", "trigger_injection_rate", 0.0, 1.0)

    # --- Adversary type ---
    adv_type = cfg.get("adversary", {}).get("type", "")
    if adv_type not in VALID_ADVERSARY_TYPES:
        errors.append(
            f"adversary.type='{adv_type}' is not a known type. "
            f"Valid: {sorted(VALID_ADVERSARY_TYPES)}"
        )

    # --- Dataset type ---
    ds_type = cfg.get("simulation", {}).get("dataset_type", "")
    if ds_type not in VALID_DATASET_TYPES:
        errors.append(
            f"simulation.dataset_type='{ds_type}' is not valid. "
            f"Valid: {sorted(VALID_DATASET_TYPES)}"
        )

    # --- Consistency checks ---
    num_clients = cfg["simulation"]["num_clients"]
    num_false = cfg["simulation"]["num_false_nodes"]
    if num_false >= num_clients:
        errors.append(
            f"num_false_nodes ({num_false}) must be < num_clients ({num_clients})"
        )

    if adv_type == "sybil_only" and num_false < 2:
        errors.append("sybil_only requires at least 2 false nodes (leader + 1 ghost)")

    # --- Report ---
    if errors:
        msg = "params.yaml validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)

    log.info("params.yaml validation passed ✓")
    return cfg


# Standalone check
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        validate_config()
        print("Configuration is valid.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)
