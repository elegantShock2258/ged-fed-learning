"""
server package
--------------
Contains all server-side components of the Causal PoR federated learning system.

Modules:
    aggregator         -- PoRStrategy: Flower strategy with PoR dual-aggregation
                          (Logic Validator + FedAvg weight aggregation).
    generate_consensus -- Script to generate the initial global consensus graph
                          using server-side reserved data samples.
    logic_validator    -- SimGNN and LogicValidator: the GED-based graph validation
                          mechanism that decides whether to accept or reject clients.
    train_simgnn       -- Script to pre-train the SimGNN Logic Validator on
                          synthetic graph pairs derived from the consensus graph.
"""
