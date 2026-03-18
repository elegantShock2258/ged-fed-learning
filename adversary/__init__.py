"""
adversary package
-----------------
Contains the FalseNode adversarial client implementation.

FalseNode poisons tabular training data (feature + label corruption)
to mount explanation-poisoning attacks against the PoR defense.
Poisoned clients produce structurally crippled causal graphs that the
PoR Logic Validator catches via high GED distance from the consensus.
"""
