"""
Simple server integration smoke tests.
"""


def test_server_module_functions_exist():
    import server.generate_consensus as gc
    import server.train_simgnn as ts

    assert hasattr(gc, 'generate_global_consensus')
    assert hasattr(ts, 'train_simgnn')
