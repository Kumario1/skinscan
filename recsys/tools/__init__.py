"""Batch/offline build tools. This is the ONLY package where network and LLM
client code may live — recsys/ engine modules never touch the network
(enforced by tests/test_no_network.py)."""
