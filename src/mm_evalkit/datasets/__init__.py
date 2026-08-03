"""Readers and ordered identities for the human datasets mm-evalkit compares against.

Each module owns one released dataset: how it is laid out on disk, which
channels it provides, and how its EMG names map onto model actuators. Nothing
here knows about policies, metrics, or figures -- workflows import datasets,
never the other way round.
"""
