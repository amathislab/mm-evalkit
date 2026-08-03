"""Analysis workflows, one per scientific unit of comparison.

A workflow owns the statistical unit it is valid for: ``gait`` compares
normalized gait cycles, ``bimanual`` compares movement repetitions. Datasets
are attached to a workflow by the registry, so several datasets can share one
workflow when their analysis unit genuinely matches.
"""
