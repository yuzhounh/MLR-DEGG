"""MLR-DEGG: spatiotemporal double-exponent group-regularized multinomial logistic regression.

The implementation follows ``manuscript/manuscript.tex`` (version 0.4).  Equation numbers in
docstrings refer to the labels used there, e.g. ``(eq:bc-update)``.
"""

from .contrast import contrast_matrix
from .estimator import MLRDEGGClassifier
from .graphs import Graph, chain_graph, gaussian_graph, spatiotemporal_operators

__all__ = [
    "Graph",
    "MLRDEGGClassifier",
    "chain_graph",
    "contrast_matrix",
    "gaussian_graph",
    "spatiotemporal_operators",
]

__version__ = "0.4.0"
