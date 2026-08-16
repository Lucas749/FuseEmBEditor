"""Compatibility adapter for the original FORECasT-BE sklearn models."""

import sys
import types
import warnings

import numpy as np
from sklearn.base import InconsistentVersionWarning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, _tree


class _LegacyLoss:
    pass


def _install_loss_module():
    module = types.ModuleType("sklearn.ensemble._gb_losses")
    for name in (
        "LeastSquaresError",
        "LeastAbsoluteError",
        "HuberLossFunction",
        "QuantileLossFunction",
    ):
        setattr(module, name, _LegacyLoss)
    sys.modules[module.__name__] = module


def _install_tree_loader():
    original_tree = _tree.Tree

    class CompatibleTree(original_tree):
        def __setstate__(self, state):
            nodes = state.get("nodes")
            if nodes is not None and nodes.dtype != _tree.NODE_DTYPE:
                converted = np.zeros(nodes.shape, dtype=_tree.NODE_DTYPE)
                for name in nodes.dtype.names:
                    converted[name] = nodes[name]
                state["nodes"] = converted
            super().__setstate__(state)

    _tree.Tree = CompatibleTree


def _install_estimator_loaders():
    original_gradient_setstate = GradientBoostingRegressor.__setstate__
    original_tree_setstate = DecisionTreeRegressor.__setstate__

    def gradient_setstate(self, state):
        original_gradient_setstate(self, state)
        if not hasattr(self, "_loss"):
            if self.loss == "ls":
                self.loss = "squared_error"
            elif self.loss == "lad":
                self.loss = "absolute_error"
            self._loss = self._get_loss(sample_weight=None)

    def tree_setstate(self, state):
        original_tree_setstate(self, state)
        if not hasattr(self, "monotonic_cst"):
            self.monotonic_cst = None

    GradientBoostingRegressor.__setstate__ = gradient_setstate
    DecisionTreeRegressor.__setstate__ = tree_setstate


def install_forecast_sklearn_compatibility():
    """Enable loading FORECasT-BE models saved with sklearn 0.22."""
    if "sklearn.ensemble._gb_losses" in sys.modules:
        return
    warnings.filterwarnings("once", category=InconsistentVersionWarning)
    _install_loss_module()
    _install_tree_loader()
    _install_estimator_loaders()
