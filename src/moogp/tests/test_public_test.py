"""Tests for the optional public ``moogp.test()`` helper."""

import importlib
import sys

import pytest


def test_importing_moogp_does_not_require_pytest_at_runtime(monkeypatch):
    test_module = importlib.import_module("moogp.test")
    monkeypatch.setitem(sys.modules, "pytest", None)

    # Reload simulates importing moogp.test inside a runtime-only environment.
    importlib.reload(test_module)

    with pytest.raises(ImportError, match=r"moogp\.test\(\) requires pytest"):
        test_module.test()
