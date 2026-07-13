"""Tests for the experiment sweep command-line interface."""

from __future__ import annotations

import pytest

from ..run import build_parser


def test_var_threshold_flag_is_mutually_exclusive_with_fixed_q():
    parser = build_parser()

    args = parser.parse_args(["--var-threshold", "0.99"])
    assert args.var_threshold == pytest.approx(0.99)
    assert args.q is None

    with pytest.raises(SystemExit):
        parser.parse_args(["--q", "3", "--var-threshold", "0.99"])
