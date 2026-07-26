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


def test_latent_ell_upper_defaults_to_20_and_accepts_override():
    parser = build_parser()

    assert parser.parse_args([]).latent_ell_upper == pytest.approx(20.0)
    assert parser.parse_args(["--latent-ell-upper", "30"]).latent_ell_upper == pytest.approx(30.0)
