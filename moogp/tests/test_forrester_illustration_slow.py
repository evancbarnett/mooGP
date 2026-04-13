import numpy as np
import matplotlib.pyplot as plt

from moogp import forrester_illustration_slow as slow_mod
from moogp.forrester_illustration import (
    plot_forrester_fit_side_by_side,
    plot_trend_recovery_two_designs,
)


class DummyTrendModel:
    def __init__(self, bhat):
        self.cache = {"bhat": np.asarray(bhat, dtype=float)}

    def predict(self, X_scaled, return_std=False):
        x = np.asarray(X_scaled[:, 0], dtype=float)
        p = self.cache["bhat"].shape[1]
        mean = np.column_stack([(j + 1) * x for j in range(p)])
        std = 0.1 * np.ones_like(mean)
        if return_std:
            return mean, std
        return mean


class DummyFigure:
    def savefig(self, *args, **kwargs):
        return None


def make_dummy_data():
    x = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    y = np.column_stack(
        [
            x[:, 0],
            1.0 + 2.0 * x[:, 0],
            -1.0 + 0.5 * x[:, 0],
        ]
    )
    return {
        "X": x,
        "X_scaled": 2.0 * (x - 0.5),
        "y": y,
        "f": y.copy(),
    }


def test_plot_trend_recovery_two_designs_skips_mogp_when_missing():
    data_lhs = make_dummy_data()
    data_log = make_dummy_data()
    moogp_lhs = DummyTrendModel([[0.5, 1.0, -0.5], [0.0, 0.5, 0.25]])
    moogp_log = DummyTrendModel([[0.25, 0.75, -0.25], [0.1, 0.4, 0.2]])

    fig = plot_trend_recovery_two_designs(
        data_lhs,
        moogp_lhs,
        None,
        data_log,
        moogp_log,
        None,
        output_idx=1,
    )

    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "MOOGP" in labels
    assert "MOGP" not in labels
    plt.close(fig)


def test_plot_forrester_fit_side_by_side_shares_y_axis_by_row():
    data_lhs = make_dummy_data()
    data_log = make_dummy_data()
    moogp_lhs = DummyTrendModel([[0.5, 1.0, -0.5], [0.0, 0.5, 0.25]])
    moogp_log = DummyTrendModel([[0.25, 0.75, -0.25], [0.1, 0.4, 0.2]])

    fig = plot_forrester_fit_side_by_side(
        moogp_lhs=moogp_lhs,
        X_lhs=data_lhs["X"],
        Y_lhs=data_lhs["y"],
        moogp_log=moogp_log,
        X_log=data_log["X"],
        Y_log=data_log["y"],
    )

    assert fig.axes[0].get_shared_y_axes().joined(fig.axes[0], fig.axes[1])
    plt.close(fig)


def test_run_forrester_illustration_slow_can_disable_mogp(monkeypatch, tmp_path):
    data = make_dummy_data()
    fit_calls = []

    def fake_generate_forrester_data(*args, **kwargs):
        return data

    def fake_log_lhs_1d_rescaled(*args, **kwargs):
        return data["X"]

    def fake_fit_moogp_forrester(*args, orthogonal, data=None, **kwargs):
        fit_calls.append(orthogonal)
        model = DummyTrendModel([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        return model, data["X"], data["X_scaled"], data["y"]

    def fake_plot_forrester_fit(model, X, X_scaled, Y, n_plot=400, non_ortho_model=None):
        assert non_ortho_model is None
        return DummyFigure()

    def fake_plot_forrester_fit_side_by_side(
        moogp_lhs,
        X_lhs,
        Y_lhs,
        mogp_lhs=None,
        moogp_log=None,
        X_log=None,
        Y_log=None,
        mogp_log=None,
        **kwargs,
    ):
        assert mogp_lhs is None
        assert mogp_log is None
        return DummyFigure()

    def fake_plot_trend_recovery_two_designs(
        data_lhs,
        moogp_lhs,
        mogp_lhs,
        data_log,
        moogp_log,
        mogp_log,
        **kwargs,
    ):
        assert mogp_lhs is None
        assert mogp_log is None
        return DummyFigure()

    def fake_evaluate_moogp(*args, non_ortho_model=None, output_idx=0, **kwargs):
        assert non_ortho_model is None
        return {
            "predictive": {"moogp": {}, "mogp": None, "ols": None},
            "trend_table": {
                "output_idx": output_idx,
                "rows": {
                    "moogp": {"rmspe": 0.0, "beta0": 0.0, "beta1": 0.0},
                    "ols": {"rmspe": 0.0, "beta0": 0.0, "beta1": 0.0},
                },
            },
        }

    monkeypatch.setattr(slow_mod, "generate_forrester_data", fake_generate_forrester_data)
    monkeypatch.setattr(slow_mod, "log_lhs_1d_rescaled", fake_log_lhs_1d_rescaled)
    monkeypatch.setattr(slow_mod, "fit_moogp_forrester", fake_fit_moogp_forrester)
    monkeypatch.setattr(slow_mod, "plot_forrester_fit", fake_plot_forrester_fit)
    monkeypatch.setattr(slow_mod, "plot_forrester_fit_side_by_side", fake_plot_forrester_fit_side_by_side)
    monkeypatch.setattr(slow_mod, "plot_trend_recovery_two_designs", fake_plot_trend_recovery_two_designs)
    monkeypatch.setattr(slow_mod, "evaluate_moogp", fake_evaluate_moogp)
    monkeypatch.setattr(slow_mod, "print_predictive_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(slow_mod, "print_trend_comparison_table", lambda *args, **kwargs: None)

    result = slow_mod.run_forrester_illustration_slow(
        n_train=5,
        seed=0,
        MOGP=False,
        outdir=tmp_path,
    )

    assert fit_calls == [True, True]
    assert result["mogp_lhs"] is None
    assert result["mogp_log"] is None
    assert result["outdir"] == tmp_path
