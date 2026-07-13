import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .. import benchmark_lib
from ..benchmark_lib import (
    DatasetBundle,
    ExperimentConfig,
    FittedPredictor,
    PredictionBundle,
    build_dataset_bundle,
    compute_metrics,
    fit_method_local,
    method_python_executable,
    normalize_covariance,
    run_benchmarks,
    run_single_method_job,
    run_single_method_local,
)


def test_build_borehole_bundle_reuses_output_locations_and_shapes():
    bundle = build_dataset_bundle(function="borehole", n=8, p=4, n_test=6, seed_data=123)
    bundle_same = build_dataset_bundle(function="borehole", n=8, p=4, n_test=6, seed_data=123)

    assert bundle.train_data["X_scaled"].shape == (8, 4)
    assert bundle.train_data["Y"].shape == (8, 4)
    assert bundle.test_X_scaled.shape == (6, 4)
    assert bundle.test_Y_true.shape == (6, 4)
    assert bundle.extra["locations_phys"].shape == (4, 2)
    assert not np.allclose(bundle.train_data["Y"], bundle.extra["Y_train_clean"])
    assert np.allclose(bundle.train_data["Y"], bundle_same.train_data["Y"])
    assert np.allclose(bundle.test_Y_true, bundle_same.test_Y_true)


def test_run_benchmarks_parallel_path_writes_rows_for_stub_methods(tmp_path: Path):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("LCGP", "PUQ"),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=10,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=2,
        base_seed=123,
        results_dir=tmp_path,
    )

    def fake_run_single_method_job(run_id, function, method, n, p, rep, seed_data, config):
        return {
            "run_id": run_id,
            "function": function,
            "method": method,
            "n": n,
            "p": p,
            "q": 2,
            "rep": rep,
            "seed_data": seed_data,
            "seed_model": 12345,
            "status": "not_implemented" if method == "PUQ" else "ok",
            "error": "stubbed" if method == "PUQ" else "",
            "train_time_sec": 0.01,
            "pred_time_sec": 0.02,
            "nit": None,
            "njev": None,
            "nfev": None,
            "rmse": 0.3,
            "coverage_95": 0.95,
            "interval_len_95": 1.2,
            "dss_diag": 0.4,
            "dss_full": None,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(benchmark_lib, "run_single_method_job", fake_run_single_method_job)
    try:
        summary = run_benchmarks(config)
    finally:
        monkeypatch.undo()
    contents = (tmp_path / "results.csv").read_text(encoding="utf-8")

    assert len(summary["rows"]) == 2
    assert all(row["q"] == 2 for row in summary["rows"])
    assert "q" in contents.splitlines()[0].split(",")
    assert "not_implemented" in contents


def test_experiment_config_round_trips_python_paths(tmp_path: Path):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP", "OILMM"),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
        moogp_python=Path("/tmp/moogp-python"),
        oilmm_python=Path("/tmp/oilmm-python"),
    )

    restored = ExperimentConfig.from_metadata(config.to_metadata())

    assert restored.moogp_python == Path("/tmp/moogp-python")
    assert restored.oilmm_python == Path("/tmp/oilmm-python")


def test_experiment_config_round_trips_var_threshold(tmp_path: Path):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP", "LCGP"),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=None,
        var_threshold=0.8,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )

    restored = ExperimentConfig.from_metadata(config.to_metadata())

    assert restored == config
    assert restored.q is None
    assert restored.var_threshold == pytest.approx(0.8)


def test_var_threshold_selects_rank_and_passes_it_to_method_adapter(
    tmp_path: Path,
    monkeypatch,
):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("LCGP",),
        sample_sizes=(4,),
        output_dims=(3,),
        reps=1,
        n_test=2,
        q=None,
        var_threshold=0.8,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    first_component = np.array([-1.0, -1.0, 1.0, 1.0])
    second_component = np.array([-1.0, 1.0, -1.0, 1.0])
    y_train = np.column_stack(
        [first_component, first_component, second_component]
    )
    bundle = DatasetBundle(
        function="borehole",
        n=4,
        p=3,
        seed_data=456,
        train_data={
            "X_scaled": np.arange(4, dtype=float)[:, None],
            "Y": y_train,
            "y": y_train,
        },
        test_X_scaled=np.array([[0.25], [0.75]]),
        test_Y_true=np.zeros((2, 3), dtype=float),
    )
    captured: dict[str, ExperimentConfig] = {}

    def fake_fit_method_local(*, method, bundle, seed_model, config):
        captured[method] = config
        return FittedPredictor(
            predict_fn=lambda Xstar: PredictionBundle(
                mean=np.zeros((np.asarray(Xstar).shape[0], 3), dtype=float),
                std=np.ones((np.asarray(Xstar).shape[0], 3), dtype=float),
            ),
            status="ok",
            train_time_sec=0.1,
        )

    monkeypatch.setattr(benchmark_lib, "fit_method_local", fake_fit_method_local)
    monkeypatch.setattr(benchmark_lib, "normalized_rmse", lambda *_args: 0.0)
    monkeypatch.setattr(
        benchmark_lib,
        "compute_metrics",
        lambda *_args, **_kwargs: {
            "rmse": 0.0,
            "normalized_rmse": 0.0,
            "coverage_95": 1.0,
            "interval_len_95": 1.0,
            "dss_diag": 0.0,
            "dss_full": None,
        },
    )

    row = run_single_method_local(
        run_id="run_threshold",
        function="borehole",
        method="LCGP",
        n=4,
        p=3,
        rep=1,
        bundle=bundle,
        config=config,
    )

    # After column-wise z-scoring, the controlled response has squared
    # singular values 8, 4, 0. The first component explains 2/3, so a 0.8
    # threshold must select q=2 using the strict cumulative-variance rule.
    assert row["q"] == 2
    assert row["var_threshold"] == pytest.approx(0.8)
    assert captured["LCGP"].q == 2
    assert captured["LCGP"].var_threshold is None


def test_method_python_executable_uses_method_specific_interpreters(tmp_path: Path):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP", "OILMM", "LCGP", "PUQ"),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
        moogp_python=Path("/tmp/moogp-python"),
        oilmm_python=Path("/tmp/oilmm-python"),
        puq_python=Path("/tmp/puq-python"),
    )

    assert method_python_executable("MOOGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("MOGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("LCGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("OILMM", config) == Path("/tmp/oilmm-python")
    assert method_python_executable("PUQ", config) == Path("/tmp/puq-python")


def test_run_single_method_job_dispatches_oilmm_to_configured_python(tmp_path: Path, monkeypatch):
    oilmm_python = tmp_path / "python-oilmm"
    oilmm_python.write_text("#!/bin/sh\n", encoding="utf-8")
    oilmm_python.chmod(0o755)

    config = ExperimentConfig(
        functions=("borehole",),
        methods=("OILMM",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
        moogp_python=Path(sys.executable),
        oilmm_python=oilmm_python,
    )

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        payload = json.loads(kwargs["input"])
        row = {
            "run_id": payload["run_id"],
            "function": payload["function"],
            "method": payload["method"],
            "n": payload["n"],
            "p": payload["p"],
            "q": 2,
            "rep": payload["rep"],
            "seed_data": payload["seed_data"],
            "seed_model": 12345,
            "status": "ok",
            "error": "",
            "train_time_sec": 0.1,
            "pred_time_sec": 0.2,
            "nit": None,
            "njev": None,
            "nfev": None,
            "rmse": 0.3,
            "coverage_95": 0.95,
            "interval_len_95": 1.2,
            "dss_diag": 0.4,
            "dss_full": None,
        }
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(row), stderr="")

    monkeypatch.setattr(benchmark_lib.subprocess, "run", fake_run)

    row = run_single_method_job(
        run_id="run_123",
        function="borehole",
        method="OILMM",
        n=8,
        p=3,
        rep=1,
        seed_data=456,
        config=config,
    )

    assert row["method"] == "OILMM"
    assert row["q"] == 2
    assert row["status"] == "ok"
    assert calls["cmd"][0] == str(oilmm_python)
    assert calls["cmd"][1:] == ["-m", "experiments.method_runner"]


def test_fit_method_local_lcgp_returns_prediction_bundle(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("LCGP",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=2,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=3, n_test=5, seed_data=123)

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.random = SimpleNamespace(set_seed=lambda seed: None)

    class FakeLCGP:
        def __init__(self, *, y, x, q, verbose=False):
            self.y = y
            self.x = x
            self.q = q
            self.verbose = verbose
            self.loss = lambda: 0.0
            self.trainable_variables = []

        def predict(self, Xstar):
            nstar = np.asarray(Xstar, dtype=float).shape[0]
            p = self.y.shape[0]
            mean = np.full((p, nstar), 1.5, dtype=float)
            var = np.full((p, nstar), 0.25, dtype=float)
            return np.stack([mean, var], axis=0)

    class FakeScipyOptimizer:
        def minimize(self, loss, variables, options=None):
            return SimpleNamespace(success=True, message="stub optimizer")

    fake_gpflow = types.ModuleType("gpflow")
    fake_gpflow.optimizers = SimpleNamespace(Scipy=lambda: FakeScipyOptimizer())

    fake_lcgp = types.ModuleType("lcgp")
    fake_lcgp.LCGP = FakeLCGP

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "gpflow", fake_gpflow)
    monkeypatch.setitem(sys.modules, "lcgp", fake_lcgp)

    predictor = fit_method_local(method="LCGP", bundle=bundle, seed_model=456, config=config)
    prediction = predictor.predict(bundle.test_X_scaled)

    assert predictor.status in {"ok", "opt_failed"}
    assert predictor.train_time_sec is not None
    assert prediction.mean.shape == bundle.test_Y_true.shape
    assert prediction.std is not None
    assert prediction.std.shape == bundle.test_Y_true.shape
    assert np.all(np.isfinite(prediction.mean))
    assert np.all(prediction.std > 0.0)


def test_fit_method_local_oilmm_uses_benchmark_iteration_budget(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("OILMM",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=7,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=3, n_test=5, seed_data=123)

    captured: dict[str, object] = {}

    fake_tf = types.SimpleNamespace(
        float32=np.float32,
        random=SimpleNamespace(set_seed=lambda seed: captured.setdefault("tf_seed", seed)),
    )

    class FakePosterior:
        def predict(self, Xstar):
            nstar = np.asarray(Xstar, dtype=float).shape[0]
            p = bundle.test_Y_true.shape[1]
            mean = np.full((nstar, p), 1.0, dtype=float)
            var = np.full((nstar, p), 0.25, dtype=float)
            return mean, var

    class FakeOILMM:
        def __init__(self, dtype, build_latent_processes, num_outputs):
            captured["dtype"] = dtype
            captured["num_outputs"] = num_outputs
            self.posterior = FakePosterior()

        def fit(self, X_fit, Y_train, **kwargs):
            captured["fit_kwargs"] = dict(kwargs)
            captured["fit_shape"] = (np.asarray(X_fit).shape, np.asarray(Y_train).shape)

        def condition(self, X_fit, Y_train):
            return self.posterior

    def fake_runtime():
        captured["runtime_loaded"] = True
        _ = benchmark_lib.time.perf_counter()
        return {
            "tf": fake_tf,
            "dtype": np.float32,
            "OILMM": FakeOILMM,
            "EQ": lambda: SimpleNamespace(stretch=lambda *_args, **_kwargs: None),
            "GP": lambda *_args, **_kwargs: None,
            "to_numpy": lambda z: np.asarray(z, dtype=float),
        }

    perf_counter_values = iter([1.0, 10.0, 10.4])
    monkeypatch.setattr(benchmark_lib, "_get_oilmm_runtime", fake_runtime)
    monkeypatch.setattr(benchmark_lib.time, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(benchmark_lib.np.random, "seed", lambda seed: captured.setdefault("np_seed", seed))

    predictor = fit_method_local(method="OILMM", bundle=bundle, seed_model=456, config=config)
    prediction = predictor.predict(bundle.test_X_scaled)

    assert predictor.status == "ok"
    assert predictor.train_time_sec == pytest.approx(0.4)
    assert predictor.nit is None
    assert predictor.njev is None
    assert predictor.nfev is None
    assert captured["runtime_loaded"] is True
    assert captured["np_seed"] == 456
    assert captured["tf_seed"] == 456
    assert captured["num_outputs"] == bundle.test_Y_true.shape[1]
    assert captured["fit_kwargs"] == {"trace": False, "jit": False, "iters": config.maxiter}
    assert prediction.mean.shape == bundle.test_Y_true.shape
    assert prediction.std is not None
    assert prediction.std.shape == bundle.test_Y_true.shape


def test_fit_method_local_puq_returns_prediction_bundle(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("PUQ",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=4,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=3, n_test=5, seed_data=123)

    captured: dict[str, object] = {}

    class FakePUQPrediction:
        def __init__(self, info):
            self._info = info

    class FakePUQEmulator:
        def __init__(self, *, x, theta, f, method, args):
            captured["x_shape"] = np.asarray(x).shape
            captured["theta_shape"] = np.asarray(theta).shape
            captured["f_shape"] = np.asarray(f).shape
            captured["method"] = method
            captured["args"] = dict(args)
            self._x = np.asarray(x, dtype=float)
            self._p = np.asarray(f).shape[1]

        def predict(self, x, thetaprime=None):
            captured["thetaprime"] = thetaprime
            x = np.asarray(x, dtype=float)
            mean = np.full((self._p, x.shape[0]), 0.5, dtype=float)
            var = np.full((self._p, x.shape[0]), 0.25, dtype=float)
            nugs = np.full((self._p, x.shape[0]), 0.04, dtype=float)
            return FakePUQPrediction({"mean": mean, "var": var, "nugs": nugs})

    fake_surrogate = types.ModuleType("PUQ.surrogate")
    fake_surrogate.emulator = FakePUQEmulator
    fake_puq_pkg = types.ModuleType("PUQ")
    monkeypatch.setitem(sys.modules, "PUQ", fake_puq_pkg)
    monkeypatch.setitem(sys.modules, "PUQ.surrogate", fake_surrogate)

    predictor = fit_method_local(method="PUQ", bundle=bundle, seed_model=456, config=config)
    prediction = predictor.predict(bundle.test_X_scaled)

    assert predictor.status == "ok"
    assert predictor.train_time_sec is not None
    assert predictor.nit is None
    assert captured["method"] == "multihetGP"
    assert captured["args"]["maxit"] == config.maxiter
    assert captured["x_shape"] == bundle.train_data["X_scaled"].shape
    assert captured["f_shape"] == bundle.train_data["Y"].shape
    assert captured["theta_shape"][0] == bundle.train_data["Y"].shape[1]
    assert captured["thetaprime"] is None
    assert prediction.mean.shape == bundle.test_Y_true.shape
    assert prediction.std is not None
    assert prediction.std.shape == bundle.test_Y_true.shape
    assert prediction.cov is None
    assert np.all(prediction.std > 0.0)
    # Predictive std must combine the mean-process variance and the
    # observation-noise (nugget): sqrt(var + nugs) = sqrt(0.25 + 0.04).
    assert np.allclose(prediction.std, np.sqrt(0.29))


def test_fit_method_local_puq_raises_with_install_hint_when_missing(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("PUQ",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=4,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=3, n_test=5, seed_data=123)

    monkeypatch.delitem(sys.modules, "PUQ", raising=False)
    monkeypatch.delitem(sys.modules, "PUQ.surrogate", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("PUQ"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        fit_method_local(method="PUQ", bundle=bundle, seed_model=456, config=config)

    message = str(excinfo.value)
    assert "PUQ" in message
    assert "pip install" in message
    assert "hetGPy" in message


def test_puq_smoke_fit_and_predict_on_toy_multioutput_dataset(tmp_path: Path):
    pytest.importorskip("PUQ")
    pytest.importorskip("hetgpy")

    config = ExperimentConfig(
        functions=("forrester_mixed",),
        methods=("PUQ",),
        sample_sizes=(20,),
        output_dims=(3,),
        reps=1,
        n_test=10,
        q=2,
        maxiter=10,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=2026,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(
        function="forrester_mixed",
        n=20,
        p=3,
        n_test=10,
        seed_data=2026,
    )

    predictor = fit_method_local(method="PUQ", bundle=bundle, seed_model=2026, config=config)
    prediction = predictor.predict(bundle.test_X_scaled)

    assert predictor.status == "ok"
    assert predictor.train_time_sec is not None
    assert prediction.mean.shape == bundle.test_Y_true.shape
    assert prediction.std is not None
    assert prediction.std.shape == bundle.test_Y_true.shape
    assert np.all(np.isfinite(prediction.mean))
    assert np.all(prediction.std > 0.0)


def test_compute_metrics_accepts_diag_and_full_covariance_shapes():
    y_true = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    y_mean = y_true + 0.1
    y_std = np.full_like(y_true, 0.5)
    cov_npp = np.repeat(np.eye(2, dtype=float)[None, :, :] * 0.25, repeats=3, axis=0)

    metrics = compute_metrics(
        y_true,
        PredictionBundle(mean=y_mean, std=y_std, cov=cov_npp),
    )
    cov_ppn = normalize_covariance(np.moveaxis(cov_npp, 0, -1), n=3, p=2)
    metrics_ppn = compute_metrics(
        y_true,
        PredictionBundle(mean=y_mean, std=y_std, cov=np.moveaxis(cov_ppn, 0, -1)),
    )

    assert metrics["rmse"] > 0.0
    assert metrics["coverage_95"] is not None
    assert metrics["interval_len_95"] is not None
    assert metrics["dss_diag"] is not None
    assert metrics["dss_full"] is not None
    assert np.isclose(metrics["dss_full"], metrics_ppn["dss_full"])


def _capture_moogp_like_fit_call(method: str, bundle, config, monkeypatch):
    """Run ``_fit_moogp_like`` with MOOGP.__init__/fit stubbed; return (model, fit_kwargs)."""

    captured_models: list[object] = []
    captured_fit: dict[str, object] = {}
    from moogp.model import MOOGP as _RealMOOGP
    orig_init = _RealMOOGP.__init__

    def capture_init(self, *init_args, **init_kwargs):
        orig_init(self, *init_args, **init_kwargs)
        captured_models.append(self)

    def fake_fit(self, data, theta0=None, bounds=None, optimizer_opts=None):
        captured_fit["theta0"] = theta0
        captured_fit["bounds"] = bounds
        self.opt_result = SimpleNamespace(success=True, message="")
        self.fitted = True
        return self

    monkeypatch.setattr("moogp.model.MOOGP.__init__", capture_init)
    monkeypatch.setattr("moogp.model.MOOGP.fit", fake_fit)

    predictor = fit_method_local(method=method, bundle=bundle, seed_model=456, config=config)

    assert predictor.status == "ok"
    assert len(captured_models) == 1
    return captured_models[0], captured_fit


def _moogp_smoke_config(method: str, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        functions=("borehole",),
        methods=(method,),
        sample_sizes=(8,),
        output_dims=(4,),
        reps=1,
        n_test=5,
        q=3,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )


def test_fit_method_local_moogp_delegates_data_aware_init_to_model(tmp_path: Path, monkeypatch):
    # The adapter no longer hand-builds theta0/bounds: it omits both so MOOGP.fit
    # derives the data-aware initialization internally from the standardized
    # working-scale outputs. standardize_y="zscore" guarantees the in-model
    # sigma_eps init sees zero-mean / unit-variance Y (the scale guarantee the
    # deleted append_sigma_eps capture test used to protect; now exercised
    # end-to-end in src/moogp/tests/test_auto_init.py).
    config = _moogp_smoke_config("MOOGP", tmp_path)
    bundle = build_dataset_bundle(function="borehole", n=8, p=4, n_test=5, seed_data=123)

    model, fit_kwargs = _capture_moogp_like_fit_call("MOOGP", bundle, config, monkeypatch)

    assert fit_kwargs["theta0"] is None
    assert fit_kwargs["bounds"] is None
    assert model.orthogonal is True
    assert model.standardize_y == "zscore"


def test_fit_method_local_mogp_delegates_data_aware_init_to_model(tmp_path: Path, monkeypatch):
    config = _moogp_smoke_config("MOGP", tmp_path)
    bundle = build_dataset_bundle(function="borehole", n=8, p=4, n_test=5, seed_data=123)

    model, fit_kwargs = _capture_moogp_like_fit_call("MOGP", bundle, config, monkeypatch)

    assert fit_kwargs["theta0"] is None
    assert fit_kwargs["bounds"] is None
    assert model.orthogonal is False
    assert model.standardize_y == "zscore"


def test_fit_method_local_moogp_extracts_optimizer_diagnostics(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP",),
        sample_sizes=(8,),
        output_dims=(4,),
        reps=1,
        n_test=5,
        q=3,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=4, n_test=5, seed_data=123)

    def fake_fit(self, data, theta0=None, bounds=None, optimizer_opts=None):
        self.opt_result = SimpleNamespace(success=True, message="", nit=7, njev=8, nfev=9)
        self.fitted = True
        return self

    monkeypatch.setattr("moogp.model.MOOGP.fit", fake_fit)

    predictor = fit_method_local(method="MOOGP", bundle=bundle, seed_model=456, config=config)

    assert predictor.status == "ok"
    assert predictor.train_time_sec is not None
    assert predictor.nit == 7
    assert predictor.njev == 8
    assert predictor.nfev == 9


def test_run_single_method_local_uses_predictor_train_time_and_keeps_metrics_outside_prediction_timer(
    tmp_path: Path,
    monkeypatch,
):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP",),
        sample_sizes=(8,),
        output_dims=(3,),
        reps=1,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=123,
        results_dir=tmp_path,
    )
    bundle = build_dataset_bundle(function="borehole", n=8, p=3, n_test=5, seed_data=123)

    p_cols = bundle.test_Y_true.shape[1]
    predictor = FittedPredictor(
        predict_fn=lambda Xstar: PredictionBundle(
            mean=np.zeros((np.asarray(Xstar).shape[0], p_cols), dtype=float),
        ),
        status="ok",
        error="",
        train_time_sec=1.25,
        nit=4,
        njev=5,
        nfev=6,
    )

    monkeypatch.setattr(benchmark_lib, "fit_method_local", lambda **_kwargs: predictor)
    monkeypatch.setattr(
        benchmark_lib,
        "compute_metrics",
        lambda y_true, prediction: (
            benchmark_lib.time.perf_counter(),
            {
                "rmse": 0.1,
                "coverage_95": None,
                "interval_len_95": None,
                "dss_diag": None,
                "dss_full": None,
            },
        )[1],
    )
    perf_counter_values = iter([10.0, 10.2, 99.0])
    monkeypatch.setattr(benchmark_lib.time, "perf_counter", lambda: next(perf_counter_values))

    row = run_single_method_local(
        run_id="run_123",
        function="borehole",
        method="MOOGP",
        n=8,
        p=3,
        rep=1,
        bundle=bundle,
        config=config,
    )

    assert row["train_time_sec"] == 1.25
    assert row["pred_time_sec"] == pytest.approx(0.2)
    assert row["nit"] == 4
    assert row["njev"] == 5
    assert row["nfev"] == 6
    assert row["rmse"] == 0.1


def test_append_results_rows_rejects_old_results_schema(tmp_path: Path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("run_id,function,method,n,p,rep\n", encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "current schema"):
        benchmark_lib.append_results_rows(
            results_path,
            [
                {
                    "run_id": "run_1",
                    "function": "borehole",
                    "method": "MOOGP",
                    "n": 8,
                    "p": 3,
                    "q": 2,
                    "rep": 1,
                    "seed_data": 1,
                    "seed_model": 2,
                }
            ],
        )
