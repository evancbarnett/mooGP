import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import benchmark_lib
from ..benchmark_lib import (
    ExperimentConfig,
    PredictionBundle,
    build_dataset_bundle,
    compute_metrics,
    fit_method_local,
    make_latent_theta0_and_bounds,
    method_python_executable,
    normalize_covariance,
    run_benchmarks,
    run_single_method_job,
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


def test_make_latent_theta0_and_bounds_matches_forrester_defaults():
    theta0, bounds = make_latent_theta0_and_bounds(q=2, d=3, seed_model=999)

    expected = np.array(
        [
            np.log(1.0),
            np.log(0.5),
            np.log(0.5),
            np.log(0.5),
            np.log(1.0),
            np.log(0.5),
            np.log(0.5),
            np.log(0.5),
        ]
    )

    assert np.allclose(theta0, expected)
    assert len(bounds) == theta0.size


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

    summary = run_benchmarks(config)
    contents = (tmp_path / "results.csv").read_text(encoding="utf-8")

    assert len(summary["rows"]) == 2
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


def test_method_python_executable_uses_method_specific_interpreters(tmp_path: Path):
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP", "OILMM", "LCGP"),
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

    assert method_python_executable("MOOGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("MOGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("LCGP", config) == Path("/tmp/moogp-python")
    assert method_python_executable("OILMM", config) == Path("/tmp/oilmm-python")
    assert method_python_executable("PUQ", config) is None


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
            "rep": payload["rep"],
            "seed_data": payload["seed_data"],
            "seed_model": 12345,
            "status": "ok",
            "error": "",
            "train_time_sec": 0.1,
            "pred_time_sec": 0.2,
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
    assert row["status"] == "ok"
    assert calls["cmd"][0] == str(oilmm_python)
    assert calls["cmd"][1:] == ["-m", "experiments.method_runner"]


def test_fit_method_local_lcgp_returns_prediction_bundle(tmp_path: Path):
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

    predictor = fit_method_local(method="LCGP", bundle=bundle, seed_model=456, config=config)
    prediction = predictor.predict(bundle.test_X_scaled)

    assert predictor.status in {"ok", "opt_failed"}
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
