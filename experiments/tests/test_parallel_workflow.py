"""Tests for the per-cell parallel workflow (emit_job_list + run_one + merge)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from .. import benchmark_lib, run_one
from ..benchmark_lib import (
    ExperimentConfig,
    RESULT_COLUMNS,
    emit_job_list,
    iter_job_cells,
    per_job_csv_name,
    write_single_row_csv,
)
from ..merge_results import merge_per_job_csvs


def _make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        functions=("borehole", "forrester_mixed"),
        methods=("MOOGP", "LCGP"),
        sample_sizes=(50, 100),
        output_dims=(4,),
        reps=2,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=2026,
        results_dir=tmp_path,
    )


def test_iter_job_cells_visits_full_grid_once():
    config = ExperimentConfig(
        functions=("borehole",),
        methods=("MOOGP", "LCGP"),
        sample_sizes=(50, 100),
        output_dims=(4, 8),
        reps=3,
        n_test=5,
        q=2,
        maxiter=5,
        jitter=1e-6,
        noise_var_frac=1e-2,
        use_fast=True,
        jobs=1,
        base_seed=1,
        results_dir=Path("."),
    )
    cells = list(iter_job_cells(config))

    expected = (
        len(config.functions)
        * len(config.methods)
        * len(config.sample_sizes)
        * len(config.output_dims)
        * config.reps
    )
    assert len(cells) == expected
    assert len(set(cells)) == expected  # no duplicates
    for function, method, n, p, rep in cells:
        assert function in config.functions
        assert method in config.methods
        assert n in config.sample_sizes
        assert p in config.output_dims
        assert 1 <= rep <= config.reps


def test_emit_job_list_writes_config_metadata_and_jobs(tmp_path: Path):
    config = _make_config(tmp_path)
    jobs_path = tmp_path / "jobs.txt"

    summary = emit_job_list(config=config, jobs_path=jobs_path)

    assert summary["jobs_path"] == jobs_path
    assert summary["output_dir"] == tmp_path / "jobs"
    assert summary["n_jobs"] == sum(1 for _ in iter_job_cells(config))

    config_payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    restored = ExperimentConfig.from_metadata(config_payload)
    assert restored == config

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["runs"][-1]["run_id"] == summary["run_id"]
    assert metadata["runs"][-1]["job_list_path"] == str(jobs_path)

    lines = jobs_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["n_jobs"]
    for line in lines:
        run_id, function, method, n_str, p_str, rep_str = line.split()
        assert run_id == summary["run_id"]
        assert function in config.functions
        assert method in config.methods
        int(n_str), int(p_str), int(rep_str)


def test_write_single_row_csv_has_canonical_schema(tmp_path: Path):
    out = tmp_path / "out.csv"
    row = {column: None for column in RESULT_COLUMNS}
    row.update({"run_id": "run_1", "function": "borehole", "method": "MOOGP", "n": 50, "p": 4})

    write_single_row_csv(out, row)
    write_single_row_csv(out, row)  # idempotent rewrite

    with out.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RESULT_COLUMNS
        records = list(reader)
    assert len(records) == 1
    assert records[0]["run_id"] == "run_1"


def test_run_one_main_writes_one_row(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    summary = emit_job_list(config=config, jobs_path=tmp_path / "jobs.txt")

    captured: dict[str, object] = {}

    def fake_run_single_method_job(*, run_id, function, method, n, p, rep, seed_data, config):
        captured["call"] = {
            "run_id": run_id,
            "function": function,
            "method": method,
            "n": n,
            "p": p,
            "rep": rep,
            "seed_data": seed_data,
        }
        return {column: None for column in RESULT_COLUMNS} | {
            "run_id": run_id,
            "function": function,
            "method": method,
            "n": n,
            "p": p,
            "q": 2,
            "rep": rep,
            "seed_data": seed_data,
            "seed_model": 42,
            "status": "ok",
            "error": "",
            "train_time_sec": 0.5,
            "pred_time_sec": 0.1,
            "rmse": 0.123,
        }

    monkeypatch.setattr(run_one, "run_single_method_job", fake_run_single_method_job)

    argv = [
        "experiments.run_one",
        "--config", str(tmp_path / "config.json"),
        "--run-id", summary["run_id"],
        "--function", "borehole",
        "--method", "MOOGP",
        "--n", "50",
        "--p", "4",
        "--rep", "1",
        "--output-dir", str(tmp_path / "jobs"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = run_one.main()
    assert rc == 0

    out_file = tmp_path / "jobs" / per_job_csv_name("borehole", "MOOGP", 50, 4, 1)
    assert out_file.exists()
    with out_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RESULT_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["function"] == "borehole"
    assert rows[0]["method"] == "MOOGP"

    expected_seed = benchmark_lib.stable_seed(config.base_seed, "borehole", 50, 4, 1, "data")
    assert captured["call"]["seed_data"] == expected_seed


def test_run_one_skip_existing_does_not_invoke_job(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    summary = emit_job_list(config=config, jobs_path=tmp_path / "jobs.txt")

    out_file = tmp_path / "jobs" / per_job_csv_name("borehole", "MOOGP", 50, 4, 1)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("preexisting\n", encoding="utf-8")

    def boom(**_kwargs):
        raise AssertionError("run_single_method_job should be skipped when --skip-existing is set.")

    monkeypatch.setattr(run_one, "run_single_method_job", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiments.run_one",
            "--config", str(tmp_path / "config.json"),
            "--run-id", summary["run_id"],
            "--function", "borehole",
            "--method", "MOOGP",
            "--n", "50",
            "--p", "4",
            "--rep", "1",
            "--output-dir", str(tmp_path / "jobs"),
            "--skip-existing",
        ],
    )

    assert run_one.main() == 0
    assert out_file.read_text(encoding="utf-8") == "preexisting\n"


def test_merge_per_job_csvs_concatenates_rows(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    base_row = {column: None for column in RESULT_COLUMNS}

    for idx, method in enumerate(("MOOGP", "LCGP", "PUQ")):
        row = base_row | {
            "run_id": "run_1",
            "function": "borehole",
            "method": method,
            "n": 50,
            "p": 4,
            "q": 2,
            "rep": 1,
            "seed_data": idx,
            "seed_model": idx + 100,
            "status": "ok",
            "rmse": 0.1 * (idx + 1),
        }
        write_single_row_csv(jobs_dir / per_job_csv_name("borehole", method, 50, 4, 1), row)

    out = tmp_path / "results.csv"
    summary = merge_per_job_csvs(jobs_dir, out)

    assert summary == {"merged_files": 3, "rows_written": 3}
    with out.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RESULT_COLUMNS
        rows = list(reader)
    assert sorted(row["method"] for row in rows) == ["LCGP", "MOOGP", "PUQ"]


def test_merge_per_job_csvs_rejects_unknown_schema(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    bad = jobs_dir / "broken.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        merge_per_job_csvs(jobs_dir, tmp_path / "results.csv")


def test_run_one_sh_invokes_python_module_with_correct_args(tmp_path: Path):
    """End-to-end shape check on run_one.sh — uses a stub Python that records argv."""

    fake_python = tmp_path / "fake-python"
    log_path = tmp_path / "argv.json"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "python3 -c \"import json, sys; "
        f"json.dump(sys.argv[1:], open(r'{log_path}', 'w'))\" \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "jobs"

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "run_one.sh"

    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "MOOGP_REPO_ROOT": str(repo_root),
        "MOOGP_PYTHON": str(fake_python),
        "MOOGP_CONFIG": str(config_path),
        "MOOGP_OUTPUT_DIR": str(output_dir),
    }
    completed = subprocess.run(
        [str(script), "run_xyz", "borehole", "MOOGP", "50", "4", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.returncode == 0

    recorded = json.loads(log_path.read_text(encoding="utf-8"))
    assert recorded[:3] == ["-m", "experiments.run_one", "--config"]
    assert "--run-id" in recorded and recorded[recorded.index("--run-id") + 1] == "run_xyz"
    assert "--function" in recorded and recorded[recorded.index("--function") + 1] == "borehole"
    assert "--method" in recorded and recorded[recorded.index("--method") + 1] == "MOOGP"
    assert "--n" in recorded and recorded[recorded.index("--n") + 1] == "50"
    assert "--p" in recorded and recorded[recorded.index("--p") + 1] == "4"
    assert "--rep" in recorded and recorded[recorded.index("--rep") + 1] == "1"
    assert "--output-dir" in recorded
    assert recorded[recorded.index("--output-dir") + 1] == str(output_dir)
