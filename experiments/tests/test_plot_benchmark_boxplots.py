from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..plot_benchmark_boxplots import create_metric_figures, prepare_grouped_metric_data


def test_prepare_grouped_metric_data_filters_statuses_and_preserves_model_order(tmp_path: Path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "\n".join(
            [
                "run_id,function,method,n,p,q,rep,seed_data,seed_model,status,error,train_time_sec,pred_time_sec,nit,njev,nfev,rmse,coverage_95,interval_len_95,dss_diag,dss_full",
                "run_1,borehole,MOOGP,25,4,4,1,1,11,ok,,1.2,0.01,12,13,14,10.0,0.90,1.0,0.5,",
                "run_1,borehole,MOGP,25,4,4,1,1,12,ok,,0.8,0.01,9,10,11,11.0,0.94,1.0,0.5,",
                "run_1,borehole,MOOGP,50,4,4,1,1,13,opt_failed,msg,1.8,0.01,15,16,17,9.0,0.91,1.0,0.5,",
                "run_1,borehole,MOGP,50,4,4,1,1,14,error,msg,,,,,,,,",
            ]
        ),
        encoding="utf-8",
    )

    plot_data = prepare_grouped_metric_data(results_path, metric="coverage", models=("MOGP", "MOOGP"))

    assert plot_data.metric == "coverage_95"
    assert plot_data.models == ("MOGP", "MOOGP")
    assert plot_data.sample_sizes == (25, 50)
    assert plot_data.values["MOGP"][25] == [0.94]
    assert 50 not in plot_data.values["MOGP"]
    assert plot_data.values["MOOGP"][25] == [0.9]
    assert plot_data.values["MOOGP"][50] == [0.91]


def test_create_metric_figures_saves_expected_pngs(tmp_path: Path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "\n".join(
            [
                "run_id,function,method,n,p,q,rep,seed_data,seed_model,status,error,train_time_sec,pred_time_sec,nit,njev,nfev,rmse,coverage_95,interval_len_95,dss_diag,dss_full",
                "run_1,borehole,MOOGP,25,4,4,1,1,11,ok,,1.2,0.01,12,13,14,10.0,0.90,1.0,0.5,",
                "run_1,borehole,MOOGP,25,4,4,2,2,21,ok,,1.5,0.01,13,14,15,12.0,0.88,1.0,0.5,",
                "run_1,borehole,MOGP,25,4,4,1,1,12,ok,,0.8,0.01,9,10,11,11.0,0.94,1.0,0.5,",
                "run_1,borehole,MOOGP,50,4,4,1,1,13,ok,,2.1,0.01,15,16,17,9.0,0.91,1.0,0.5,",
                "run_1,borehole,MOGP,50,4,4,1,1,14,ok,,1.0,0.01,10,11,12,12.5,0.93,1.0,0.5,",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "plots"

    figures = create_metric_figures(
        results_csv=results_path,
        models=("MOOGP", "MOGP"),
        output_dir=output_dir,
        show=False,
    )

    assert set(figures) == {"train_time_sec", "rmse", "coverage_95"}
    assert (output_dir / "results_train_time_boxplot.png").exists()
    assert (output_dir / "results_rmse_boxplot.png").exists()
    assert (output_dir / "results_coverage_boxplot.png").exists()

    rmse_ax = figures["rmse"].axes[0]
    assert rmse_ax.get_xlabel() == "n"
    assert rmse_ax.get_ylabel() == "RMSE"
    assert [tick.get_text() for tick in rmse_ax.get_xticklabels()] == ["25", "50"]
    assert rmse_ax.get_legend() is not None

    for figure in figures.values():
        plt.close(figure)
