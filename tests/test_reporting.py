from pathlib import Path

from src.reporting import RobustnessReporter


def sample_results():

    return {
        "gaussian_noise": {
            1: {"eval_f1": 0.82},
            2: {"eval_f1": 0.75},
            3: {"eval_f1": 0.66},
        },
        "rotation": {
            1: {"eval_f1": 0.81},
            2: {"eval_f1": 0.73},
            3: {"eval_f1": 0.61},
        },
    }


def test_dataframe():

    reporter = RobustnessReporter(
        sample_results()
    )

    df = reporter.to_dataframe()

    assert len(df) == 6


def test_csv_export(tmp_path):

    reporter = RobustnessReporter(
        sample_results()
    )

    output = tmp_path / "results.csv"

    reporter.save_csv(output)

    assert output.exists()


def test_plot_export(tmp_path):

    reporter = RobustnessReporter(
        sample_results()
    )

    output = tmp_path / "plot.png"

    reporter.plot_f1(output)

    assert output.exists()