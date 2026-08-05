import pytest

from src.benchmark import BenchmarkRunner


class DummyEvaluator:

    def evaluate(
        self,
        corruption_name,
        severities,
    ):

        return {
            severity: {
                "eval_f1": 0.80,
            }
            for severity in severities
        }


@pytest.fixture
def benchmark():

    return BenchmarkRunner(
        DummyEvaluator()
    )


def test_runner_creation(
    benchmark,
):

    assert benchmark is not None


def test_total_conditions(
    benchmark,
):

    assert benchmark.available_conditions() == 15


def test_run_returns_dictionary(
    benchmark,
):

    results = benchmark.run()

    assert isinstance(
        results,
        dict,
    )


def test_five_corruptions(
    benchmark,
):

    results = benchmark.run()

    assert len(results) == 5


def test_three_severities(
    benchmark,
):

    results = benchmark.run()

    for values in results.values():

        assert len(values) == 3


def test_benchmark_name(
    benchmark,
):

    assert benchmark.benchmark_name() == "FUNSD Robustness Benchmark"