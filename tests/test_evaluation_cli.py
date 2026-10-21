import sys

from evaluation import agent_evaluate
from knowledge import evaluate as retrieval_evaluate


def test_agent_evaluation_cli_exits_zero_when_expectations_are_met(capsys):
    assert agent_evaluate.main() == 0
    assert "expectation_failures: 0" in capsys.readouterr().out


def test_retrieval_evaluation_cli_meets_default_thresholds(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["knowledge.evaluate", "--top-k", "3", "--min-hit-rate", "1.0", "--min-recall", "1.0"],
    )

    assert retrieval_evaluate.main() == 0
    capsys.readouterr()


def test_retrieval_evaluation_cli_fails_when_threshold_is_not_met(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["knowledge.evaluate", "--top-k", "3", "--min-hit-rate", "1.01"]
    )

    assert retrieval_evaluate.main() == 1
    assert "threshold not met" in capsys.readouterr().err
