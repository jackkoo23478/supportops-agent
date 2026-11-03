from unittest.mock import Mock

import pytest

from supportops import cli
from supportops.llm import LLMError


@pytest.fixture
def workflow(monkeypatch):
    """Replace external dependencies and the message workflow."""
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: object())
    monkeypatch.setattr(cli, "make_llm", lambda settings: object())

    handler = Mock(return_value="Order O1003 status: Shipped.")
    monkeypatch.setattr(cli, "run_message", handler)

    return handler


def set_inputs(monkeypatch, messages):
    """Provide a sequence of simulated terminal inputs."""
    inputs = iter(messages)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(inputs),
    )


def test_package_imports():
    import supportops

    assert supportops is not None


def test_single_message_prints_reply(workflow, capsys):
    exit_code = cli.main(["Where is my order O1003?"])

    assert exit_code == 0
    assert "Order O1003 status: Shipped." in capsys.readouterr().out
    workflow.assert_called_once()
    assert workflow.call_args.args[0] == "Where is my order O1003?"


def test_single_message_handles_model_failure(workflow, capsys):
    workflow.side_effect = LLMError("Ollama unavailable")

    exit_code = cli.main(["Where is my order O1003?"])

    assert exit_code == 1
    assert "Ollama unavailable" in capsys.readouterr().err


def test_cli_rejects_blank_message(capsys):
    exit_code = cli.main(["   "])

    assert exit_code == 1
    assert "Please enter a message." in capsys.readouterr().err


def test_interactive_handles_multiple_messages(
    monkeypatch,
    workflow,
    capsys,
):
    set_inputs(
        monkeypatch,
        [
            "Where is my order O1003?",
            "Where is my order O1001?",
            "exit",
        ],
    )
    workflow.side_effect = [
        "Order O1003 status: Shipped.",
        "Order O1001 status: Delivered.",
    ]

    exit_code = cli.main([])

    assert exit_code == 0
    assert workflow.call_count == 2
    assert [
        call.args[0] for call in workflow.call_args_list
    ] == [
        "Where is my order O1003?",
        "Where is my order O1001?",
    ]

    output = capsys.readouterr().out
    assert "Agent: Order O1003 status: Shipped." in output
    assert "Agent: Order O1001 status: Delivered." in output
    assert "Goodbye." in output


def test_interactive_skips_blank_input(monkeypatch, workflow):
    set_inputs(monkeypatch, ["   ", "exit"])

    assert cli.main([]) == 0
    workflow.assert_not_called()


@pytest.mark.parametrize("command", ["exit", "quit", "EXIT"])
def test_interactive_exit_commands(
    command,
    monkeypatch,
    workflow,
    capsys,
):
    set_inputs(monkeypatch, [command])

    assert cli.main([]) == 0
    workflow.assert_not_called()
    assert "Goodbye." in capsys.readouterr().out


def test_interactive_continues_after_model_failure(
    monkeypatch,
    workflow,
    capsys,
):
    set_inputs(
        monkeypatch,
        [
            "Where is my order O1003?",
            "Where is my order O1001?",
            "exit",
        ],
    )
    workflow.side_effect = [
        LLMError("Ollama unavailable"),
        "Order O1001 status: Delivered.",
    ]

    exit_code = cli.main([])

    assert exit_code == 0
    assert workflow.call_count == 2

    captured = capsys.readouterr()
    assert "Ollama unavailable" in captured.err
    assert "Agent: Order O1001 status: Delivered." in captured.out


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_interactive_handles_input_interruptions(
    error,
    monkeypatch,
    workflow,
    capsys,
):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=error),
    )

    assert cli.main([]) == 0
    workflow.assert_not_called()
    assert "Goodbye." in capsys.readouterr().out