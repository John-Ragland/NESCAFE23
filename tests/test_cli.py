"""Tests for the interactive helpers of the nescafe23 CLI."""

import os

import pytest
from dotenv import dotenv_values
from nescafe23 import cli


class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


@pytest.fixture
def prompts(monkeypatch):
    """Answer the confirm/path prompts of `_set_data_dir` with the values in ``answers``."""
    answers = {}
    monkeypatch.setattr(
        cli.questionary, "confirm", lambda *a, **k: _Answer(answers["confirm"])
    )
    monkeypatch.setattr(cli.questionary, "path", lambda *a, **k: _Answer(answers["path"]))
    monkeypatch.delenv("NESCAFE23_DATADIR", raising=False)
    return answers


def test_set_data_dir_keeps_other_keys(tmp_path, prompts):
    env = tmp_path / ".env"
    env.write_text('GLIDER_DATA="/glider"\nNESCAFE23_DATADIR="/old/"\nWOA2023_PATH="/woa"\n')
    data = tmp_path / "data"
    data.mkdir()
    prompts.update(confirm=True, path=str(data))

    assert cli._set_data_dir(env) == f"{data}/"
    values = dotenv_values(env)
    assert values == {
        "GLIDER_DATA": "/glider",
        "NESCAFE23_DATADIR": f"{data}/",
        "WOA2023_PATH": "/woa",
    }
    assert os.environ["NESCAFE23_DATADIR"] == f"{data}/"


def test_set_data_dir_creates_env(tmp_path, prompts):
    env = tmp_path / ".env"
    prompts.update(path=str(tmp_path))
    assert cli._set_data_dir(env) == f"{tmp_path}/"
    assert dotenv_values(env) == {"NESCAFE23_DATADIR": f"{tmp_path}/"}


def test_set_data_dir_cancel(tmp_path, prompts):
    env = tmp_path / ".env"
    prompts.update(path=None)
    assert cli._set_data_dir(env) is None
    assert not env.exists()
    assert "NESCAFE23_DATADIR" not in os.environ


def test_set_data_dir_keep_existing(tmp_path, prompts):
    env = tmp_path / ".env"
    env.write_text('NESCAFE23_DATADIR="/old/"\n')
    prompts.update(confirm=False)
    assert cli._set_data_dir(env) is None
    assert dotenv_values(env) == {"NESCAFE23_DATADIR": "/old/"}
