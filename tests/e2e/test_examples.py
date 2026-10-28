"""Run every example against the test database, the way a user would."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=[path.name for path in EXAMPLES])
def test_example_runs_cleanly(example: Path, connection_string: str) -> None:
    url = connection_string.replace("cockroachdb+psycopg://", "cockroachdb://", 1)
    env = dict(os.environ, COCKROACHDB_URL=url)
    result = subprocess.run(
        [sys.executable, str(example)], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"{example.name} exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
    )
    assert "Traceback" not in result.stderr, result.stderr[-4000:]
    # Python's warnings, as "file:line: Category: message"; the retry log is not one
    warnings = re.findall(r"^\S+:\d+: \w*Warning: .*$", result.stderr, re.MULTILINE)
    assert not warnings, "\n".join(warnings)
