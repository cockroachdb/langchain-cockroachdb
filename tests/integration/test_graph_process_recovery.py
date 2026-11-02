"""Real LangGraph workflows survive application-process loss on a shared saver.

The database stays online. PCR and exactly-once external effects are outside
these tests' scope.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest

pytest.importorskip("langgraph.graph")

from langchain_cockroachdb import CockroachDBSaver  # noqa: E402
from tests.integration.graph_recovery_worker import config, run_worker  # noqa: E402


@contextmanager
def application(
    url: str, mode: str, scenario: str, phase: str, agents: list[str], audit_dir: Path
) -> Iterator[tuple[multiprocessing.Process, Connection]]:
    # spawn guarantees recovery cannot inherit an old pool, loop or graph.
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=run_worker,
        args=(sender, url, mode, scenario, phase, agents, audit_dir),
        name=f"graph-recovery-{mode}-{scenario}-{phase}",
    )
    process.start()
    sender.close()
    try:
        yield process, receiver
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=10)
        receiver.close()
        assert not process.is_alive(), "recovery worker did not stop"
        process.close()


def receive(receiver: Connection, process: multiprocessing.Process, deadline: float) -> tuple:
    diagnostic = f"{process.name} (pid={process.pid}, exitcode={process.exitcode})"
    assert receiver.poll(max(0, deadline - monotonic())), (
        f"{diagnostic} did not complete its checkpoint barriers within 45 seconds"
    )
    try:
        event, agent, state = receiver.recv()
    except EOFError:
        pytest.fail(f"{diagnostic} closed its pipe before reaching its checkpoint barrier")
    if event == "error":
        pytest.fail(f"{diagnostic} failed for thread {agent}: {state}")
    return event, agent, state


def assert_recoverable_state(
    inspector: CockroachDBSaver, agent: str, scenario: str, audit_dir: Path, paused: dict
) -> None:
    # A successful parallel node is still a pending write here, not a
    # completed super-step checkpoint.
    saved = inspector.get_tuple(config(agent))
    assert saved is not None
    assert saved.checkpoint["channel_values"]["agent"] == agent
    assert "finished" not in saved.checkpoint["channel_values"]
    audit = (audit_dir / f"{agent}.log").read_text().splitlines()
    if scenario == "pending":
        prepared = [value for _, channel, value in saved.pending_writes if channel == "prepared"]
        assert prepared == [f"prepared:{agent}"]
        assert "prepared" not in saved.checkpoint["channel_values"]
        assert "work" not in saved.checkpoint["channel_values"]
        assert sorted(audit) == ["prepare", "work"]
    else:
        assert saved.checkpoint["channel_values"]["prepared"] == f"prepared:{agent}"
        assert "approved" not in saved.checkpoint["channel_values"]
        interrupts = [
            value for _, channel, value in saved.pending_writes if channel == "__interrupt__"
        ]
        assert len(interrupts) == 1
        assert len(interrupts[0]) == 1
        assert interrupts[0][0].value == {"agent": agent}
        assert paused[agent] == {
            "agent": agent,
            "prepared": f"prepared:{agent}",
            "__interrupt__": interrupts[0],
        }
        assert audit == ["prepare", "approve"]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("scenario", ["approval", "pending"])
def test_concurrent_graphs_resume_in_a_fresh_process(
    libpq_url: str, tmp_path: Path, mode: str, scenario: str
) -> None:
    agents = [f"recovery_{uuid4().hex}_{index}" for index in range(3)]
    with CockroachDBSaver.from_conn_string(libpq_url) as inspector:
        inspector.setup()
        try:
            with application(libpq_url, mode, scenario, "crash", agents, tmp_path) as (
                process,
                pipe,
            ):
                expected = {
                    (event, agent)
                    for agent in agents
                    for event in (
                        ["paused"] if scenario == "approval" else ["persisted", "blocked"]
                    )
                }
                paused = {}
                deadline = monotonic() + 45
                while expected:
                    event, agent, state = receive(pipe, process, deadline)
                    assert (event, agent) in expected
                    expected.remove((event, agent))
                    if event == "paused":
                        paused[agent] = state
                process.kill()
                process.join(timeout=10)
                assert not process.is_alive()
                assert process.exitcode != 0

            # Inspect from outside the killed application before reconnecting.
            for agent in agents:
                assert_recoverable_state(inspector, agent, scenario, tmp_path, paused)

            with application(libpq_url, mode, scenario, "resume", agents, tmp_path) as (
                process,
                pipe,
            ):
                results = {}
                deadline = monotonic() + 45
                while len(results) < len(agents):
                    event, agent, state = receive(pipe, process, deadline)
                    assert event == "result"
                    assert agent in agents and agent not in results
                    results[agent] = state
                process.join(timeout=30)
                assert process.exitcode == 0

            for agent in agents:
                suffix = f"approved:{agent}" if scenario == "approval" else f"worked:{agent}"
                state = results[agent]
                assert state == {
                    "agent": agent,
                    "prepared": f"prepared:{agent}",
                    "finished": f"prepared:{agent}|{suffix}",
                    "approved" if scenario == "approval" else "work": suffix,
                }
                saved = inspector.get_tuple(config(agent))
                assert saved is not None
                for channel, value in state.items():
                    assert saved.checkpoint["channel_values"][channel] == value
                assert not saved.pending_writes

                audit = (tmp_path / f"{agent}.log").read_text().splitlines()
                assert audit.count("prepare") == 1
                assert audit.count("finish") == 1
                # LangGraph re-enters an interrupted or unfinished node. A
                # tool effect before that boundary would therefore repeat.
                assert audit.count("approve" if scenario == "approval" else "work") == 2
        finally:
            for agent in agents:
                inspector.delete_thread(agent)
