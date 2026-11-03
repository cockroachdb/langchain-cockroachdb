"""Spawned applications used by the LangGraph process-recovery tests."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, TypedDict

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver

POOL_OPTIONS = {"autocommit": True, "row_factory": dict_row, "gssencmode": "disable"}


def make_redactor(url: str) -> Callable[[str], str]:
    """Return a function that scrubs the URL and its password from text.

    A connection error from libpq can echo a malformed URL, password included,
    so the worker redacts tracebacks before sending them to the parent.
    """
    secrets = {url}
    try:
        password = conninfo_to_dict(url).get("password")
    except psycopg.Error:
        password = None
    if password:
        secrets.add(password)
    userinfo = re.compile(r"([a-z][a-z0-9+.-]*://[^:/@\s]*):[^@\s]*@", re.I)

    def redact(text: str) -> str:
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(str(secret), "<redacted>")
        return userinfo.sub(r"\1:<redacted>@", text)

    return redact


class AgentState(TypedDict, total=False):
    agent: str
    prepared: str
    work: str
    approved: str
    finished: str


def config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def record(audit_dir: Path, agent: str, event: str) -> None:
    # This separate durable log stands in for an external tool's side effects.
    # Checkpoint recovery cannot roll it back.
    with (audit_dir / f"{agent}.log").open("a") as audit:
        audit.write(event + "\n")
        audit.flush()
        os.fsync(audit.fileno())


def build_graph(saver: Any, scenario: str, work: Callable, audit_dir: Path) -> Any:
    def prepare(state: AgentState) -> dict:
        record(audit_dir, state["agent"], "prepare")
        return {"prepared": f"prepared:{state['agent']}"}

    def approve(state: AgentState) -> dict:
        record(audit_dir, state["agent"], "approve")
        return {"approved": interrupt({"agent": state["agent"]})}

    def finish(state: AgentState) -> dict:
        record(audit_dir, state["agent"], "finish")
        return {"finished": f"{state['prepared']}|{state.get('approved', state.get('work'))}"}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    if scenario == "approval":
        graph.add_node("approve", approve)
        graph.add_edge("prepare", "approve")
        graph.add_edge("approve", "finish")
    else:
        graph.add_node("parallel_work", work)
        graph.add_edge(START, "parallel_work")
        graph.add_edge(["prepare", "parallel_work"], "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=saver)


class ObservedSyncSaver(CockroachDBSaver):
    """Notify the parent only after a successful node's pending writes commit."""

    def __init__(self, pool: ConnectionPool, notify: Callable) -> None:
        super().__init__(pool)
        self.notify = notify

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        if any(channel == "prepared" for channel, _ in writes):
            self.notify("persisted", config["configurable"]["thread_id"], None)


class ObservedAsyncSaver(AsyncCockroachDBSaver):
    """Async counterpart of ObservedSyncSaver; database operations are unchanged."""

    def __init__(self, pool: AsyncConnectionPool, notify: Callable) -> None:
        super().__init__(pool)
        self.notify = notify

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await super().aput_writes(config, writes, task_id, task_path)
        if any(channel == "prepared" for channel, _ in writes):
            self.notify("persisted", config["configurable"]["thread_id"], None)


def graph_input(scenario: str, phase: str, agent: str) -> Any:
    if phase == "crash":
        return {"agent": agent}
    return Command(resume=f"approved:{agent}") if scenario == "approval" else None


def run_sync(
    url: str, scenario: str, phase: str, agents: list[str], audit_dir: Path, notify: Callable
) -> None:
    redact = make_redactor(url)

    def work(state: AgentState) -> dict:
        agent = state["agent"]
        record(audit_dir, agent, "work")
        if phase == "crash":
            notify("blocked", agent, None)
            threading.Event().wait()
        return {"work": f"worked:{agent}"}

    with ConnectionPool(url, min_size=1, max_size=2, kwargs=POOL_OPTIONS) as pool:
        saver = (
            ObservedSyncSaver(pool, notify)
            if scenario == "pending" and phase == "crash"
            else CockroachDBSaver(pool)
        )
        saver.setup()
        graph = build_graph(saver, scenario, work, audit_dir)

        def invoke(agent: str) -> None:
            # The next node must not start before its parent checkpoint commits.
            # Async durability permits that checkpoint to be lost on a crash.
            try:
                state = graph.invoke(
                    graph_input(scenario, phase, agent), config(agent), durability="sync"
                )
            except Exception as error:
                # Report failures even if another graph is intentionally blocked.
                # A redacted traceback keeps credentials out of libpq's messages.
                notify(
                    "error",
                    agent,
                    redact(
                        "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    ),
                )
                raise
            notify("paused" if phase == "crash" else "result", agent, state)

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            list(executor.map(invoke, agents))
        if phase == "crash":
            threading.Event().wait()


async def run_async(
    url: str, scenario: str, phase: str, agents: list[str], audit_dir: Path, notify: Callable
) -> None:
    redact = make_redactor(url)

    async def work(state: AgentState) -> dict:
        agent = state["agent"]
        record(audit_dir, agent, "work")
        if phase == "crash":
            notify("blocked", agent, None)
            await asyncio.Event().wait()
        return {"work": f"worked:{agent}"}

    async with AsyncConnectionPool(
        url, min_size=1, max_size=2, kwargs=POOL_OPTIONS, open=False
    ) as pool:
        saver = (
            ObservedAsyncSaver(pool, notify)
            if scenario == "pending" and phase == "crash"
            else AsyncCockroachDBSaver(pool)
        )
        await saver.setup()
        graph = build_graph(saver, scenario, work, audit_dir)

        async def invoke(agent: str) -> None:
            # Match the sync worker's durability contract.
            try:
                state = await graph.ainvoke(
                    graph_input(scenario, phase, agent), config(agent), durability="sync"
                )
            except Exception as error:
                notify(
                    "error",
                    agent,
                    redact(
                        "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    ),
                )
                raise
            notify("paused" if phase == "crash" else "result", agent, state)

        await asyncio.gather(*(invoke(agent) for agent in agents))
        if phase == "crash":
            await asyncio.Event().wait()


def run_worker(
    sender: Connection,
    url: str,
    mode: str,
    scenario: str,
    phase: str,
    agents: list[str],
    audit_dir: Path,
) -> None:
    # A sync graph writes checkpoints on worker threads. Serialize pipe sends
    # so concurrent notifications cannot corrupt a message.
    lock = threading.Lock()

    def notify(event: str, agent: str, state: Any) -> None:
        with lock:
            sender.send((event, agent, state))

    try:
        if mode == "sync":
            run_sync(url, scenario, phase, agents, audit_dir, notify)
        else:
            asyncio.run(run_async(url, scenario, phase, agents, audit_dir, notify))
    finally:
        sender.close()
