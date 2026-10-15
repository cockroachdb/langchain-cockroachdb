"""Unit tests for the socket hang guard."""

import asyncio
import logging
import socket
import time

import pytest

import langchain_cockroachdb._guard as guard_module
from langchain_cockroachdb._guard import aabandon_after, abandon_after, shutdown_socket


@pytest.fixture
def socket_pair():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


def test_shutdown_socket_unblocks_a_reader(socket_pair) -> None:
    a, _ = socket_pair
    shutdown_socket(a.fileno())
    assert a.recv(1) == b""


def test_shutdown_socket_ignores_bad_descriptors() -> None:
    shutdown_socket(999_999)


def test_abandon_after_fires_when_the_block_overruns(socket_pair, caplog) -> None:
    a, _ = socket_pair
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        abandon_after(a.fileno(), 0.1, label="slow query"),
    ):
        assert a.recv(1) == b""
    assert "slow query" in caplog.text


def test_abandon_after_is_cancelled_when_the_block_finishes(socket_pair) -> None:
    a, b = socket_pair
    with abandon_after(a.fileno(), 0.2, label="fast query"):
        pass
    time.sleep(0.3)
    b.sendall(b"x")
    assert a.recv(1) == b"x"


def test_guards_are_disabled_by_none(socket_pair) -> None:
    a, _ = socket_pair
    with abandon_after(None, 0.01, label="x"):
        time.sleep(0.05)
    with abandon_after(a.fileno(), None, label="x"):
        time.sleep(0.05)


async def test_aabandon_after_fires(socket_pair, caplog) -> None:
    a, _ = socket_pair
    a.setblocking(False)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        async with aabandon_after(a.fileno(), 0.1, label="slow async query"):
            data = await asyncio.get_running_loop().sock_recv(a, 1)
    assert data == b""
    assert "slow async query" in caplog.text
    assert "0.1s attempt timeout" in caplog.text


async def test_aabandon_after_is_cancelled_on_exit(socket_pair) -> None:
    a, b = socket_pair
    async with aabandon_after(a.fileno(), 0.1, label="fast async query"):
        pass
    await asyncio.sleep(0.2)
    b.sendall(b"y")
    assert a.recv(1) == b"y"


def test_no_socket_logs_once(caplog, monkeypatch) -> None:
    monkeypatch.setattr(guard_module, "_no_socket_logged", False)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        with abandon_after(None, 1.0, label="test"):
            pass
        with abandon_after(None, 1.0, label="test"):
            pass
    assert caplog.text.count("driver does not expose a socket") == 1


async def test_async_no_socket_logs_once(caplog, monkeypatch) -> None:
    monkeypatch.setattr(guard_module, "_no_socket_logged", False)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        async with aabandon_after(None, 1.0, label="test"):
            pass
        async with aabandon_after(None, 1.0, label="test"):
            pass
    assert caplog.text.count("driver does not expose a socket") == 1


def test_no_socket_does_not_log_when_seconds_is_none(caplog, monkeypatch) -> None:
    monkeypatch.setattr(guard_module, "_no_socket_logged", False)
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        abandon_after(None, None, label="test"),
    ):
        pass
    assert "driver does not expose a socket" not in caplog.text


def test_disabled_guard_leaves_socket_working(socket_pair) -> None:
    """When the guard is disabled (fileno=None), the socket still works."""
    a, b = socket_pair
    with abandon_after(None, 0.01, label="disabled"):
        pass
    b.sendall(b"hello")
    assert a.recv(5) == b"hello"
