"""Setup shared by the integration tests."""

from __future__ import annotations

import pytest

from langchain_cockroachdb.checkpointer.base import BaseCockroachDBSaver


@pytest.fixture(autouse=True)
def _short_ttl_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the wait enable_ttl makes for the savers under a second.

    When the TTL setting changes, enable_ttl pauses TTL and waits
    TTL_STATE_CACHE_SECONDS, 60 by default, plus TTL_PAUSE_MARGIN_SECONDS,
    so that every saver reads the new setting. Lowering both for every saver
    keeps that wait, and the savers' cache, short in tests.
    """
    monkeypatch.setattr(BaseCockroachDBSaver, "TTL_STATE_CACHE_SECONDS", 0.5)
    monkeypatch.setattr(BaseCockroachDBSaver, "TTL_PAUSE_MARGIN_SECONDS", 0.1)
