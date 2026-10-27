"""Unit tests for store configuration validation."""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from langchain_cockroachdb.indexes import DistanceStrategy
from langchain_cockroachdb.store.config import (
    CockroachDBIndexConfig,
    CockroachDBTTLConfig,
    resolve_index_config,
    resolve_ttl_config,
    ttl_job_cron_for,
)


def test_resolve_index_defaults():
    """Test index resolution with default values."""
    config: CockroachDBIndexConfig = {"dims": 3, "embed": FakeEmbeddings(size=3)}
    resolved = resolve_index_config(config)

    assert resolved is not None
    assert resolved.dims == 3
    assert isinstance(resolved.embeddings, FakeEmbeddings)
    assert resolved.fields == (("$", "$"),)
    assert resolved.estimated_vectors == 1
    assert resolved.distance == DistanceStrategy.COSINE
    assert resolved.cspann == {}
    assert resolved.flat is False
    assert resolved.beam_size is None


def test_resolve_index_rejects_bad_values():
    """Test that resolve_index_config rejects invalid values."""
    # Missing dims
    with pytest.raises(ValueError, match="dims"):
        resolve_index_config({"embed": FakeEmbeddings(size=3)})  # type: ignore

    # dims is 0
    with pytest.raises(ValueError, match="dims"):
        resolve_index_config({"dims": 0, "embed": FakeEmbeddings(size=3)})

    # dims is a string
    with pytest.raises(ValueError, match="dims"):
        resolve_index_config({"dims": "3", "embed": FakeEmbeddings(size=3)})  # type: ignore

    # Bad distance_type
    with pytest.raises(ValueError, match="distance_type"):
        resolve_index_config(
            {
                "dims": 3,
                "embed": FakeEmbeddings(size=3),
                "distance_type": "dot",  # type: ignore
            }
        )

    # cspann with only min
    with pytest.raises(ValueError, match="cspann"):
        resolve_index_config(
            {"dims": 3, "embed": FakeEmbeddings(size=3), "cspann": {"min_partition_size": 100}}
        )

    # cspann max < 4 * min
    with pytest.raises(ValueError, match="cspann"):
        resolve_index_config(
            {
                "dims": 3,
                "embed": FakeEmbeddings(size=3),
                "cspann": {"min_partition_size": 100, "max_partition_size": 200},
            }
        )

    # beam_size is 0
    with pytest.raises(ValueError, match="beam_size"):
        resolve_index_config({"dims": 3, "embed": FakeEmbeddings(size=3), "beam_size": 0})


def test_resolve_index_tokenizes_fields():
    """Test that fields are tokenized and estimated_vectors is computed."""
    config: CockroachDBIndexConfig = {
        "dims": 3,
        "embed": FakeEmbeddings(size=3),
        "fields": ["text", "messages[*].content"],
    }
    resolved = resolve_index_config(config)

    assert resolved is not None
    # "text" tokenizes to 1 token
    # "messages[*].content" tokenizes to 3 tokens (messages, [*], content)
    assert resolved.estimated_vectors == 4
    assert len(resolved.fields) == 2
    # Check that the fields are tuples of (path, tokens)
    assert resolved.fields[0][0] == "text"
    assert resolved.fields[1][0] == "messages[*].content"

    # Test with indexed array element
    config2: CockroachDBIndexConfig = {
        "dims": 3,
        "embed": FakeEmbeddings(size=3),
        "fields": ["a.b[0].c"],
    }
    resolved2 = resolve_index_config(config2)
    assert resolved2 is not None
    # "a.b[0].c" tokenizes to 4 tokens (a, b, [0], c)
    assert resolved2.estimated_vectors == 4

    # Test with root path plus another field
    config3: CockroachDBIndexConfig = {
        "dims": 3,
        "embed": FakeEmbeddings(size=3),
        "fields": ["$", "text"],
    }
    resolved3 = resolve_index_config(config3)
    assert resolved3 is not None
    # "$" counts as 1, "text" tokenizes to 1
    assert resolved3.estimated_vectors == 2


def test_resolve_ttl_none_filters_but_writes_nothing():
    """Test that ttl=None returns default ResolvedTTL with filtering enabled."""
    resolved = resolve_ttl_config(None)

    assert resolved.default_ttl is None
    assert resolved.refresh_on_read is True
    assert resolved.omit_expired is True
    assert resolved.sweep_interval_minutes is None
    assert resolved.cron == "@hourly"


def test_resolve_ttl_omit_expired_opt_out():
    """Test that omit_expired can be set to False."""
    config: CockroachDBTTLConfig = {"omit_expired": False}
    resolved = resolve_ttl_config(config)

    assert resolved.omit_expired is False


def test_ttl_job_cron_for():
    """Test cron expression generation."""
    # None -> @hourly
    assert ttl_job_cron_for(None) == "@hourly"

    # >= 60 -> @hourly
    assert ttl_job_cron_for(60) == "@hourly"
    assert ttl_job_cron_for(90) == "@hourly"

    # 1 <= n < 60 -> */n * * * *
    assert ttl_job_cron_for(5) == "*/5 * * * *"
    assert ttl_job_cron_for(1) == "*/1 * * * *"

    # Non-integer below 60 -> ValueError
    with pytest.raises(ValueError):
        ttl_job_cron_for(0.5)

    with pytest.raises(ValueError):
        ttl_job_cron_for(2.5)


def test_ttl_job_cron_override_validated():
    """Test that ttl_job_cron is validated."""
    # Valid cron
    config: CockroachDBTTLConfig = {"ttl_job_cron": "@daily"}
    resolved = resolve_ttl_config(config)
    assert resolved.cron == "@daily"

    # Invalid cron (SQL injection attempt)
    with pytest.raises(ValueError, match="cron"):
        resolve_ttl_config({"ttl_job_cron": "x; DROP"})


def test_resolve_index_rejects_a_flat_that_is_not_a_bool():
    """flat must be a bool: the string "false" would be truthy and skip the index."""
    for bad in ("false", "true", 0, 1, None):
        with pytest.raises(ValueError, match="flat must be a bool"):
            resolve_index_config({"dims": 3, "embed": FakeEmbeddings(size=3), "flat": bad})  # type: ignore[typeddict-item]
    for flat in (True, False):
        resolved = resolve_index_config({"dims": 3, "embed": FakeEmbeddings(size=3), "flat": flat})
        assert resolved is not None and resolved.flat is flat
