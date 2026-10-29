"""Unit tests for reconcile types, policies and decision validation."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from langgraph.store.base import SearchItem

from langchain_cockroachdb.store.reconcile import (
    Action,
    Candidate,
    Conflicts,
    Decision,
    Proposal,
    first_of,
    keep_all,
    skip_duplicates,
    supersede_same_slot,
    supersede_similar,
    validate_decision,
)

# Test time constants
T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 3, 0, 0, 0, tzinfo=timezone.utc)


def make_proposal(
    key: str,
    *,
    value: dict[str, object] | None = None,
    slot: str | None = "slot",
    slot_value: object = "seat_preference",
    valid_at: datetime = T1,
) -> Proposal:
    """Build a proposal for testing."""
    if value is None:
        value = {"text": "test"}
    return Proposal(
        namespace=("user", "123"),
        key=key,
        value=value,
        slot=slot,
        slot_value=slot_value,
        provenance=None,
        valid_at=valid_at,
    )


def candidate(
    key: str,
    *,
    reason: str = "similar",
    score: float | None = None,
    valid_at: datetime = T0,
    value: dict[str, object] | None = None,
    version: int = 1,
    provenance: dict[str, object] | None = None,
    confirmed_at: datetime | None = None,
    confirmations: int = 0,
) -> Candidate:
    """Build a candidate for testing."""
    if value is None:
        value = {"text": "old"}

    search_item = SearchItem(
        namespace=("user", "123"),
        key=key,
        value=value,
        created_at=valid_at,
        updated_at=valid_at,
        score=score,
    )

    return Candidate(
        item=search_item,
        reason=reason,  # type: ignore
        version=version,
        valid_at=valid_at,
        provenance=provenance,
        confirmed_at=confirmed_at,
        confirmations=confirmations,
    )


def test_supersede_same_slot_supersedes_older_facts_in_the_slot() -> None:
    proposal = make_proposal("new", valid_at=T1)
    decision = supersede_same_slot()(proposal, [candidate("old", reason="slot", valid_at=T0)])
    assert decision == Decision(actions={"old": Action.SUPERSEDE})


def test_supersede_same_slot_records_a_late_fact_as_superseded() -> None:
    proposal = make_proposal("new", valid_at=T0)
    found = [candidate("a", reason="slot", valid_at=T1), candidate("b", reason="slot", valid_at=T2)]
    assert supersede_same_slot()(proposal, found) == Decision(new="superseded", superseded_by="b")


def test_supersede_same_slot_skips_an_older_value_for_the_same_key() -> None:
    proposal = make_proposal("k", valid_at=T0)
    assert supersede_same_slot()(proposal, [candidate("k", reason="key", valid_at=T1)]) == Decision(
        new="skip"
    )


def test_supersede_same_slot_writes_over_an_older_target_in_the_slot() -> None:
    proposal = make_proposal("k", valid_at=T1)
    candidates = [candidate("k", reason="slot", valid_at=T0)]
    decision = supersede_same_slot()(proposal, candidates)
    assert decision == Decision()
    validate_decision(proposal, candidates, decision)


def test_skip_duplicates_confirms_the_closest_near_duplicate() -> None:
    found = [candidate("a", score=0.98), candidate("b", score=0.99), candidate("c", score=0.5)]
    assert skip_duplicates()(make_proposal("new"), found) == Decision(
        new="skip", actions={"b": Action.CONFIRM}
    )


def test_skip_duplicates_treats_an_equal_value_as_a_duplicate() -> None:
    proposal = make_proposal("new", value={"text": "same"})
    found = [candidate("k", reason="slot", value={"text": "same"})]
    assert skip_duplicates()(proposal, found).new == "skip"


def test_skip_duplicates_writes_when_nothing_is_close() -> None:
    found = [candidate("a", score=0.5), candidate("b", score=0.6)]
    assert skip_duplicates()(make_proposal("new"), found) == Decision()


def test_supersede_similar_uses_its_threshold() -> None:
    proposal = make_proposal("new")
    # At threshold 0.8, score 0.85 should be superseded
    found = [candidate("a", score=0.85)]
    assert supersede_similar(threshold=0.8)(proposal, found) == Decision(
        actions={"a": Action.SUPERSEDE}
    )

    # At threshold 0.9, score 0.85 should be kept (writes normally)
    assert supersede_similar(threshold=0.9)(proposal, found) == Decision()


def test_supersede_similar_does_not_supersede_the_target() -> None:
    proposal = make_proposal("k", valid_at=T1)
    candidates = [
        candidate("k", reason="key", score=0.95, valid_at=T0),
        candidate("other", score=0.95, valid_at=T0),
    ]
    decision = supersede_similar(threshold=0.9)(proposal, candidates)
    assert decision == Decision(actions={"other": Action.SUPERSEDE})
    validate_decision(proposal, candidates, decision)


def test_keep_all_writes_and_keeps() -> None:
    found = [candidate("a"), candidate("b")]
    assert keep_all()(make_proposal("new"), found) == Decision()


def test_first_of_returns_the_first_decision_that_acts() -> None:
    proposal = make_proposal("new")
    # skip_duplicates doesn't act (no duplicates), supersede_same_slot does
    found = [candidate("a", reason="slot", score=0.5, valid_at=T0)]
    policy = first_of(skip_duplicates(), supersede_same_slot())
    assert policy(proposal, found) == Decision(actions={"a": Action.SUPERSEDE})


def test_first_of_writes_when_no_policy_acts() -> None:
    proposal = make_proposal("new")
    found = [candidate("a", score=0.5)]  # Not a duplicate, not in slot
    policy = first_of(skip_duplicates(), keep_all())
    assert policy(proposal, found) == Decision()


def test_validate_decision_rejects_unknown_candidate_key() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(actions={"unknown": Action.SUPERSEDE})
    with pytest.raises(ValueError, match="unknown"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_superseded_without_superseded_by() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(new="superseded")
    with pytest.raises(ValueError, match="superseded_by"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_superseded_by_not_in_candidates() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(new="superseded", superseded_by="unknown")
    with pytest.raises(ValueError, match="unknown"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_superseded_by_same_key() -> None:
    proposal = make_proposal("k")
    found = [candidate("k")]
    decision = Decision(new="superseded", superseded_by="k")
    with pytest.raises(ValueError, match="k"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_superseded_by_not_kept_or_confirmed() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(new="superseded", superseded_by="a", actions={"a": Action.SUPERSEDE})
    with pytest.raises(ValueError, match="a"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_superseded_by_with_other_new() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(new="write", superseded_by="a")
    with pytest.raises(ValueError, match="superseded_by"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_write_with_target_action() -> None:
    proposal = make_proposal("k")
    found = [candidate("k")]
    decision = Decision(new="write", actions={"k": Action.SUPERSEDE})
    with pytest.raises(ValueError, match="k"):
        validate_decision(proposal, found, decision)


def test_validate_decision_rejects_skip_with_supersede_action() -> None:
    proposal = make_proposal("new")
    found = [candidate("a")]
    decision = Decision(new="skip", actions={"a": Action.SUPERSEDE})
    with pytest.raises(ValueError, match="a"):
        validate_decision(proposal, found, decision)


def test_policies_reject_a_threshold_that_is_not_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        skip_duplicates(threshold=math.nan)

    with pytest.raises(ValueError, match="finite"):
        skip_duplicates(threshold=math.inf)

    with pytest.raises(ValueError, match="finite"):
        supersede_similar(threshold="0.9")  # type: ignore


def test_policies_do_not_mutate_candidates() -> None:
    proposal = make_proposal("new")
    found = [candidate("a", score=0.99), candidate("b", score=0.5)]
    original_found = list(found)

    # Run multiple policies
    skip_duplicates()(proposal, found)
    supersede_same_slot()(proposal, found)
    keep_all()(proposal, found)

    # Verify candidates weren't mutated
    assert found == original_found


def test_conflicts_keeps_its_state_out_of_repr_and_comparison() -> None:
    import langchain_cockroachdb
    from langchain_cockroachdb import store as store_package

    proposal = make_proposal("k")
    found = (candidate("a", score=0.9),)
    with_state = Conflicts(
        proposal=proposal,
        candidates=found,
        target_version=2,
        target_updated_at=T0,
        _state=object(),  # type: ignore[arg-type]
    )
    plain = Conflicts(proposal=proposal, candidates=found, target_version=2, target_updated_at=T0)
    assert with_state == plain
    assert "_state" not in repr(with_state)
    assert plain._state is None
    # Exported from the store package, not from the top-level package
    assert store_package.Conflicts is Conflicts
    assert "Conflicts" in store_package.__all__
    assert not hasattr(langchain_cockroachdb, "Conflicts")


def test_the_reconcile_types_resolve_their_type_hints() -> None:
    # Tools that read type hints at runtime, such as schema builders, must not
    # meet a name that only type checkers can see
    import typing

    from langchain_cockroachdb.store.reconcile import MemoryRecord, MemoryVersion, ReconcileResult

    for kind in (
        Conflicts,
        Proposal,
        Candidate,
        Decision,
        ReconcileResult,
        MemoryRecord,
        MemoryVersion,
    ):
        assert typing.get_type_hints(kind)
    assert "_state" in typing.get_type_hints(Conflicts)
