"""Unit tests for reconcile types, policies and decision validation."""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import Item, SearchItem

from langchain_cockroachdb.store.reconcile import (
    Action,
    Candidate,
    Conflicts,
    Decision,
    MemoryRecord,
    MemoryVersion,
    Proposal,
    ReconcileResult,
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


def test_validate_decision_rejects_superseding_the_new_key() -> None:
    # A late fact goes to history under its own key. The live memory under
    # that key cannot be superseded by it
    proposal = make_proposal("k", valid_at=T1)
    found = [candidate("k", reason="key", valid_at=T0), candidate("b", reason="slot", valid_at=T2)]
    decision = Decision(new="superseded", superseded_by="b", actions={"k": Action.SUPERSEDE})
    with pytest.raises(ValueError, match="supersede the proposal's own key: k"):
        validate_decision(proposal, found, decision)
    # Deleting or confirming it, or superseding another candidate, stays valid
    for actions in ({"k": Action.DELETE}, {"k": Action.CONFIRM}):
        validate_decision(proposal, found, Decision("superseded", "b", actions))
    older = candidate("a", reason="slot", valid_at=T0)
    validate_decision(
        proposal, [*found, older], Decision("superseded", "b", {"a": Action.SUPERSEDE})
    )


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


# A LangGraph checkpoint keeps a Conflicts and a Decision while a person
# decides, and the results and records an agent keeps in its state

# Aware, with an offset other than UTC's and microseconds
AWARE = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone(timedelta(hours=2)))


def checkpoint_round_trip(value: object) -> object:
    """``value`` as a LangGraph checkpoint gives it back."""
    serde = JsonPlusSerializer()
    return serde.loads_typed(serde.dumps_typed(value))


def assert_same_types(before: object, after: object) -> None:
    """Assert that ``after`` has the types of ``before`` all the way down.

    Tuples stay tuples, enums stay enums, and aware datetimes keep their offset.
    """
    assert type(after) is type(before), (before, after)
    if dataclasses.is_dataclass(before):
        for f in dataclasses.fields(before):
            assert_same_types(getattr(before, f.name), getattr(after, f.name))
    elif isinstance(before, Item):
        names = ["namespace", "key", "value", "created_at", "updated_at"]
        for name in [*names, "score"] if isinstance(before, SearchItem) else names:
            assert_same_types(getattr(before, name), getattr(after, name))
    elif isinstance(before, (tuple, list)):
        assert isinstance(after, (tuple, list))
        for one, other in zip(before, after, strict=True):
            assert_same_types(one, other)
    elif isinstance(before, dict):
        assert isinstance(after, dict) and list(after) == list(before)
        for key, one in before.items():
            assert_same_types(one, after[key])
    elif isinstance(before, datetime):
        assert isinstance(after, datetime)
        assert after.utcoffset() == before.utcoffset() is not None


ROUND_TRIPS = {
    "proposal": Proposal(
        namespace=("user", "123"),
        key="k",
        value={"text": "aisle", "tags": ["a", "b"]},
        slot="seat",
        slot_value="x",
        provenance={"source": "chat", "confidence": 0.9},
        valid_at=AWARE,
    ),
    "candidate": Candidate(
        item=SearchItem(
            namespace=("user", "123"),
            key="old",
            value={"text": "window"},
            created_at=AWARE,
            updated_at=AWARE,
            score=0.85,
        ),
        reason="similar",
        version=3,
        valid_at=AWARE,
        provenance={"source": "import"},
        confirmed_at=AWARE,
        confirmations=2,
    ),
    "candidate without a score": candidate("s", reason="slot", valid_at=AWARE),
    "decision": Decision(
        new="superseded",
        superseded_by="newer",
        actions={
            "newer": Action.CONFIRM,
            "old": Action.SUPERSEDE,
            "gone": Action.DELETE,
            "kept": Action.KEEP,
        },
    ),
    "default decision": Decision(),
    "result": ReconcileResult(
        key="k",
        new="written",
        version=4,
        superseded=("old",),
        deleted=("gone", "lost"),
        confirmed=(),
        kept=("kept",),
        already_applied=True,
    ),
    "version": MemoryVersion(
        namespace=("user", "123"),
        key="k",
        version=2,
        value={"text": "window"},
        provenance={"source": "chat"},
        valid_at=AWARE,
        invalid_at=AWARE + timedelta(days=1),
        written_at=AWARE,
        ended_at=AWARE + timedelta(days=2),
        reason="superseded",
        superseded_by="new",
    ),
    "open version": MemoryVersion(
        namespace=("user", "123"),
        key="k",
        version=1,
        value={"text": "window"},
        provenance=None,
        valid_at=AWARE,
        invalid_at=None,
        written_at=AWARE,
        ended_at=AWARE,
        reason="deleted",
        superseded_by=None,
    ),
    "record": MemoryRecord(
        item=Item(
            namespace=("user", "123"),
            key="k",
            value={"text": "window"},
            created_at=AWARE,
            updated_at=AWARE,
        ),
        version=2,
        provenance={"source": "chat"},
        valid_at=AWARE,
        confirmed_at=None,
        confirmations=0,
    ),
}


@pytest.mark.parametrize("value", list(ROUND_TRIPS.values()), ids=list(ROUND_TRIPS))
def test_every_reconcile_type_survives_a_checkpoint_round_trip(value: object) -> None:
    back = checkpoint_round_trip(value)

    assert back == value
    assert_same_types(value, back)


def test_a_decision_from_a_checkpoint_is_still_valid() -> None:
    proposal = make_proposal("k", valid_at=T1)
    found = [candidate("k", reason="key"), candidate("old", reason="slot"), candidate("dup")]
    decision = Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})

    back = checkpoint_round_trip(decision)

    assert isinstance(back, Decision)
    validate_decision(proposal, found, back)
    assert back.action("old") is Action.SUPERSEDE and back.action("k") is Action.KEEP


def test_a_decision_keeps_its_actions_as_actions() -> None:
    # An LLM judge's JSON, or a checkpoint whose serializer does not know
    # Action, gives the actions as their values
    given = {"old": "supersede", "dup": Action.CONFIRM}
    decision = Decision(actions=given)

    assert decision.actions == {"old": Action.SUPERSEDE, "dup": Action.CONFIRM}
    assert all(type(action) is Action for action in decision.actions.values())
    # The decision keeps its own copy
    given["old"] = "delete"
    assert decision.action("old") is Action.SUPERSEDE
    # An action that is not one of Action's fails at once, and so does a new
    # memory's fate that is not one of the three
    with pytest.raises(ValueError, match="remove"):
        Decision(actions={"old": "remove"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="'write', 'skip' or 'superseded', got 'writ'"):
        Decision(new="writ")  # type: ignore[arg-type]
