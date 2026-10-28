"""Reconcile types, policies and decision validation for the CockroachDB store.

Provides types and built-in policies for memory reconciliation: finding
candidates a new memory may conflict with and deciding which to keep, supersede,
delete or confirm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from langgraph.store.base import Item, SearchItem

if TYPE_CHECKING:
    # base.py imports this module, so the store's private state is named for
    # type checkers only
    from langchain_cockroachdb.store.base import ConflictsState


class Action(str, Enum):
    """Action to take on a candidate memory."""

    KEEP = "keep"
    SUPERSEDE = "supersede"
    DELETE = "delete"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Proposal:
    """A new memory being reconciled against existing candidates.

    Attributes:
        namespace: Namespace tuple.
        key: Memory key.
        value: Memory value as JSON-serializable dict.
        slot: Slot field name, or None.
        slot_value: Value of the slot field.
        provenance: Provenance metadata, or None.
        valid_at: When the fact became true (timezone aware).
    """

    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    slot: str | None
    slot_value: Any
    provenance: dict[str, Any] | None
    valid_at: datetime


@dataclass(frozen=True)
class Candidate:
    """A live memory that may conflict with a proposal.

    Attributes:
        item: SearchItem with the candidate's data and optional score.
        reason: Why this is a candidate (key, slot, or similar).
        version: Version number of this memory.
        valid_at: When the fact became true.
        provenance: Provenance metadata, or None.
        confirmed_at: When last confirmed, or None.
        confirmations: Number of confirmations.
    """

    item: SearchItem
    reason: Literal["key", "slot", "similar"]
    version: int
    valid_at: datetime
    provenance: dict[str, Any] | None
    confirmed_at: datetime | None
    confirmations: int

    @property
    def key(self) -> str:
        """The candidate's key."""
        return self.item.key

    @property
    def score(self) -> float | None:
        """The candidate's similarity score, or None for key and slot matches."""
        return self.item.score


@dataclass(frozen=True)
class Decision:
    """Policy decision on what to do with the proposal and candidates.

    Attributes:
        new: What to do with the new memory (write, skip, or superseded).
        superseded_by: Key that superseded the proposal, or None.
        actions: Mapping from candidate key to action (missing means KEEP).
    """

    new: Literal["write", "skip", "superseded"] = "write"
    superseded_by: str | None = None
    actions: Mapping[str, Action] = field(default_factory=dict)

    def action(self, key: str) -> Action:
        """Return the action for a candidate key (KEEP if not specified)."""
        return self.actions.get(key, Action.KEEP)

    def acts(self) -> bool:
        """Return True if this decision does more than just write and keep all."""
        return self.new != "write" or len(self.actions) > 0


@dataclass(frozen=True)
class ReconcileResult:
    """Result of a reconcile or resolve operation.

    Attributes:
        key: Key of the memory (written or not).
        new: What happened to the new memory (written, skipped, or superseded).
        version: New version number, or None if not written.
        superseded: Keys of candidates that were superseded.
        deleted: Keys of candidates that were deleted.
        confirmed: Keys of candidates that were confirmed.
        kept: Keys of candidates that were kept.
        already_applied: Whether an earlier attempt had already committed.
    """

    key: str
    new: Literal["written", "skipped", "superseded"]
    version: int | None
    superseded: tuple[str, ...]
    deleted: tuple[str, ...]
    confirmed: tuple[str, ...]
    kept: tuple[str, ...]
    already_applied: bool = False


@dataclass(frozen=True)
class MemoryVersion:
    """A historical version of a memory.

    Attributes:
        namespace: Namespace tuple.
        key: Memory key.
        version: Version number.
        value: Memory value.
        provenance: Provenance metadata, or None.
        valid_at: When the fact became true.
        invalid_at: When the fact became invalid, or None.
        written_at: When the store wrote this version.
        ended_at: When the store replaced this version.
        reason: Why this version ended (overwritten, superseded, deleted).
        superseded_by: Key that superseded this version, or None.
    """

    namespace: tuple[str, ...]
    key: str
    version: int
    value: dict[str, Any]
    provenance: dict[str, Any] | None
    valid_at: datetime
    invalid_at: datetime | None
    written_at: datetime
    ended_at: datetime
    reason: Literal["overwritten", "superseded", "deleted"]
    superseded_by: str | None


@dataclass(frozen=True)
class MemoryRecord:
    """A live memory with its reconciliation metadata.

    Attributes:
        item: The memory's Item.
        version: Version number.
        provenance: Provenance metadata, or None.
        valid_at: When the fact became true.
        confirmed_at: When last confirmed, or None.
        confirmations: Number of confirmations.
    """

    item: Item
    version: int
    provenance: dict[str, Any] | None
    valid_at: datetime
    confirmed_at: datetime | None
    confirmations: int


@dataclass(frozen=True)
class Conflicts:
    """What find_conflicts found: a new memory and the memories it may conflict with.

    Nothing is written yet. resolve() applies a decision about them in one
    transaction, once it has checked that the target and every candidate
    the decision names still have the version and updated_at they had here.

    Attributes:
        proposal: The new memory, with its key and valid time.
        candidates: The live memories it may conflict with, strongest
            reason first, then best score, then key.
        target_version: The version of the live memory under the new
            memory's key, or None if there is none.
        target_updated_at: When that memory was last written, or None.
    """

    proposal: Proposal
    candidates: tuple[Candidate, ...]
    target_version: int | None
    target_updated_at: datetime | None
    # The store's plan for the write: the item, its vectors and the versions
    # resolve() checks. Left out of repr and comparison
    _state: ConflictsState | None = field(default=None, repr=False, compare=False)


class ConflictsChanged(Exception):
    """Raised when candidates changed between find_conflicts and resolve.

    Attributes:
        keys: Tuple of keys that changed.
    """

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        key_list = ", ".join(keys)
        super().__init__(f"Conflicts changed for keys: {key_list}")


# Policy type
Policy = Callable[[Proposal, Sequence[Candidate]], Decision]


def _validate_threshold(threshold: float, name: str) -> None:
    """Validate that threshold is a finite number.

    Args:
        threshold: Threshold value to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If threshold is not a finite number.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"{name} must be a finite number, got {type(threshold).__name__}")
    if not math.isfinite(threshold):
        raise ValueError(f"{name} must be a finite number, got {threshold}")


def supersede_same_slot() -> Policy:
    """Return a policy that supersedes older facts in the same slot.

    When slot candidates exist (reason "slot"):
    1. If the target exists and has a later valid_at, skip the proposal.
    2. If any slot candidate has a later valid_at, record the proposal as
       superseded by the latest such candidate.
    3. Otherwise, supersede all slot candidates.

    Returns:
        Policy that supersedes same-slot candidates.
    """

    def policy(proposal: Proposal, candidates: Sequence[Candidate]) -> Decision:
        slot_candidates = [c for c in candidates if c.reason == "slot"]

        # Check if target (same key) is newer (regardless of reason)
        target = next((c for c in candidates if c.key == proposal.key), None)
        if target and target.valid_at > proposal.valid_at:
            return Decision(new="skip")

        if not slot_candidates:
            return Decision()

        # Check if any slot candidate is newer (excluding target)
        newer = [
            c for c in slot_candidates if c.valid_at > proposal.valid_at and c.key != proposal.key
        ]
        if newer:
            # Find the latest one
            latest = max(newer, key=lambda c: c.valid_at)
            return Decision(new="superseded", superseded_by=latest.key)

        # Supersede all slot candidates (excluding target)
        return Decision(
            actions={c.key: Action.SUPERSEDE for c in slot_candidates if c.key != proposal.key}
        )

    return policy


def skip_duplicates(threshold: float = 0.97) -> Policy:
    """Return a policy that skips near-duplicate memories.

    If a candidate scores at or above threshold, or has an equal value,
    skip the proposal and confirm the highest-scoring duplicate (equal
    values count as score 1.0).

    Args:
        threshold: Similarity threshold (default 0.97).

    Returns:
        Policy that skips duplicates.

    Raises:
        ValueError: If threshold is not a finite number.
    """
    _validate_threshold(threshold, "threshold")

    def policy(proposal: Proposal, candidates: Sequence[Candidate]) -> Decision:
        # Find duplicates: scored >= threshold or equal value
        duplicates = []
        for c in candidates:
            if c.item.value == proposal.value:
                # Equal value counts as score 1.0
                duplicates.append((c, 1.0))
            elif c.score is not None and c.score >= threshold:
                duplicates.append((c, c.score))

        if not duplicates:
            return Decision()

        # Confirm the highest scoring duplicate
        best = max(duplicates, key=lambda pair: pair[1])
        return Decision(new="skip", actions={best[0].key: Action.CONFIRM})

    return policy


def supersede_similar(threshold: float = 0.9) -> Policy:
    """Return a policy that supersedes similar memories.

    Works like supersede_same_slot, but matches candidates scoring at or
    above threshold instead of slot candidates.

    Args:
        threshold: Similarity threshold (default 0.9).

    Returns:
        Policy that supersedes similar candidates.

    Raises:
        ValueError: If threshold is not a finite number.
    """
    _validate_threshold(threshold, "threshold")

    def policy(proposal: Proposal, candidates: Sequence[Candidate]) -> Decision:
        # Find matches: slot candidates or scored >= threshold
        matches = [
            c
            for c in candidates
            if c.reason == "slot" or (c.score is not None and c.score >= threshold)
        ]
        if not matches:
            return Decision()

        # Check if target (same key) is newer
        target = next((c for c in matches if c.key == proposal.key), None)
        if target and target.valid_at > proposal.valid_at:
            return Decision(new="skip")

        # Check if any match is newer (excluding target)
        newer = [c for c in matches if c.valid_at > proposal.valid_at and c.key != proposal.key]
        if newer:
            # Find the latest one
            latest = max(newer, key=lambda c: c.valid_at)
            return Decision(new="superseded", superseded_by=latest.key)

        # Supersede all matches (excluding target)
        return Decision(actions={c.key: Action.SUPERSEDE for c in matches if c.key != proposal.key})

    return policy


def keep_all() -> Policy:
    """Return a policy that writes the new memory and keeps all candidates.

    Returns:
        Policy that keeps all candidates.
    """

    def policy(proposal: Proposal, candidates: Sequence[Candidate]) -> Decision:
        return Decision()

    return policy


def first_of(*policies: Policy) -> Policy:
    """Return a policy that uses the first decision that acts.

    Tries each policy in order and returns the first decision that does
    more than just write and keep all candidates. If no policy acts,
    returns a plain write decision.

    Args:
        *policies: Policies to try in order.

    Returns:
        Combined policy.
    """

    def policy(proposal: Proposal, candidates: Sequence[Candidate]) -> Decision:
        for p in policies:
            decision = p(proposal, candidates)
            if decision.acts():
                return decision
        return Decision()

    return policy


def _validate_superseded_by(
    proposal: Proposal,
    candidate_keys: set[str],
    decision: Decision,
) -> None:
    """Validate superseded_by field in a decision."""
    if decision.superseded_by is None:
        raise ValueError("Decision with new='superseded' must have superseded_by set")

    if decision.superseded_by not in candidate_keys:
        raise ValueError(
            f"Decision superseded_by names unknown candidate key: {decision.superseded_by}"
        )

    if decision.superseded_by == proposal.key:
        raise ValueError(
            f"Decision superseded_by cannot be the proposal's own key: {decision.superseded_by}"
        )

    # superseded_by must be kept or confirmed
    action = decision.action(decision.superseded_by)
    if action not in (Action.KEEP, Action.CONFIRM):
        raise ValueError(
            f"Decision superseded_by key must be kept or confirmed, not {action.value}: {decision.superseded_by}"
        )


def validate_decision(
    proposal: Proposal,
    candidates: Sequence[Candidate],
    decision: Decision,
) -> None:
    """Validate a decision against the proposal and candidates.

    Args:
        proposal: The proposal being reconciled.
        candidates: Candidate memories.
        decision: The decision to validate.

    Raises:
        ValueError: If the decision is invalid (with a message naming the key).
    """
    candidate_keys = {c.key for c in candidates}

    # Check that all action keys are candidates
    for key in decision.actions:
        if key not in candidate_keys:
            raise ValueError(f"Action names unknown candidate key: {key}")

    # Check superseded_by rules
    if decision.new == "superseded":
        _validate_superseded_by(proposal, candidate_keys, decision)
    elif decision.superseded_by is not None:
        raise ValueError("Decision superseded_by can only be set when new='superseded'")

    # Check write constraints
    if decision.new == "write" and proposal.key in decision.actions:
        # Target cannot have any action but KEEP
        action = decision.actions[proposal.key]
        if action != Action.KEEP:
            raise ValueError(
                f"Decision with new='write' cannot give the target key action {action.value}: {proposal.key}"
            )

    # Check skip constraints
    if decision.new == "skip":
        # Cannot have SUPERSEDE actions (nothing replaces it)
        for key, action in decision.actions.items():
            if action == Action.SUPERSEDE:
                raise ValueError(
                    f"Decision with new='skip' cannot have SUPERSEDE action (use DELETE): {key}"
                )
