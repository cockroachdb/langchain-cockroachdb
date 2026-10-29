"""Code shared by the sync and async CockroachDB stores.

Everything here runs without a database: grouping and validating ops,
planning the rows a put batch writes, building search, candidate and
decision statements, checking a decision that resolve applies and what
changed since find_conflicts, and turning result rows into items,
candidates and reconcile results. The store classes run the statements.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.embed import get_text_at_path, tokenize_path

from langchain_cockroachdb import _psycopg_conn
from langchain_cockroachdb.indexes import DistanceStrategy, _find_vector_index
from langchain_cockroachdb.store.config import (
    DEFAULT_RECONCILE_SCAN_LIMIT,
    ResolvedHistory,
    ResolvedIndex,
    ResolvedTTL,
    resolve_ttl_config,
)
from langchain_cockroachdb.store.filters import translate_filter
from langchain_cockroachdb.store.namespaces import (
    from_text,
    literal_prefix,
    match_pattern,
    to_text,
    validate_labels,
)
from langchain_cockroachdb.store.reconcile import (
    Action,
    Candidate,
    Conflicts,
    ConflictsChanged,
    Decision,
    MemoryRecord,
    MemoryVersion,
    Policy,
    Proposal,
    ReconcileResult,
    validate_decision,
)
from langchain_cockroachdb.store.sql import (
    MAX_SEARCH_NAMESPACES,
    NAMESPACES,
    ItemRow,
    ItemRowV2,
    StoreSQL,
    VectorRow,
    bind,
)

logger = logging.getLogger(__name__)

# Rows in one multi-row item upsert. Each row takes at most 6 bound
# parameters, so a statement stays far below the limit of 65,535 parameters.
ITEMS_PER_STATEMENT = 500

# Rows in one multi-row vector upsert. VECTOR rows are large, so batches stay
# small, like the vector store's default batch_size.
VECTORS_PER_STATEMENT = 100

# The lowest score of a similar candidate on a cosine store when no threshold
# is given. A cosine score is 1 minus the distance. On l2 and inner_product
# stores scores are negated distances, and no default fits.
DEFAULT_COSINE_THRESHOLD = 0.8

# Why a memory is a candidate, strongest reason first
_REASON_RANK = {"key": 0, "slot": 1, "similar": 2}


def _check_paging(limit: int, offset: int) -> None:
    """Raise ValueError for negative paging, before any database work."""
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")


def _memory_prefix(namespace: Sequence[str], method: str) -> str:
    """The namespace text of one memory, for methods that take a namespace and a key.

    Raises:
        InvalidNamespaceError: If the namespace is empty or a label is invalid.
    """
    labels = validate_labels(namespace)
    if not labels:
        raise InvalidNamespaceError(f"{method} needs a non-empty namespace")
    return to_text(labels)


def _check_limit(limit: object) -> None:
    """Raise ValueError unless ``limit`` is a non-negative int."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"limit must be a non-negative integer, got {limit!r}")
    _check_paging(limit, 0)


def _check_filter(filter: object) -> None:
    """Raise ValueError unless ``filter`` is None or a filter dict that search accepts."""
    if filter is not None and not isinstance(filter, Mapping):
        raise ValueError(f"filter must be a dict or None, got {type(filter).__name__}")
    translate_filter(filter)


def _check_ttl(ttl: object) -> None:
    """Raise ValueError unless ``ttl`` is None or a positive, finite number of minutes."""
    if ttl is None:
        return
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or not math.isfinite(ttl)
        or ttl <= 0
    ):
        raise ValueError(f"ttl must be a positive number of minutes, or None, got {ttl!r}")


def _check_valid_at(valid_at: object) -> None:
    """Raise ValueError unless ``valid_at`` is None or a timezone-aware datetime."""
    if valid_at is not None and (
        not isinstance(valid_at, datetime) or valid_at.utcoffset() is None
    ):
        raise ValueError(f"valid_at must be a timezone-aware datetime, got {valid_at!r}")


def _check_policy(policy: object) -> None:
    """Raise ValueError unless ``policy`` can be called to decide."""
    if not callable(policy):
        raise ValueError(
            "policy must be a function of the proposal and its candidates that returns "
            f"a Decision, got {type(policy).__name__}"
        )


def _serialized_provenance(provenance: object) -> tuple[dict[str, Any] | None, str | None]:
    """The provenance as its JSON reads back, and serialized, or (None, None).

    Raises:
        ValueError: If it is not a dict that serializes as JSON.
    """
    if provenance is None:
        return None, None
    if not isinstance(provenance, dict):
        raise ValueError(
            f"provenance must be a dict (a JSON object) or None, got {type(provenance).__name__}"
        )
    try:
        text = json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"provenance is not JSON serializable: {error}") from error
    return json.loads(text), text


def _slot_of(value: Mapping[str, Any], slot: object) -> tuple[Any, str | None]:
    """The value of the slot field, and serialized, or (None, None) without a slot.

    ``value`` is the new memory's value as its JSON reads back, so it holds
    no NaN or infinite number.

    Raises:
        ValueError: If ``slot`` does not name a top-level field of the value
            that holds a string, number or boolean.
    """
    if slot is None:
        return None, None
    if not isinstance(slot, str):
        raise ValueError(f"slot must name a top-level field of the value, got {slot!r}")
    if slot not in value:
        raise ValueError(f"The slot field {slot!r} is not a top-level field of the value")
    found = value[slot]
    # bool is an int
    if not isinstance(found, (str, int, float)):
        raise ValueError(
            f"The slot field {slot!r} must hold a string, number or boolean, "
            f"got {type(found).__name__}"
        )
    return found, json.dumps(found)


def needs_exact_search(rows: Sequence[Mapping[str, Any]], *, limit: int, over_fetch: int) -> bool:
    """Whether an indexed semantic search must run again in its exact form.

    ``rows`` are what StoreSQL.semantic_search_in returned for one search,
    after dedup and paging. Each row carries ``window_size``, the number of
    vectors the index returned into the window, and ``namespace_vectors``,
    the number of vectors the namespaces hold, counted up to ``over_fetch``.

    A result of ``limit`` rows is complete. A shorter one is exact only when
    the index returned every vector the namespaces hold: the window is not
    full and its size equals the count. A full window may have left out
    vectors beyond it, and the count cannot show otherwise, since it stops
    at the window's size. A window smaller than the count was cut short by
    the index itself, which a small beam_size or small partitions can do.
    In both cases matches may be missing, because candidates that failed
    the filter or had expired took their places. With no rows at all the
    sizes are unknown. All of these run again.
    """
    if len(rows) >= limit:
        return False
    if not rows:
        return True
    window = int(rows[0]["window_size"])
    held = int(rows[0]["namespace_vectors"])
    return window >= over_fetch or window < held


def needs_exact_candidates(
    rows: Sequence[Mapping[str, Any]], *, threshold: float, limit: int, over_fetch: int
) -> bool:
    """Whether an index scan for similar candidates must run again in its exact form.

    ``rows`` are what the index form of StoreSQL.similar_candidates returned
    for one vector of a new memory, with ``limit`` its LIMIT and
    ``over_fetch`` its window. Each carries ``window_size``, the vectors the
    C-SPANN index returned into the window, ``lowest_score``, the lowest
    score among them, counted before the filter, the expiry check and the
    excluded keys removed any, and ``namespace_vectors``, the namespace's
    vectors counted up to ``over_fetch``.

    A scan that returned ``limit`` hits is complete: a vector beyond the
    window scores no higher than the window's lowest, so it cannot outrank
    any hit. Otherwise the window must be whole for its result to stand: it
    holds ``over_fetch`` vectors, or every vector of the namespace. A window
    with fewer while the namespace holds more was cut short by the index, as
    a small beam size or small partitions can, and a vector it never reached
    may score higher than its lowest. A whole window whose lowest score is
    below ``threshold`` held every vector at or above it, up to the index's
    approximation, so its result stands however few rows passed. When its
    lowest score is at or above the threshold, the rows the filter, the
    expiry check or the excluded keys removed may have taken every place in
    it, and matches may lie beyond it. An empty window tells nothing about
    the namespace. All of these run again. Unlike needs_exact_search, a
    short result alone never does: short results are the normal case here.
    """
    hits = sum(1 for row in rows if row["key"] is not None)
    if hits >= limit:
        return False
    if not rows:
        return True
    window = int(rows[0]["window_size"])
    lowest = rows[0]["lowest_score"]
    if window == 0 or lowest is None:
        return True
    if window < over_fetch and window < int(rows[0]["namespace_vectors"]):
        return True
    return float(lowest) >= threshold


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    """Group ops by type, keeping each op's position in the batch.

    Validates the namespace labels of every op, like PostgresStore, and the
    paging of searches and listings, so a bad op fails the batch before any
    statement runs.

    Returns:
        The ops of each type as (position, op) pairs, and the number of ops.

    Raises:
        InvalidNamespaceError: If a label is empty, not a string or contains a dot.
        ValueError: If an op has an unknown type or negative paging.
    """
    grouped: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    total = 0
    for position, op in enumerate(ops):
        if isinstance(op, (GetOp, PutOp)):
            validate_labels(op.namespace)
        elif isinstance(op, SearchOp):
            validate_labels(op.namespace_prefix)
            _check_paging(op.limit, op.offset)
        elif isinstance(op, ListNamespacesOp):
            for condition in op.match_conditions or ():
                validate_labels(condition.path)
            _check_paging(op.limit, op.offset)
        else:
            raise ValueError(f"Unknown operation type: {type(op)}")
        grouped[type(op)].append((position, op))
        total += 1
    return grouped, total


@dataclass
class PutPlan:
    """What a batch of puts writes, worked out before the first attempt.

    Attributes:
        deletes: Namespace text to the keys deleted there.
        items: Item rows to upsert, one per (namespace, key).
        embed_requests: (prefix, key, field_name, text) for every text to
            embed, one per (prefix, key, field_name).
        reindexed: Namespace text to the put keys whose vectors are deleted
            before new ones are written. Every put item when the store has an index.
        write_id: The batch's write id, chosen once before the first attempt
            when migration 2's columns exist. The item upserts store it, so
            a retry after a commit whose result was lost changes nothing.
            None writes with the 0.5 statements.
    """

    deletes: dict[str, list[str]] = field(default_factory=dict)
    items: list[ItemRow] = field(default_factory=list)
    embed_requests: list[tuple[str, str, str, str]] = field(default_factory=list)
    reindexed: dict[str, list[str]] = field(default_factory=dict)
    write_id: str | None = None

    def __post_init__(self) -> None:
        # A LangGraph checkpoint gives tuples back as lists, as when a
        # Conflicts waits in graph state for a person to decide
        self.items = [(prefix, key, value, ttl) for prefix, key, value, ttl in self.items]
        self.embed_requests = [
            (prefix, key, name, text) for prefix, key, name, text in self.embed_requests
        ]


@dataclass(frozen=True)
class SchemaCheck:
    """What the schema check found.

    Attributes:
        columns: Whether setup() has applied migration 2, which adds the
            columns that versions, reconciliation and history use.
        history: Whether the history table exists. Always False on a store
            without history, which never looks for it.
    """

    columns: bool
    history: bool


@dataclass
class NamespaceLookup:
    """The namespace lookup of an indexed semantic search, and its exact forms.

    Attributes:
        position: The position of the search statement in the batch's
            search statements.
        statement: The lookup, StoreSQL.namespace_lookup.
        fallback: StoreSQL.semantic_search over the namespace prefix, used
            instead of the indexed search when the lookup finds more than
            MAX_SEARCH_NAMESPACES namespaces.
        exact: StoreSQL.exact_semantic_search_in, run after the indexed
            search when needs_exact_search says so. None once nothing can
            call for it: the fallback ran, or the lookup found no namespace.
        limit: The search's limit.
        over_fetch: The window of the indexed search.
    """

    position: int
    statement: tuple[str, list]
    fallback: tuple[str, list]
    exact: tuple[str, list] | None
    limit: int
    over_fetch: int


@dataclass
class BatchPlan:
    """The statements of a batch, built before any of them runs.

    Attributes:
        results: One slot per op, in the order of the batch. Searches with
            ``limit == 0`` and the puts are already filled in.
        get_groups: The gets, grouped by (namespace, refresh choice), in
            the order of ``get_statements``.
        get_statements: One read statement per get group.
        searches: The searches that need a statement, as (position, op).
        search_statements: One statement per search, in the order of ``searches``.
        search_to_embed: (statement position, query text) for every search
            statement that needs the embedding of its query.
        search_lookups: The namespace lookup of every indexed semantic
            search. The lookup finds the namespaces bound in place of
            NAMESPACES.
        exact_searches: The lookups whose indexed search came back short
            and may have missed vectors (see needs_exact_search), set once
            the search group ran. Their exact form runs next, and its rows
            replace the search's result.
        search_session: Statements that run first in the search group, in
            its transaction: the beam size, when set and a query is present.
        listings: The list_namespaces ops, as (position, op).
        listing_statements: One statement per listing.
        puts: What the batch's puts and deletes write, or None without puts.
    """

    results: list[Result]
    get_groups: list[tuple[tuple[tuple[str, ...], bool], list[tuple[int, GetOp]]]] = field(
        default_factory=list
    )
    get_statements: list[tuple[str, list]] = field(default_factory=list)
    searches: list[tuple[int, SearchOp]] = field(default_factory=list)
    search_statements: list[tuple[str, list]] = field(default_factory=list)
    search_to_embed: list[tuple[int, str]] = field(default_factory=list)
    search_lookups: list[NamespaceLookup] = field(default_factory=list)
    exact_searches: list[NamespaceLookup] = field(default_factory=list)
    search_session: list[str] = field(default_factory=list)
    listings: list[tuple[int, ListNamespacesOp]] = field(default_factory=list)
    listing_statements: list[tuple[str, list]] = field(default_factory=list)
    puts: PutPlan | None = None

    def reads(
        self,
    ) -> list[tuple[str, list[tuple[str, list]], Callable[[list[Any]], None], list[str]]]:
        """The read groups to run, in order: gets, searches, then listings.

        Each group is (label, statements, fill, session). Run the session
        statements, then the statements, as one round trip, in one
        transaction when there are session statements, and pass ``fill``
        the rows of each statement, in order; it puts them into
        ``results``. Empty groups are left out. Bind the query vectors and
        namespaces first. Then run ``exact_statements()``, if any, as one
        more group and pass its rows to ``fill_exact``.
        """
        groups = [
            ("store get", self.get_statements, self._fill_gets, []),
            ("store search", self.search_statements, self._fill_searches, self.search_session),
            ("store list namespaces", self.listing_statements, self._fill_listings, []),
        ]
        return [group for group in groups if group[1]]

    def query_texts(self) -> list[str]:
        """The distinct query texts to embed, in order of first use."""
        return list(dict.fromkeys(text for _, text in self.search_to_embed))

    def bind_query_vectors(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Put the embedding of each query text into its search statements."""
        texts = dict(self.search_to_embed)
        for position, text in self.search_to_embed:
            self.search_statements[position] = StoreSQL.bind_query_vector(
                self.search_statements[position], vectors[text]
            )
        for lookup in self.search_lookups:
            vector = vectors[texts[lookup.position]]
            lookup.fallback = StoreSQL.bind_query_vector(lookup.fallback, vector)
            if lookup.exact is not None:
                lookup.exact = StoreSQL.bind_query_vector(lookup.exact, vector)

    def lookup_statements(self) -> list[tuple[str, list]]:
        """The namespace lookups, to run as one read group before the searches."""
        return [lookup.statement for lookup in self.search_lookups]

    def bind_namespaces(self, found: list[Any]) -> None:
        """Put the namespaces each lookup found into its search statements.

        A search whose lookup found more than MAX_SEARCH_NAMESPACES
        namespaces runs its exact one-statement form instead.
        """
        for lookup, rows in zip(self.search_lookups, found, strict=True):
            if len(rows) > MAX_SEARCH_NAMESPACES:
                self.search_statements[lookup.position] = lookup.fallback
                lookup.exact = None
                continue
            namespaces = [row["prefix"] for row in rows]
            self.search_statements[lookup.position] = bind(
                self.search_statements[lookup.position], NAMESPACES, namespaces
            )
            # With no namespace to search, the empty result is already exact
            if lookup.exact is not None and namespaces:
                lookup.exact = bind(lookup.exact, NAMESPACES, namespaces)
            else:
                lookup.exact = None

    def exact_statements(self) -> list[tuple[str, list]]:
        """The exact searches that indexed searches called for, once the search group ran."""
        return [lookup.exact for lookup in self.exact_searches if lookup.exact is not None]

    def fill_exact(self, found: list[Any]) -> None:
        """Replace the result of each search that ran again with the exact rows."""
        searches = [lookup for lookup in self.exact_searches if lookup.exact is not None]
        for lookup, rows in zip(searches, found, strict=True):
            i, _ = self.searches[lookup.position]
            self.results[i] = [_row_to_search_item(row) for row in rows]

    def _fill_gets(self, found: list[Any]) -> None:
        for ((namespace, _), items), rows in zip(self.get_groups, found, strict=True):
            by_key = {row["key"]: row for row in rows}
            for i, op in items:
                row = by_key.get(op.key)
                self.results[i] = None if row is None else _row_to_item(namespace, row)

    def _fill_searches(self, found: list[Any]) -> None:
        for (i, _), rows in zip(self.searches, found, strict=True):
            self.results[i] = [_row_to_search_item(row) for row in rows]
        self.exact_searches = [
            lookup
            for lookup in self.search_lookups
            if lookup.exact is not None
            and needs_exact_search(
                found[lookup.position], limit=lookup.limit, over_fetch=lookup.over_fetch
            )
        ]

    def _fill_listings(self, found: list[Any]) -> None:
        for (i, _), rows in zip(self.listings, found, strict=True):
            self.results[i] = [from_text(row["ns"]) for row in rows]


@dataclass(frozen=True)
class ProposalPlan:
    """A new memory for find_conflicts or reconcile, checked before any I/O.

    Attributes:
        namespace: The memory's namespace.
        prefix: The namespace as text.
        key: The memory's key, a uuid4 when none was given.
        value: The value, as its JSON reads back.
        value_json: The value serialized.
        slot: The top-level field that holds the slot, or None.
        slot_value: The slot's value, or None without a slot.
        slot_value_json: The slot's value serialized, or None without a slot.
        threshold: The lowest score of a similar candidate, or None when no
            similar candidates are searched: the store has no index, or the
            call passed index=False.
        limit: Most similar candidates, besides the key and slot matches.
        filter: Narrows the slot and similar candidates by value.
        index: Which fields to embed, as put takes it.
        ttl: Minutes the memory lives once written, or None.
        provenance: The provenance, as its JSON reads back, or None.
        provenance_json: The provenance serialized, or None.
        valid_at: The valid time the caller gave, or None. Without one the
            new memory is valid from the database's now() (default_valid_at),
            and its write stores no valid time, so the memory's valid time is
            its updated_at, as after a plain put.
        put: The write as a put plan of one item, with its write id and the
            texts to embed.
    """

    namespace: tuple[str, ...]
    prefix: str
    key: str
    value: dict[str, Any]
    value_json: str
    slot: str | None
    slot_value: Any
    slot_value_json: str | None
    threshold: float | None
    limit: int
    filter: Mapping[str, Any] | None
    index: Literal[False] | list[str] | None
    ttl: float | None
    provenance: dict[str, Any] | None
    provenance_json: str | None
    valid_at: datetime | None
    put: PutPlan

    def __post_init__(self) -> None:
        # Tuples come back from a LangGraph checkpoint as lists
        object.__setattr__(self, "namespace", tuple(self.namespace))
        if isinstance(self.index, tuple):
            object.__setattr__(self, "index", list(self.index))

    @property
    def embed_requests(self) -> list[tuple[str, str, str, str]]:
        """(prefix, key, field_name, text) for every vector the memory gets."""
        return self.put.embed_requests

    @property
    def write_id(self) -> str:
        """The write id, chosen with the plan, so every attempt reuses it.

        Raises:
            ValueError: If the plan has none, as only a plan built by hand can.
        """
        if self.put.write_id is None:
            raise ValueError("The plan of the new memory has no write id")
        return self.put.write_id


@dataclass(frozen=True)
class KeySlotRead:
    """What the first candidate read found.

    Attributes:
        now: The database's now(), the default valid time.
        rows: The live target, if any, and the slot matches, as
            StoreSQL.key_slot_candidates returns them.
        vectors: The namespace's vectors, counted up to one past
            reconcile_scan_limit, or None when the similar read has no scan
            to choose.
    """

    now: datetime
    rows: list[dict[str, Any]]
    vectors: int | None


@dataclass(frozen=True)
class SimilarRead:
    """The second candidate read, built from what the first found.

    Run the session statements, then ``statements()``, as one round trip.
    If ``exact_statements()`` of their rows returns any, run those as one
    more round trip and pass both to ``fill_exact``.

    Attributes:
        scores: One StoreSQL.candidate_scores statement per vector of the
            new memory, or none without key and slot candidates.
        similar: One StoreSQL.similar_candidates statement per vector of the
            new memory, or none when the limit is 0.
        session: Statements that run first, in the read's transaction: the
            beam size of an index scan, when set.
        exact: The exact form of each statement in ``similar`` when those are
            index scans, else none. One runs only when its index scan's
            window may have left matches out (needs_exact_candidates).
        threshold: The lowest score of a similar candidate, or None when
            ``similar`` is empty.
        limit: The LIMIT of each statement in ``similar``.
        over_fetch: The window of each index scan in ``similar``.
    """

    scores: list[tuple[str, list]]
    similar: list[tuple[str, list]]
    session: list[str]
    exact: list[tuple[str, list]] = field(default_factory=list)
    threshold: float | None = None
    limit: int = 0
    over_fetch: int = 0

    def statements(self) -> list[tuple[str, list]]:
        """The statements to send as one round trip: the scores, then the similar reads."""
        return [*self.scores, *self.similar]

    def exact_statements(
        self, found: Sequence[Sequence[Mapping[str, Any]]]
    ) -> list[tuple[str, list]]:
        """The exact form of each index scan whose window may have left matches out.

        ``found`` are the rows of ``statements()``, in order.
        """
        return [self.exact[i] for i in self._reruns(found)]

    def fill_exact(
        self,
        found: Sequence[Sequence[dict[str, Any]]],
        exact_found: Sequence[Sequence[dict[str, Any]]],
    ) -> list[Sequence[dict[str, Any]]]:
        """``found`` with the rows of each index scan that ran again replaced by its exact rows.

        ``exact_found`` are the rows of ``exact_statements(found)``, in order.
        """
        filled = list(found)
        for i, rows in zip(self._reruns(found), exact_found, strict=True):
            filled[len(self.scores) + i] = rows
        return filled

    def _reruns(self, found: Sequence[Sequence[Mapping[str, Any]]]) -> list[int]:
        """The positions in ``similar`` of the index scans that must run again."""
        if not self.exact or self.threshold is None:
            return []
        threshold = self.threshold
        return [
            i
            for i, rows in enumerate(found[len(self.scores) :])
            if needs_exact_candidates(
                rows, threshold=threshold, limit=self.limit, over_fetch=self.over_fetch
            )
        ]


@dataclass(frozen=True)
class StoreIdentity:
    """Which store made a Conflicts: the items table and the index it embeds for.

    resolve() writes what find_conflicts planned, so it must run on a store
    with the same table and index.

    Attributes:
        items: The items table, with its schema.
        dims: The index's dims, or None without an index.
        distance: The index's distance, as configured, or None.
        fields: The index's fields, or None.
    """

    items: str
    dims: int | None
    distance: str | None
    fields: tuple[str, ...] | None

    def __post_init__(self) -> None:
        # Tuples come back from a LangGraph checkpoint as lists
        if self.fields is not None:
            object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class ConflictsState:
    """What Conflicts keeps from find_conflicts, for resolve().

    Attributes:
        store: Which store made it.
        plan: The new memory's plan, with its write id.
        vector_rows: Its vectors, embedded once.
        target: The (version, updated_at) of the live memory under its key,
            or None if there was none or it had expired.
        tokens: The (version, updated_at) of each candidate, by key.
    """

    store: StoreIdentity
    plan: ProposalPlan
    vector_rows: tuple[VectorRow, ...]
    target: tuple[int, datetime] | None
    tokens: dict[str, tuple[int, datetime]]

    def __post_init__(self) -> None:
        # Tuples come back from a LangGraph checkpoint as lists. Each vector
        # row keeps its embedding a list, as VectorRow has it
        object.__setattr__(self, "vector_rows", tuple(tuple(row) for row in self.vector_rows))
        if self.target is not None:
            object.__setattr__(self, "target", tuple(self.target))
        object.__setattr__(
            self, "tokens", {key: tuple(token) for key, token in self.tokens.items()}
        )


@dataclass(frozen=True)
class ResolvePlan:
    """What resolve() sends to apply a decision about a Conflicts, built before any I/O.

    Each attempt sends ``reads`` as one round trip and passes their rows to
    BaseCockroachDBStore._check_resolve. Unless that returns the result of
    an earlier commit or raises ConflictsChanged, the attempt sends
    ``writes`` and commits, and passes the rows they return to
    _reconcile_result with ``plan``, ``candidates`` and ``decision``.

    Attributes:
        plan: The new memory's plan, as find_conflicts made it, with its write id.
        candidates: The candidates find_conflicts found.
        decision: The decision about them, validated.
        expected: The (version, updated_at) that find_conflicts read, by
            key: first the new memory's key, with None when no live memory
            had it, then each candidate the decision does more than keep or
            names as superseding the new memory, in candidate order.
        removed: The candidates the decision deletes or supersedes, other
            than the one under the new memory's key. On a retry within the
            same call, one of them that is gone counts as unchanged: the
            earlier attempt may have removed it, and removing it again
            changes nothing.
        reads: The read of the live (version, updated_at) of the keys in
            ``expected`` (StoreSQL.tokens), then the reads of the rows that
            carry the write id under the new key and under every
            candidate's key.
        writes: The statements that apply the decision, as reconcile's.
    """

    plan: ProposalPlan
    candidates: tuple[Candidate, ...]
    decision: Decision
    expected: dict[str, tuple[int, datetime] | None]
    removed: frozenset[str]
    reads: list[tuple[str, list]]
    writes: list[tuple[str, list]]


def _row_to_item(namespace: tuple[str, ...], row: dict[str, Any]) -> Item:
    """Build an Item from a get row."""
    return Item(
        value=row["value"],
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_search_item(row: dict[str, Any]) -> SearchItem:
    """Build a SearchItem from a search row. The namespace comes from its prefix."""
    score = row.get("score")
    return SearchItem(
        namespace=from_text(row["prefix"]),
        key=row["key"],
        value=row["value"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=None if score is None else float(score),
    )


def _row_to_version(namespace: tuple[str, ...], key: str, row: Mapping[str, Any]) -> MemoryVersion:
    """Build a MemoryVersion from a row of StoreSQL.history_versions."""
    return MemoryVersion(
        namespace=namespace,
        key=key,
        version=int(row["version"]),
        value=row["value"],
        provenance=row["provenance"],
        valid_at=row["valid_at"],
        invalid_at=row["invalid_at"],
        written_at=row["written_at"],
        ended_at=row["ended_at"],
        reason=row["reason"],
        superseded_by=row["superseded_by"],
    )


def _row_to_record(namespace: tuple[str, ...], row: dict[str, Any]) -> MemoryRecord:
    """Build a MemoryRecord from a row of StoreSQL.record."""
    return MemoryRecord(
        item=_row_to_item(namespace, row),
        version=int(row["version"]),
        provenance=row["provenance"],
        valid_at=row["valid_at"],
        confirmed_at=row["confirmed_at"],
        confirmations=int(row["confirmations"]),
    )


def _row_to_candidate(
    namespace: tuple[str, ...],
    row: Mapping[str, Any],
    reason: Literal["key", "slot", "similar"],
    score: float | None,
) -> Candidate:
    """Build a Candidate from a row of a candidate statement."""
    return Candidate(
        item=SearchItem(
            namespace=namespace,
            key=row["key"],
            value=row["value"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            score=score,
        ),
        reason=reason,
        version=int(row["version"]),
        valid_at=row["valid_at"],
        provenance=row["provenance"],
        confirmed_at=row["confirmed_at"],
        confirmations=int(row["confirmations"]),
    )


def _candidate_order(candidate: Candidate) -> tuple[int, bool, float, str]:
    """Sort key of candidates: strongest reason, then best score (none last), then key."""
    score = candidate.score
    return (
        _REASON_RANK[candidate.reason],
        score is None,
        0.0 if score is None else -score,
        candidate.key,
    )


def merge_candidates(
    namespace: tuple[str, ...],
    key: str,
    key_slot_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Sequence[Mapping[str, Any]]],
    similar_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    limit: int,
) -> tuple[Candidate, ...]:
    """Merge the rows of the candidate statements into one candidate per key.

    In ``key_slot_rows``, the rows of StoreSQL.key_slot_candidates, the row
    under ``key`` is the target (reason "key") and the others are slot
    matches. Each list in ``score_rows`` holds the rows of
    StoreSQL.candidate_scores for one vector of the new memory, and each list
    in ``similar_rows`` those of StoreSQL.similar_candidates. A key takes its
    best score over all of them, so a key or slot match has a score whenever
    it has vectors. Key matches come first, then slot matches, then similar
    ones; then the best score, with no score last; then the key. ``limit``
    caps the similar candidates only, never the key and slot matches.

    A similar row under ``key`` is left out when the key read found no
    target: the target is found by its key alone. So is the row without a
    key that an index scan returns when no candidate passes.
    """
    scores: dict[str, float] = {}
    for rows in score_rows:
        for row in rows:
            scores[row["key"]] = max(float(row["score"]), scores.get(row["key"], -math.inf))
    # The best similar row of each key, over every vector of the new memory
    similar: dict[str, tuple[float, Mapping[str, Any]]] = {}
    for rows in similar_rows:
        for row in rows:
            if row["key"] is None:
                continue
            score = float(row["score"])
            if row["key"] not in similar or score > similar[row["key"]][0]:
                similar[row["key"]] = (score, row)

    matches: list[Candidate] = []
    for row in key_slot_rows:
        found = [scores[row["key"]]] if row["key"] in scores else []
        if row["key"] in similar:
            found.append(similar.pop(row["key"])[0])
        reason: Literal["key", "slot"] = "key" if row["key"] == key else "slot"
        matches.append(_row_to_candidate(namespace, row, reason, max(found, default=None)))
    similar.pop(key, None)
    others = [
        _row_to_candidate(namespace, row, "similar", score) for score, row in similar.values()
    ]
    matches.sort(key=_candidate_order)
    others.sort(key=_candidate_order)
    return tuple(matches + others[:limit])


def default_valid_at(now: datetime, candidates: Sequence[Candidate]) -> datetime:
    """The valid time of a new memory given without one.

    The database's ``now``, or the latest updated_at of the candidates if
    that is later. The database's clock sets every updated_at, but the node
    that wrote a candidate may have a clock ahead of the node that read
    ``now``, and a correction must not look older than the fact it corrects.
    A candidate's valid_at is never taken: a valid time a caller set in the
    future still counts as newer than the new memory.
    """
    return max([now, *(candidate.item.updated_at for candidate in candidates)])


def _acted(candidates: Sequence[Candidate], decision: Decision, action: Action) -> list[str]:
    """The keys of the candidates that ``decision`` gives ``action``, in candidate order."""
    return [c.key for c in candidates if decision.action(c.key) == action]


def changed_keys(
    expected: Mapping[str, tuple[int, datetime] | None],
    rows: Iterable[Mapping[str, Any]],
    *,
    may_be_gone: Collection[str] = (),
) -> tuple[str, ...]:
    """The keys whose live memory is not the one that was read, in the order of ``expected``.

    ``expected`` maps each key to the (version, updated_at) of its live
    memory when it was read, or to None when no live memory had the key.
    ``rows`` are the live memories now, with ``key``, ``version`` and
    ``updated_at``, as StoreSQL.tokens reads them. A key changed when its
    memory was written since, which moves the version, or only updated_at
    when a 0.5.x store wrote it; when it was deleted or expired; and when a
    key that had no live memory has one now. Confirming a memory changes
    neither the version nor updated_at, so it never counts. A token that
    came back from a checkpoint as a list compares as the tuple it was.

    A key in ``may_be_gone`` whose memory is gone counts as unchanged, as
    for the retry of a write that deletes it, whose earlier attempt may have
    committed. A live memory under it that differs still counts.
    """
    live = {row["key"]: (int(row["version"]), row["updated_at"]) for row in rows}
    return tuple(
        key
        for key, token in expected.items()
        if not (key in may_be_gone and key not in live)
        and live.get(key) != (None if token is None else tuple(token))
    )


class BaseCockroachDBStore:
    """State and planning shared by CockroachDBStore and its async twin."""

    # Names the store in log and error messages of the psycopg machinery
    component: ClassVar[str] = "store"
    _log: ClassVar[logging.Logger] = logger
    # Seconds a plain write trusts a schema check that found no migration 2,
    # or no history table. Another process may run setup() meanwhile, so
    # after that the next write reads the schema again. What was found is
    # trusted for good.
    _SCHEMA_RECHECK_SECONDS: ClassVar[float] = 60.0

    sql: StoreSQL
    index: ResolvedIndex | None
    ttl: ResolvedTTL
    # The most vectors a namespace may hold for find_conflicts and reconcile
    # to rank all of them. Above it they use the C-SPANN index
    reconcile_scan_limit: int
    # Version history settings, or None for a store without history
    _history: ResolvedHistory | None
    # Whether a TTL configuration was given. Without one the store never
    # refreshes an expiry and setup() leaves the table TTL settings alone.
    _ttl_given: bool
    _warned_query_without_index: bool
    # Pooled batches run concurrently; the lock makes the warning appear once
    _warning_lock: threading.Lock
    # Whether migration 2's columns exist, and whether the history table
    # does. None until read. On a store with history the schema check
    # (_schema, or _aschema on the async store) reads both; a store without
    # history reads the table only for delete_namespace. setup() clears them
    # once it has created what they describe.
    _columns_ready: bool | None
    _history_ready: bool | None
    # When the schema check last read them, by time.monotonic()
    _schema_read_at: float
    # Pooled batches check the schema concurrently. The lock keeps a read
    # that finishes late from replacing a True another read cached.
    _schema_lock: threading.Lock
    # Whether the store's role was denied the migrations table (SQLSTATE
    # 42501). From then on the schema check reads the items table's columns
    # instead, for good: grants rarely change, and that read is always
    # allowed for the role.
    _migrations_denied: bool
    # Whether the fallback to the column read has been logged, to log it once
    _logged_column_fallback: bool
    # Marks the threads that are running a reconcile policy of this store,
    # as ``inside``. A store call from a policy is refused (_refuse_inside_policy)
    _policy_marks: threading.local

    def _configure(
        self,
        *,
        index: ResolvedIndex | None,
        ttl: Any,
        table_name: str,
        schema: str,
        history: ResolvedHistory | None = None,
        reconcile_scan_limit: int = DEFAULT_RECONCILE_SCAN_LIMIT,
    ) -> None:
        """Set the shared state from validated settings."""
        self.index = index
        self.ttl = resolve_ttl_config(ttl)
        self.reconcile_scan_limit = reconcile_scan_limit
        self._history = history
        self._ttl_given = ttl is not None
        self.sql = StoreSQL(schema, table_name, index=index)
        self._warned_query_without_index = False
        self._warning_lock = threading.Lock()
        self._columns_ready = None
        self._history_ready = None
        self._schema_read_at = 0.0
        self._schema_lock = threading.Lock()
        self._migrations_denied = False
        self._logged_column_fallback = False
        self._policy_marks = threading.local()

    def _trusted(self, ready: bool | None, *, refresh: bool) -> bool | None:
        """A cached schema answer, or None when the schema check must read it again.

        True holds for good: migrations are never undone, and setup() never
        drops a table. False holds for _SCHEMA_RECHECK_SECONDS after the read
        that found it, because another process may run setup() meanwhile,
        and not at all with ``refresh``.
        """
        if ready is False and (
            refresh or time.monotonic() - self._schema_read_at >= self._SCHEMA_RECHECK_SECONDS
        ):
            return None
        return ready

    def _cached_columns(self, *, refresh: bool) -> bool | None:
        """Whether migration 2's columns exist, if the cache can answer, else None."""
        return self._trusted(self._columns_ready, refresh=refresh)

    def _cached_history(self, *, refresh: bool) -> bool | None:
        """Whether the history table exists, if the cache can answer, else None.

        A store without history never looks for the table: always False.
        """
        if self._history is None:
            return False
        return self._trusted(self._history_ready, refresh=refresh)

    def _cache_columns(self, rows: Sequence[Mapping[str, Any]]) -> bool:
        """Cache and return whether migration 2's columns exist.

        ``rows`` are what the schema check's first read returned, as
        ``ready``: StoreSQL.reconcile_migration_recorded, or
        reconcile_columns_exist once the migrations table was denied. A True
        cached before is kept: a read that started before another process
        ran setup() can finish late.
        """
        with self._schema_lock:
            ready = bool(rows and rows[0]["ready"]) or self._columns_ready is True
            self._schema_read_at = time.monotonic()
            self._columns_ready = ready
        return ready

    def _cache_history(self, rows: Sequence[Mapping[str, Any]]) -> bool:
        """Cache and return whether the history table exists.

        ``rows`` are what StoreSQL.history_exists returned. A True cached
        before is kept, as in _cache_columns. On a store with history the
        answer is part of the schema check. A store without history reads it
        only for delete_namespace, so that read leaves the schema check's
        clock alone: a missing migration 2 is not trusted any longer for it.
        """
        with self._schema_lock:
            ready = bool(rows and rows[0]["ready"]) or self._history_ready is True
            if self._history is not None:
                self._schema_read_at = time.monotonic()
            self._history_ready = ready
        return ready

    def _schema_statements(self, *, read_migrations: bool) -> list[tuple[str, list]]:
        """The schema check's reads, sent as one round trip.

        Whether migration 2 is in place, and on a store with history whether
        the history table exists. With ``read_migrations`` the first read is
        the migrations table. setup() records migration 2 there only after
        its schema change has settled on every node, so a store that finds
        the row can use the new columns wherever its next statement runs.

        A role with no grant on that table reads the items table's own
        columns instead (see _fall_back_to_columns). They can show up a
        moment before every node uses them, so on a multi-node cluster one
        write of such a role can fail in the seconds after migration 2
        completes. Run setup() before rolling out 0.6.0, or grant the role
        SELECT on the migrations table, to avoid that.
        """
        if read_migrations:
            statements = [self.sql.reconcile_migration_recorded()]
        else:
            statements = [self.sql.reconcile_columns_exist()]
        if self._history is not None:
            statements.append(self.sql.history_exists())
        return statements

    def _fall_back_to_columns(self, error: BaseException, *, read_migrations: bool) -> bool:
        """Whether a schema check denied with SQLSTATE 42501 runs again with the column read.

        Only a denied migrations read falls back. The store remembers the
        denial, so its later checks go straight to the column read without
        the error. The failed statement ended its round trip, so the column
        read and the history check go out together in the next one. Inside a
        transaction the caller opened, the denial aborted that transaction
        and no read can run in it: the error goes to the caller, whose retry
        reads the columns.
        """
        if not read_migrations:
            return False
        if not self._migrations_denied:
            with self._warning_lock:
                if not self._logged_column_fallback:
                    self._log.info(
                        "The store's role has no SELECT on %s. The schema check now reads the "
                        "items table's columns.",
                        self.sql.migrations_table,
                    )
                    self._logged_column_fallback = True
        self._migrations_denied = True
        return not _psycopg_conn.in_caller_transaction(error)

    def _cached_schema(self, *, refresh: bool) -> SchemaCheck | None:
        """What the schema check can answer without a read, or None to read."""
        columns = self._cached_columns(refresh=refresh)
        history = self._cached_history(refresh=refresh)
        if columns is None or history is None:
            return None
        return SchemaCheck(columns=columns, history=history)

    def _cache_schema(self, found: Sequence[Sequence[Mapping[str, Any]]]) -> SchemaCheck:
        """Cache and return what the reads of _schema_statements found."""
        columns = self._cache_columns(found[0])
        history = self._cache_history(found[1]) if self._history is not None else False
        return SchemaCheck(columns=columns, history=history)

    def _require_columns(self, ready: bool) -> None:
        """Raise ValueError unless setup() has applied migration 2.

        ``ready`` is what the schema check returned, run with refresh=True:
        _reconcile_columns, or _areconcile_columns on the async store. A
        store that found no migration 2 earlier then sees a setup() that
        another process ran since.

        Raises:
            ValueError: If the items table lacks the columns that
                reconciliation needs.
        """
        if not ready:
            raise ValueError(
                f"The table {self.sql.items} does not have the columns this method "
                "needs yet; run setup() to add them."
            )

    def _require_history(self) -> None:
        """Raise ValueError unless the store keeps history.

        Raises:
            ValueError: If the store was created without history.
        """
        if self._history is None:
            raise ValueError(
                "This store keeps no history. Create it with history=True, or a "
                "HistoryConfig, and run setup() to create the history table."
            )

    def _require_history_table(self, schema: SchemaCheck) -> None:
        """Raise ValueError unless setup() has made the store ready to keep history.

        ``schema`` is what the schema check returned, read with refresh=True,
        so a setup() that another process ran since is seen.

        Raises:
            ValueError: If migration 2's columns or the history table are missing.
        """
        if not schema.columns:
            raise ValueError(
                f"The table {self.sql.items} does not have the columns that history "
                "needs yet; run setup() to add them and to create the history table "
                f"{self.sql.history}."
            )
        if not schema.history:
            raise ValueError(
                f"The history table {self.sql.history} does not exist yet; "
                "run setup() to create it."
            )

    def _refresh(self, refresh_ttl: bool) -> bool:
        """Whether a read pushes back the expiry of the rows it returns.

        BaseStore resolves refresh_ttl to true when the store has no TTL
        configuration. Such a store still leaves expiry alone, so it never
        extends rows that another store wrote with a TTL.
        """
        return bool(refresh_ttl) and self._ttl_given

    def _plan_batch(self, ops: Iterable[Op]) -> BatchPlan:
        """Validate a batch and build all its statements.

        Everything that can fail without the database fails here, before
        any statement runs: labels, paging, and values that are not JSON.
        """
        grouped, total = _group_ops(ops)
        plan = BatchPlan(results=[None] * total)

        groups: dict[tuple[tuple[str, ...], bool], list[tuple[int, GetOp]]] = defaultdict(list)
        for i, op in grouped.get(GetOp, []):
            if isinstance(op, GetOp):
                groups[(tuple(op.namespace), self._refresh(op.refresh_ttl))].append((i, op))
        for i, op in grouped.get(SearchOp, []):
            if not isinstance(op, SearchOp):
                continue
            if op.limit == 0:
                plan.results[i] = []
            else:
                plan.searches.append((i, op))
        plan.listings = [
            (i, op)
            for i, op in grouped.get(ListNamespacesOp, [])
            if isinstance(op, ListNamespacesOp)
        ]
        puts = [op for _, op in grouped.get(PutOp, []) if isinstance(op, PutOp)]

        if puts:
            plan.puts = self._plan_puts(puts)
        if plan.searches:
            plan.search_statements, plan.search_to_embed, plan.search_lookups = self._search_plan(
                plan.searches
            )
            if plan.search_to_embed and self.index is not None and self.index.beam_size:
                plan.search_session = [self.sql.beam_size_statement(self.index.beam_size)]
        plan.listing_statements = [self._listing_statement(op) for _, op in plan.listings]
        # One statement per namespace and refresh choice. With refresh, the
        # statement also pushes back the expiry of the rows it reads.
        plan.get_groups = list(groups.items())
        plan.get_statements = [
            self.sql.get(
                to_text(namespace),
                list(dict.fromkeys(op.key for _, op in items)),
                omit_expired=self.ttl.omit_expired,
                refresh=refresh,
            )
            for (namespace, refresh), items in plan.get_groups
        ]
        return plan

    def _plan_puts(self, put_ops: Iterable[PutOp]) -> PutPlan:
        """Plan the rows a batch of puts writes.

        The last op for a (namespace, key) wins. Each item gets the op's TTL
        as given, like PostgresStore: BaseStore.put already filled in the
        configured default, so None means no expiry. Values are serialized here,
        so an unserializable value fails the batch before anything is
        embedded or sent.

        Raises:
            ValueError: If a value cannot be serialized as JSON.
        """
        latest: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for op in put_ops:
            latest[(tuple(op.namespace), op.key)] = op

        plan = PutPlan()
        # One text per vector row: a path listed twice is embedded once
        requests: dict[tuple[str, str, str], str] = {}
        for (namespace, key), op in latest.items():
            prefix = to_text(namespace)
            if op.value is None:
                plan.deletes.setdefault(prefix, []).append(key)
                continue
            try:
                value_json = json.dumps(op.value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"value for {prefix}/{key} is not JSON serializable: {error}"
                ) from error
            plan.items.append((prefix, key, value_json, op.ttl))
            if self.index is None:
                continue
            plan.reindexed.setdefault(prefix, []).append(key)
            if op.index is False:
                continue
            paths: Sequence[tuple[str, Any]] = (
                self.index.fields
                if op.index is None
                else [(path, tokenize_path(path)) for path in op.index]
            )
            for path, tokens in paths:
                texts = get_text_at_path(op.value, tokens)
                if len(texts) > 1:
                    for i, text in enumerate(texts):
                        requests[(prefix, key, f"{path}.{i}")] = text
                elif texts:
                    requests[(prefix, key, path)] = texts[0]
        plan.embed_requests = [
            (prefix, key, field_name, text) for (prefix, key, field_name), text in requests.items()
        ]
        return plan

    def _vector_rows(self, plan: PutPlan, embeddings: Sequence[Sequence[float]]) -> list[VectorRow]:
        """Pair the embeddings of ``plan.embed_requests`` with their rows.

        Each vector row gets the TTL of its item, so vectors expire with it.

        Args:
            plan: The put plan whose texts were embedded.
            embeddings: One embedding per embed request, in the same order.

        Raises:
            ValueError: If the model returned the wrong number of embeddings,
                an embedding whose size is not the index's dims, or one with
                NaN or infinite values.
        """
        if self.index is None or not plan.embed_requests:
            return []
        if len(embeddings) != len(plan.embed_requests):
            raise ValueError(
                f"The embedding model returned {len(embeddings)} embeddings for "
                f"{len(plan.embed_requests)} texts"
            )
        ttls = {(prefix, key): ttl for prefix, key, _, ttl in plan.items}
        rows: list[VectorRow] = []
        # The length check above keeps the two lists in step
        for i, (prefix, key, field_name, _) in enumerate(plan.embed_requests):
            embedding = embeddings[i]
            self._check_embedding(embedding, f"{prefix}/{key}, field {field_name}")
            rows.append((prefix, key, field_name, list(embedding), ttls[(prefix, key)]))
        return rows

    def _check_embedding(self, embedding: Sequence[float], what: str) -> None:
        """Raise ValueError if an embedding does not fit the index.

        Raises:
            ValueError: If its size is not the index's dims, or it holds NaN
                or infinite values.
        """
        if self.index is None:
            return
        if len(embedding) != self.index.dims:
            raise ValueError(
                f"The embedding model returned a vector of {len(embedding)} dimensions, "
                f"but the index has {self.index.dims}"
            )
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError(
                f"The embedding model returned a vector with NaN or infinite values for {what}"
            )

    def _bind_query_vectors(
        self, plan: BatchPlan, texts: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Check the embeddings of ``plan.query_texts()`` and bind them.

        Raises:
            ValueError: If the model returned the wrong number of embeddings,
                or one that does not fit the index.
        """
        if len(vectors) != len(texts):
            raise ValueError(
                f"The embedding model returned {len(vectors)} embeddings for {len(texts)} queries"
            )
        for text, vector in zip(texts, vectors, strict=True):
            self._check_embedding(vector, f"the query {text!r}")
        plan.bind_query_vectors(dict(zip(texts, vectors, strict=True)))

    def _search_plan(
        self, search_ops: Sequence[tuple[int, SearchOp]]
    ) -> tuple[list[tuple[str, list]], list[tuple[int, str]], list[NamespaceLookup]]:
        """Build one statement per search op.

        A semantic search under a namespace on a store with a vector index
        gets a namespace lookup too, an exact fallback for too many
        namespaces, and an exact form to run when its indexed pass comes
        back short. A search of the whole store, or any search on a flat
        store, runs the exact form at once. See the notes above
        StoreSQL.semantic_search.

        Returns:
            The statements, in the order of ``search_ops``; the
            (statement position, query text) pairs whose QUERY_VECTOR
            placeholders need the query's embedding; and the lookups whose
            namespaces fill the NAMESPACES placeholders.
        """
        statements: list[tuple[str, list]] = []
        to_embed: list[tuple[int, str]] = []
        lookups: list[NamespaceLookup] = []
        omit_expired = self.ttl.omit_expired
        for position, (_, op) in enumerate(search_ops):
            if op.query and self.index is not None:
                refresh = self._refresh(op.refresh_ttl)
                exact = self.sql.semantic_search(
                    namespace_prefix=op.namespace_prefix,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    omit_expired=omit_expired,
                    refresh=refresh,
                )
                to_embed.append((position, op.query))
                if not op.namespace_prefix or self.index.flat:
                    # No vector index to narrow down to: one exact scan
                    statements.append(exact)
                    continue
                # Room for every vector of each result up to the end of the
                # page, twice. PostgresStore leaves out the offset and runs
                # out of candidates on deep pages
                over_fetch = (op.limit + op.offset) * self.index.estimated_vectors * 2 + 1
                statements.append(
                    self.sql.semantic_search_in(
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        omit_expired=omit_expired,
                        over_fetch=over_fetch,
                        refresh=refresh,
                    )
                )
                # If both passes run with refresh, the first may push back the
                # expiry of an item the exact pass leaves out of the page
                lookups.append(
                    NamespaceLookup(
                        position=position,
                        statement=self.sql.namespace_lookup(op.namespace_prefix),
                        fallback=exact,
                        exact=self.sql.exact_semantic_search_in(
                            filter=op.filter,
                            limit=op.limit,
                            offset=op.offset,
                            omit_expired=omit_expired,
                            refresh=refresh,
                        ),
                        limit=op.limit,
                        over_fetch=over_fetch,
                    )
                )
                continue
            if op.query and self._first_query_without_index():
                self._log.warning(
                    "A search query was given, but the store has no index configured. "
                    "Results are filtered and ordered by updated_at, not ranked by "
                    "similarity. Pass index= to the store for semantic search."
                )
            statements.append(
                self.sql.search(
                    namespace_prefix=op.namespace_prefix,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    omit_expired=omit_expired,
                    refresh=self._refresh(op.refresh_ttl),
                )
            )
        return statements, to_embed, lookups

    def _check_vector_index(self, create_statement: str) -> None:
        """Compare the vectors table and its index with the store's configuration.

        ``create_statement`` is the SHOW CREATE TABLE output of the vectors
        table, read after setup() ran the vector statements. Those use IF NOT
        EXISTS, so they leave an existing table and index as they are.

        Raises:
            ValueError: If the VECTOR column has other dims, or the index uses
                another operator class (distance).
        """
        if self.index is None:
            return
        column = re.search(r"\bembedding VECTOR\((\d+)\)", create_statement, re.IGNORECASE)
        if column is not None and int(column.group(1)) != self.index.dims:
            raise ValueError(
                f"The vectors table {self.sql.vectors} has an embedding column of "
                f"VECTOR({column.group(1)}), but the index is configured with "
                f"dims={self.index.dims}. Use dims={column.group(1)}, or another table_name."
            )
        name = self.sql.vector_index_name
        existing = _find_vector_index(create_statement, name)
        if existing is None:
            return
        if self.index.flat:
            self._log.warning(
                "The index is configured with flat=True, but the vector index %s exists. "
                "Searches do not use it, but every put still updates it. Drop it with "
                "drop_vector_index() if no other store needs it.",
                name,
            )
            return
        wanted = self.index.distance.get_opclass()
        found = re.search(r"\bvector_\w+_ops\b", existing.definition)
        # vector_l2_ops is the default operator class
        current = found.group(0) if found else "vector_l2_ops"
        if current != wanted:
            raise ValueError(
                f"the vector index {name} uses {current}; drop it with drop_vector_index() "
                f"and run setup() again to use {wanted}"
            )
        # A partition size that is not configured accepts whatever the index has
        differences = []
        for option, size in self.index.cspann.items():
            current_size = existing.options.get(option)
            if current_size is None:
                differences.append(f"{option} = {size} (index has the default)")
            elif current_size != str(size):
                differences.append(f"{option} = {size} (index has {current_size})")
        if differences:
            self._log.warning(
                "The vector index %s already exists with other partition sizes than "
                "configured: %s. It keeps its own. Drop it with drop_vector_index() and run "
                "setup() again to use the configured ones.",
                name,
                "; ".join(differences),
            )

    def _first_query_without_index(self) -> bool:
        """True the first time a search query meets a store without an index."""
        with self._warning_lock:
            first = not self._warned_query_without_index
            self._warned_query_without_index = True
        return first

    def _write_statements(
        self, plan: PutPlan, vector_rows: Sequence[VectorRow] = ()
    ) -> list[tuple[str, list]]:
        """The statements of a put batch, in the order they must run.

        Old vectors of re-indexed items go first, then the item upserts, the
        vector upserts, and the deletes with their vectors. ``vector_rows``
        come from ``_vector_rows``, computed once before the first attempt.
        With a write id on the plan, the item upserts store it and bump each
        row's version. Without one they are the 0.5 statements. On a store
        with history, the archive statements come before all of them.
        """
        keeps_history = self._history is not None and plan.write_id is not None
        statements: list[tuple[str, list]] = []
        if self._history is not None and plan.write_id is not None:
            statements.extend(
                self._archive_statements(
                    plan, write_id=plan.write_id, ttl_days=self._history.ttl_days
                )
            )
        statements.extend(self._reindex_deletes(plan))
        rows: Sequence[ItemRow] | Sequence[ItemRowV2] = plan.items
        if plan.write_id is not None:
            # A plain put writes no provenance and no valid time
            rows = [(prefix, key, value, ttl, None, None) for prefix, key, value, ttl in plan.items]
        for start in range(0, len(rows), ITEMS_PER_STATEMENT):
            statements.append(
                self.sql.upsert_items(
                    rows[start : start + ITEMS_PER_STATEMENT],
                    write_id=plan.write_id,
                    history=keeps_history,
                )
            )
        statements.extend(self._vector_upserts(vector_rows))
        for prefix, keys in plan.deletes.items():
            statements.extend(self._delete_statements(prefix, keys))
        return statements

    def _reindex_deletes(self, plan: PutPlan) -> list[tuple[str, list]]:
        """The deletes of the old vectors of every item a put plan re-indexes."""
        return [self.sql.delete_vectors(prefix, keys) for prefix, keys in plan.reindexed.items()]

    def _vector_upserts(self, vector_rows: Sequence[VectorRow]) -> list[tuple[str, list]]:
        """The upserts of a write's vectors, VECTORS_PER_STATEMENT rows to a statement."""
        return [
            self.sql.upsert_vectors(vector_rows[start : start + VECTORS_PER_STATEMENT])
            for start in range(0, len(vector_rows), VECTORS_PER_STATEMENT)
        ]

    def _delete_statements(self, prefix: str, keys: Sequence[str]) -> list[tuple[str, list]]:
        """The deletes of some items of a namespace, with their vectors on a store with an index."""
        statements = [self.sql.delete_items(prefix, keys)]
        if self.index is not None:
            statements.append(self.sql.delete_vectors(prefix, keys))
        return statements

    def _archive_statements(
        self, plan: PutPlan, *, write_id: str, ttl_days: float | None
    ) -> list[tuple[str, list]]:
        """The statements that keep the versions a put batch ends, on a store with history.

        Each put key's live version is archived as overwritten and each
        deleted key's as deleted, both invalid from now. Keys with no live
        row archive nothing. One statement per namespace and reason.

        Args:
            plan: The put plan.
            write_id: The plan's write id.
            ttl_days: The history's TTL in days, or None.
        """
        overwritten: dict[str, list[str]] = {}
        for prefix, key, _, _ in plan.items:
            overwritten.setdefault(prefix, []).append(key)
        return [
            self.sql.archive(
                prefix,
                keys,
                reason=reason,
                superseded_by=None,
                invalid_at=None,
                write_id=write_id,
                ttl_days=ttl_days,
            )
            for reason, by_prefix in (("overwritten", overwritten), ("deleted", plan.deletes))
            for prefix, keys in by_prefix.items()
        ]

    def _listing_statement(self, op: ListNamespacesOp) -> tuple[str, list]:
        """The statement for one list_namespaces op.

        Prefix conditions also bound the rows read to the key span of their
        labels before the first ``*``.
        """
        conditions = op.match_conditions or ()
        return self.sql.list_namespaces(
            match_conditions=[
                match_pattern(condition.path, condition.match_type) for condition in conditions
            ],
            max_depth=op.max_depth,
            limit=op.limit,
            offset=op.offset,
            omit_expired=self.ttl.omit_expired,
            prefixes=[
                literal_prefix(condition.path)
                for condition in conditions
                if condition.match_type == "prefix"
            ],
        )

    def _plan_proposal(
        self,
        namespace: Sequence[str],
        key: str | None,
        value: dict[str, Any],
        *,
        slot: str | None,
        threshold: float | None,
        limit: int,
        filter: Mapping[str, Any] | None,
        index: Literal[False] | list[str] | None,
        ttl: float | None | NotProvided,
        provenance: dict[str, Any] | None,
        valid_at: datetime | None,
        method: str,
    ) -> ProposalPlan:
        """Check a new memory and plan its write, before anything is embedded or sent.

        The arguments are those of find_conflicts. The key, a uuid4 when
        None, and the write id are chosen here, once, so every attempt reuses
        them. The write is one put: the value serialized, the TTL (the
        store's default when not provided), and the texts to embed.

        Args:
            method: The calling method, named in errors.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid.
            ValueError: If another argument is invalid, as find_conflicts
                describes.
        """
        prefix = _memory_prefix(namespace, method)
        if not isinstance(value, dict):
            raise ValueError(f"{method} needs the value as a dict, got {type(value).__name__}")
        _check_limit(limit)
        resolved_threshold = self._proposal_threshold(threshold, index, method)
        _check_filter(filter)
        if isinstance(ttl, NotProvided):
            minutes = self.ttl.default_ttl
        else:
            _check_ttl(ttl)
            minutes = ttl
        provenance_value, provenance_json = _serialized_provenance(provenance)
        _check_valid_at(valid_at)

        labels = tuple(namespace)
        memory_key = str(uuid.uuid4()) if key is None else str(key)
        put = self._plan_puts([PutOp(labels, memory_key, value, index=index, ttl=minutes)])
        put.write_id = str(uuid.uuid4())
        value_json = put.items[0][2]
        plain = json.loads(value_json)
        slot_value, slot_value_json = _slot_of(plain, slot)
        return ProposalPlan(
            namespace=labels,
            prefix=prefix,
            key=memory_key,
            value=plain,
            value_json=value_json,
            slot=slot,
            slot_value=slot_value,
            slot_value_json=slot_value_json,
            threshold=resolved_threshold,
            limit=limit,
            filter=filter,
            index=index,
            ttl=minutes,
            provenance=provenance_value,
            provenance_json=provenance_json,
            valid_at=valid_at,
            put=put,
        )

    def _proposal_threshold(
        self, threshold: float | None, index: Literal[False] | list[str] | None, method: str
    ) -> float | None:
        """The lowest score of a similar candidate, or None when none are searched.

        A store without an index, or a call with index=False, searches none,
        and then the threshold is ignored. Scores come from the store's
        score function: 1 minus the distance on a cosine store, the negated
        distance on an l2 or inner_product store.

        Raises:
            ValueError: If the threshold is missing on an l2 or inner_product
                store, is not a finite number, or lies outside -1 to 1 on a
                cosine store.
        """
        if self.index is None or index is False:
            return None
        cosine = self.index.distance == DistanceStrategy.COSINE
        if threshold is None:
            if cosine:
                return DEFAULT_COSINE_THRESHOLD
            raise ValueError(
                f"{method} needs a threshold on a store whose index uses l2 or "
                "inner_product distance. Scores there are negated distances, so no "
                "default fits: pass the lowest score a similar memory may have."
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise ValueError(f"threshold must be a finite number, got {threshold!r}")
        if cosine and not -1 <= threshold <= 1:
            raise ValueError(
                f"threshold must be between -1 and 1 on a cosine store, got {threshold!r}"
            )
        return float(threshold)

    def _identity(self) -> StoreIdentity:
        """Which store this is, as Conflicts records it: the items table and the index."""
        if self.index is None:
            return StoreIdentity(items=self.sql.items, dims=None, distance=None, fields=None)
        return StoreIdentity(
            items=self.sql.items,
            dims=self.index.dims,
            distance=self.index.distance.value,
            fields=tuple(path for path, _ in self.index.fields),
        )

    def _chooses_scan(self, plan: ProposalPlan) -> bool:
        """Whether the similar read runs and must choose between the exact scan and the index.

        A flat store has no index to choose. A limit of 0, or a new memory
        without vectors, needs no similar read.
        """
        return (
            self.index is not None
            and not self.index.flat
            and plan.threshold is not None
            and plan.limit > 0
            and bool(plan.embed_requests)
        )

    def _key_slot_statements(self, plan: ProposalPlan) -> list[tuple[str, list]]:
        """The first candidate read, to send as one round trip.

        The database's now(), the target and the slot matches, and, when the
        similar read must choose a scan, the namespace's vectors counted up
        to one past reconcile_scan_limit. Pass the rows to _key_slot_read.
        """
        statements = [
            self.sql.now(),
            self.sql.key_slot_candidates(
                plan.prefix,
                plan.key,
                slot_field=plan.slot,
                slot_value_json=plan.slot_value_json,
                filter=plan.filter,
            ),
        ]
        if self._chooses_scan(plan):
            statements.append(
                self.sql.namespace_vector_count(plan.prefix, self.reconcile_scan_limit + 1)
            )
        return statements

    def _key_slot_read(
        self, plan: ProposalPlan, found: Sequence[Sequence[dict[str, Any]]]
    ) -> KeySlotRead:
        """What the first candidate read found, from the rows of _key_slot_statements."""
        vectors = int(found[2][0]["n"]) if self._chooses_scan(plan) else None
        return KeySlotRead(now=found[0][0]["now"], rows=list(found[1]), vectors=vectors)

    def _similar_read(
        self, plan: ProposalPlan, vector_rows: Sequence[VectorRow], read: KeySlotRead
    ) -> SimilarRead:
        """The second candidate read, built from what the first found.

        For each vector of the new memory, one statement scores the target
        and the slot matches, and one finds the similar candidates, leaving
        those keys and the new memory's own key out. The namespace's vectors
        are ranked exactly when it holds at most reconcile_scan_limit of
        them, or the index is flat. Otherwise the C-SPANN index gives a
        window of the store search's over_fetch for ``limit``, with the beam
        size set when the index configures one, and the exact form of each
        statement is kept for a window that may have left matches out. Empty
        when the new memory has no vectors. Run it as SimilarRead says, and
        pass the rows to _conflicts.
        """
        if not vector_rows:
            return SimilarRead(scores=[], similar=[], session=[])
        keys = [row["key"] for row in read.rows]
        scores: list[tuple[str, list]] = []
        if keys:
            statement = self.sql.candidate_scores(plan.prefix, keys)
            scores = [StoreSQL.bind_query_vector(statement, row[3]) for row in vector_rows]
        index, threshold = self.index, plan.threshold
        if index is None or threshold is None or plan.limit == 0:
            return SimilarRead(scores=scores, similar=[], session=[])
        above = read.vectors is not None and read.vectors > self.reconcile_scan_limit
        scans: list[Literal["exact", "index"]] = ["exact", "index"] if above else ["exact"]
        over_fetch = plan.limit * index.estimated_vectors * 2 + 1
        built: dict[str, list[tuple[str, list]]] = {}
        for scan in scans:
            statement = self.sql.similar_candidates(
                plan.prefix,
                threshold=threshold,
                limit=plan.limit,
                filter=plan.filter,
                scan=scan,
                over_fetch=over_fetch,
                exclude=list(dict.fromkeys([plan.key, *keys])),
            )
            built[scan] = [StoreSQL.bind_query_vector(statement, row[3]) for row in vector_rows]
        if not above:
            return SimilarRead(
                scores=scores,
                similar=built["exact"],
                session=[],
                threshold=threshold,
                limit=plan.limit,
            )
        session = [self.sql.beam_size_statement(index.beam_size)] if index.beam_size else []
        return SimilarRead(
            scores=scores,
            similar=built["index"],
            session=session,
            exact=built["exact"],
            threshold=threshold,
            limit=plan.limit,
            over_fetch=over_fetch,
        )

    def _conflicts(
        self,
        plan: ProposalPlan,
        vector_rows: Sequence[VectorRow],
        read: KeySlotRead,
        similar: SimilarRead,
        found: Sequence[Sequence[dict[str, Any]]],
    ) -> Conflicts:
        """Merge what both candidate reads found into Conflicts.

        ``found`` are the rows of ``similar.statements()``, in order, after
        ``similar.fill_exact``, or nothing when there were none. Without a
        valid_at from the caller, the new memory is valid from
        default_valid_at. The state for resolve() records which store made
        it, and the target's and each candidate's (version, updated_at).
        """
        split = len(similar.scores)
        candidates = merge_candidates(
            plan.namespace, plan.key, read.rows, found[:split], found[split:], limit=plan.limit
        )
        target = next((row for row in read.rows if row["key"] == plan.key), None)
        token = None if target is None else (int(target["version"]), target["updated_at"])
        valid_at = plan.valid_at
        if valid_at is None:
            valid_at = default_valid_at(read.now, candidates)
        proposal = Proposal(
            namespace=plan.namespace,
            key=plan.key,
            value=plan.value,
            slot=plan.slot,
            slot_value=plan.slot_value,
            provenance=plan.provenance,
            valid_at=valid_at,
        )
        state = ConflictsState(
            store=self._identity(),
            plan=plan,
            vector_rows=tuple(vector_rows),
            target=token,
            tokens={c.key: (c.version, c.item.updated_at) for c in candidates},
        )
        return Conflicts(
            proposal=proposal,
            candidates=candidates,
            target_version=None if token is None else token[0],
            target_updated_at=None if token is None else token[1],
            _state=state,
        )

    @staticmethod
    def _checked_decision(
        proposal: Proposal, candidates: Sequence[Candidate], decision: object
    ) -> Decision:
        """The policy's decision, once it is a valid Decision for ``proposal`` and ``candidates``.

        Raises:
            ValueError: If the policy returned something other than a
                Decision, or a decision that validate_decision rejects.
        """
        if not isinstance(decision, Decision):
            raise ValueError(f"The policy returned {type(decision).__name__}, not a Decision")
        validate_decision(proposal, candidates, decision)
        return decision

    def _decide(
        self, policy: Policy, proposal: Proposal, candidates: Sequence[Candidate]
    ) -> object:
        """Run ``policy`` with this thread marked as inside a policy of this store.

        The policy runs in the reconcile's transaction, while the store holds
        its connection. On a single connection a store call from the policy
        would wait for that connection forever, and on a pool it would read
        outside the transaction. The mark makes such a call fail at once
        instead (_refuse_inside_policy). It ends with the policy, however
        the policy ends, and other threads never see it.
        """
        marks = self._policy_marks
        outer = getattr(marks, "inside", False)
        marks.inside = True
        try:
            return policy(proposal, candidates)
        finally:
            marks.inside = outer

    def _refuse_inside_policy(self) -> None:
        """Raise RuntimeError when this thread is running a reconcile policy of this store.

        The store checks it before every connection checkout and every
        embedding call, so a store call from a policy fails before any I/O.

        Raises:
            RuntimeError: If called from inside a reconcile policy.
        """
        if getattr(self._policy_marks, "inside", False):
            raise RuntimeError(
                "A reconcile policy must not call the store: it runs inside the reconcile's "
                "transaction, while the store holds its connection. For a decision that "
                "needs more data, use find_conflicts, decide, then call resolve."
            )

    @staticmethod
    def _decision_keys(plan: ProposalPlan, decision: Decision) -> set[str]:
        """The keys whose rows a decision can mark with the plan's write id.

        The new memory's key, which carries the memory written or confirmed
        there, the version it overwrote and a late fact recorded straight
        into history, and every key the decision does more than keep. An
        attempt that committed wrote nowhere else, so a retry looks only at
        these (_applied_statements).
        """
        acted = (key for key, action in decision.actions.items() if action != Action.KEEP)
        return {plan.key, *acted}

    def _applied_statements(
        self, plan: ProposalPlan, keys: Iterable[str]
    ) -> list[tuple[str, list]]:
        """The reads of the rows under ``keys`` that carry the plan's write id.

        ``keys`` are the keys that earlier attempts' decisions named
        (_decision_keys). The items written or confirmed under them and, on
        a store with history, the history rows created under them are looked
        up by primary key, so a retry under contention reads those rows
        only, never the whole namespace. Without keys there is nothing to
        read: an attempt that reached no decision wrote nothing. A retried
        reconcile sends these with its first candidate read. Pass their rows
        to _applied_result.
        """
        named = sorted(set(keys))
        if not named:
            return []
        statements = [self.sql.applied_items(plan.prefix, named, write_id=plan.write_id)]
        if self._history is not None:
            statements.append(self.sql.applied_history(plan.prefix, named, write_id=plan.write_id))
        return statements

    def _applied_result(
        self, plan: ProposalPlan, found: Sequence[Sequence[Mapping[str, Any]]]
    ) -> ReconcileResult | None:
        """The result of an earlier attempt that committed, rebuilt from its rows, or None.

        ``found`` are the rows of _applied_statements, in order. None means
        no row carries the write id: no earlier attempt committed, or one
        committed without writing, and the attempt runs from the start.

        The result lists only what the rows prove. The memory under the new
        key carries the write id when this write wrote it or confirmed it.
        Confirming changes neither the version nor updated_at, so a
        confirmed_at later than its updated_at means it was confirmed;
        otherwise it was written, and the row has the version. Only a write
        archives the memory it replaces under the new key as overwritten,
        so that history row proves the write too. When another writer
        replaced or confirmed the new memory before the retry, that row is
        all that is left: the memory was written, and its version is None.
        Every other item that carries the write id was confirmed. The
        history rows give the keys superseded and deleted, and a row of the
        new key that ended as superseded is the new memory, recorded
        straight into history. Nothing records which candidates were kept,
        so ``kept`` is empty, and without history so are ``superseded`` and
        ``deleted``. Keys come in key order.
        """
        items = found[0]
        history = found[1] if len(found) > 1 else []
        if not items and not history:
            return None

        def ended_as(reason: str) -> bool:
            return any(row["key"] == plan.key and row["reason"] == reason for row in history)

        overwrote = ended_as("overwritten")
        target = next((row for row in items if row["key"] == plan.key), None)
        # A write never confirms the memory under its own key
        confirmed_target = (
            target is not None
            and not overwrote
            and target["confirmed_at"] is not None
            and target["confirmed_at"] > target["updated_at"]
        )
        written = target is not None and not confirmed_target

        def keys(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
            return tuple(sorted({row["key"] for row in rows}))

        new: Literal["written", "skipped", "superseded"] = "skipped"
        if written or overwrote:
            new = "written"
        elif ended_as("superseded"):
            new = "superseded"
        return ReconcileResult(
            key=plan.key,
            new=new,
            version=int(target["version"]) if written and target is not None else None,
            superseded=keys(
                row for row in history if row["reason"] == "superseded" and row["key"] != plan.key
            ),
            deleted=keys(row for row in history if row["reason"] == "deleted"),
            confirmed=keys(row for row in items if not (written and row["key"] == plan.key)),
            kept=(),
            already_applied=True,
        )

    def _decision_statements(
        self,
        plan: ProposalPlan,
        candidates: Sequence[Candidate],
        decision: Decision,
        *,
        vector_rows: Sequence[VectorRow],
        valid_at: datetime,
    ) -> list[tuple[str, list]]:
        """The statements that apply a decision about a new memory, in the order they run.

        ``decision`` is valid for ``candidates`` (_checked_decision).
        ``vector_rows`` are the new memory's vectors, embedded once, and
        ``valid_at`` is its valid time as the policy saw it.

        On a store with history, every version the decision ends is archived
        first. The memory under the new key, when the new memory overwrites
        it, and the candidates it supersedes end at the valid time the caller
        gave, or now without one; deleted candidates end now. A new memory
        that a newer fact supersedes goes straight into history, valid until
        that fact became valid. Then a new memory to write is written as put
        writes it, with the write id, its provenance and the caller's valid
        time, and its vectors replace the old ones. The superseded and
        deleted candidates go with their vectors, and the confirmed ones are
        confirmed last. Without history nothing is archived: superseding and
        deleting remove the memory, and a late fact is kept nowhere.

        Only the item upsert of a written memory returns rows, with the
        version it wrote. Pass them to _reconcile_result.
        """
        write_id = plan.write_id
        superseded = _acted(candidates, decision, Action.SUPERSEDE)
        deleted = _acted(candidates, decision, Action.DELETE)
        writes = decision.new == "write"
        statements: list[tuple[str, list]] = []
        if self._history is not None:
            ttl_days = self._history.ttl_days
            if decision.new == "superseded":
                newer = next(c for c in candidates if c.key == decision.superseded_by)
                statements.append(
                    self.sql.history_insert_superseded(
                        plan.prefix,
                        plan.key,
                        value_json=plan.value_json,
                        provenance_json=plan.provenance_json,
                        valid_at=valid_at,
                        invalid_at=newer.valid_at,
                        superseded_by=newer.key,
                        write_id=write_id,
                        ttl_days=ttl_days,
                    )
                )
            # (reason, keys, superseded_by, invalid_at). The memory under the
            # new key is a candidate exactly when it is live
            ended: list[tuple[str, list[str], str | None, datetime | None]] = []
            if writes and any(c.key == plan.key for c in candidates):
                ended.append(("overwritten", [plan.key], None, plan.valid_at))
            if superseded:
                ended.append(("superseded", superseded, plan.key, plan.valid_at))
            if deleted:
                ended.append(("deleted", deleted, None, None))
            statements.extend(
                self.sql.archive(
                    plan.prefix,
                    keys,
                    reason=reason,
                    superseded_by=superseded_by,
                    invalid_at=invalid_at,
                    write_id=write_id,
                    ttl_days=ttl_days,
                )
                for reason, keys, superseded_by, invalid_at in ended
            )
        if writes:
            # As put writes it: the old vectors go, then the item and the new vectors
            statements.extend(self._reindex_deletes(plan.put))
            row = (plan.prefix, plan.key, plan.value_json, plan.ttl, plan.provenance_json)
            statements.append(
                self.sql.upsert_items(
                    [(*row, plan.valid_at)],
                    write_id=write_id,
                    history=self._history is not None,
                    return_version=True,
                )
            )
            statements.extend(self._vector_upserts(vector_rows))
        removed = superseded + deleted
        if removed:
            statements.extend(self._delete_statements(plan.prefix, removed))
        statements.extend(
            self.sql.confirm(plan.prefix, key, write_id=write_id)
            for key in _acted(candidates, decision, Action.CONFIRM)
        )
        return statements

    def _reconcile_result(
        self,
        plan: ProposalPlan,
        candidates: Sequence[Candidate],
        decision: Decision,
        returned: Sequence[Sequence[Mapping[str, Any]]],
    ) -> ReconcileResult:
        """What applying a decision did, as reconcile returns it.

        ``returned`` are the rows of the statements of _decision_statements
        that return rows, in order: only the item upsert of a written memory
        does, with the version it wrote. Keys come in the order of the
        candidates. The memory under the new key is not kept when the new
        memory overwrote it. On a store without history a new memory that a
        newer fact supersedes is recorded nowhere, and the result says it
        was skipped.

        Raises:
            RuntimeError: If a written memory's upsert returned no version.
                The attempt carried a fresh write id, so the upsert always
                writes its row; this can only be a broken invariant, and the
                write has committed.
        """
        writes = decision.new == "write"
        new: Literal["written", "skipped", "superseded"] = "skipped"
        version = None
        if writes:
            if not returned or not returned[0]:
                raise RuntimeError(
                    f"The upsert of {plan.prefix}/{plan.key} committed but returned no version"
                )
            new, version = "written", int(returned[0][0]["version"])
        elif decision.new == "superseded" and self._history is not None:
            new = "superseded"
        kept = _acted(candidates, decision, Action.KEEP)
        return ReconcileResult(
            key=plan.key,
            new=new,
            version=version,
            superseded=tuple(_acted(candidates, decision, Action.SUPERSEDE)),
            deleted=tuple(_acted(candidates, decision, Action.DELETE)),
            confirmed=tuple(_acted(candidates, decision, Action.CONFIRM)),
            kept=tuple(key for key in kept if not (writes and key == plan.key)),
        )

    def _plan_resolve(self, conflicts: Conflicts, decision: Decision) -> ResolvePlan:
        """Check a decision about a Conflicts and build what resolve() sends, before any I/O.

        The Conflicts must be one that find_conflicts returned, as it
        returned it or as a LangGraph checkpoint gives it back, from a store
        with this store's items table and index, since resolve writes the
        plan find_conflicts made. The decision must be valid for its
        proposal and candidates.

        The check before the write covers the new memory's key, whose memory
        the write replaces, and every candidate the decision supersedes,
        deletes or confirms, or names as superseding the new memory. A
        candidate it keeps is left out, so a write to it in between changes
        nothing. The rows that carry the write id are looked for under the
        new key and every candidate's key, not only under the keys this
        decision names: an earlier resolve of the same Conflicts may have
        decided otherwise.

        Raises:
            ValueError: If ``conflicts`` is not what find_conflicts
                returned, as when it was built by hand or changed since, or
                a store with another items table or index made it; or if
                ``decision`` is not a valid Decision for its proposal and
                candidates.
        """
        state = conflicts._state if isinstance(conflicts, Conflicts) else None
        if not isinstance(state, ConflictsState):
            raise ValueError(
                "resolve needs the Conflicts that find_conflicts returned. This one holds "
                "no plan for the write, as when it is built by hand, or read back by a "
                "checkpoint serializer that does not allow the store's types."
            )
        identity = self._identity()
        if state.store != identity:
            raise ValueError(
                f"This Conflicts was found by another store, {state.store}, while this store "
                f"is {identity}. Resolve it on a store with the same table and index."
            )
        tokens = {c.key: (c.version, c.item.updated_at) for c in conflicts.candidates}
        if conflicts.proposal.key != state.plan.key or tokens != state.tokens:
            raise ValueError(
                "This Conflicts was changed after find_conflicts returned it: its proposal "
                "or candidates are not the ones find_conflicts found."
            )
        if not isinstance(decision, Decision):
            raise ValueError(f"resolve needs a Decision, got {type(decision).__name__}")
        validate_decision(conflicts.proposal, conflicts.candidates, decision)

        plan, candidates = state.plan, conflicts.candidates
        checked = self._decision_keys(plan, decision)
        if decision.superseded_by is not None:
            checked.add(decision.superseded_by)
        expected: dict[str, tuple[int, datetime] | None] = {plan.key: state.target}
        expected.update((c.key, state.tokens[c.key]) for c in candidates if c.key in checked)
        removing = _acted(candidates, decision, Action.DELETE)
        removing += _acted(candidates, decision, Action.SUPERSEDE)
        reads = [
            self.sql.tokens(plan.prefix, sorted(expected)),
            *self._applied_statements(plan, [plan.key, *(c.key for c in candidates)]),
        ]
        writes = self._decision_statements(
            plan,
            candidates,
            decision,
            vector_rows=state.vector_rows,
            valid_at=conflicts.proposal.valid_at,
        )
        return ResolvePlan(
            plan=plan,
            candidates=candidates,
            decision=decision,
            expected=expected,
            removed=frozenset(removing) - {plan.key},
            reads=reads,
            writes=writes,
        )

    def _check_resolve(
        self,
        resolving: ResolvePlan,
        found: Sequence[Sequence[Mapping[str, Any]]],
        *,
        retry: bool,
    ) -> ReconcileResult | None:
        """What the first round trip of a resolve attempt found, before anything is written.

        ``found`` are the rows of ``resolving.reads``, in order. Rows that
        carry the write id mean that an earlier attempt, or an earlier
        resolve of the same Conflicts, committed: its result comes back,
        rebuilt from them as reconcile's retry rebuilds it (_applied_result).
        That comes first, because such a write changed the tokens itself.
        Otherwise None means that nothing the decision depends on changed,
        and the attempt applies it.

        ``retry`` says whether an earlier attempt of the same call ran. Its
        commit may have been lost, and on a store without history a decision
        that only deletes or supersedes leaves no row with the write id.
        So on a retry the candidates in ``resolving.removed`` may be gone:
        the state the decision asked for. On the first attempt they must
        still be live, as every key the check covers must.

        Raises:
            ConflictsChanged: If a key of ``resolving.expected`` changed
                (changed_keys). Raised in the attempt's transaction, it rolls
                the transaction back. It is not a database error, so the
                retry policy does not retry it.
        """
        applied = self._applied_result(resolving.plan, found[1:])
        if applied is not None:
            return applied
        gone = resolving.removed if retry else frozenset()
        changed = changed_keys(resolving.expected, found[0], may_be_gone=gone)
        if changed:
            raise ConflictsChanged(changed)
        return None
