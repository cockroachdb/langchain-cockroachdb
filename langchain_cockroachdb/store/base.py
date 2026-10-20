"""Code shared by the sync and async CockroachDB stores.

Everything here runs without a database: grouping and validating ops,
planning the rows a put batch writes, building search statements and
turning result rows into items. The store classes run the statements.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.embed import get_text_at_path, tokenize_path

from langchain_cockroachdb.store.config import (
    ResolvedIndex,
    ResolvedTTL,
    resolve_ttl_config,
)
from langchain_cockroachdb.store.namespaces import (
    from_text,
    match_pattern,
    to_text,
    validate_labels,
)
from langchain_cockroachdb.store.sql import ItemRow, StoreSQL, VectorRow

logger = logging.getLogger(__name__)

# Rows in one multi-row item upsert. Each row takes 4 bound parameters, so a
# statement stays far below the limit of 65,535 parameters.
ITEMS_PER_STATEMENT = 500

# Rows in one multi-row vector upsert. VECTOR rows are large, so batches stay
# small, like the vector store's default batch_size.
VECTORS_PER_STATEMENT = 100


def _check_paging(limit: int, offset: int) -> None:
    """Raise ValueError for negative paging, before any database work."""
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")


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
    """

    deletes: dict[str, list[str]] = field(default_factory=dict)
    items: list[ItemRow] = field(default_factory=list)
    embed_requests: list[tuple[str, str, str, str]] = field(default_factory=list)
    reindexed: dict[str, list[str]] = field(default_factory=dict)


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
    listings: list[tuple[int, ListNamespacesOp]] = field(default_factory=list)
    listing_statements: list[tuple[str, list]] = field(default_factory=list)
    puts: PutPlan | None = None

    def reads(self) -> list[tuple[str, list[tuple[str, list]], Callable[[list[Any]], None]]]:
        """The read groups to run, in order: gets, searches, then listings.

        Each group is (label, statements, fill). Run the statements as one
        round trip and pass ``fill`` the rows of each statement, in order;
        it puts them into ``results``. Empty groups are left out.
        """
        groups = [
            ("store get", self.get_statements, self._fill_gets),
            ("store search", self.search_statements, self._fill_searches),
            ("store list namespaces", self.listing_statements, self._fill_listings),
        ]
        return [group for group in groups if group[1]]

    def _fill_gets(self, found: list[Any]) -> None:
        for ((namespace, _), items), rows in zip(self.get_groups, found, strict=True):
            by_key = {row["key"]: row for row in rows}
            for i, op in items:
                row = by_key.get(op.key)
                self.results[i] = None if row is None else _row_to_item(namespace, row)

    def _fill_searches(self, found: list[Any]) -> None:
        for (i, _), rows in zip(self.searches, found, strict=True):
            self.results[i] = [_row_to_search_item(row) for row in rows]

    def _fill_listings(self, found: list[Any]) -> None:
        for (i, _), rows in zip(self.listings, found, strict=True):
            self.results[i] = [from_text(row["ns"]) for row in rows]


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


class BaseCockroachDBStore:
    """State and planning shared by CockroachDBStore and its async twin."""

    # Names the store in log and error messages of the psycopg machinery
    component: ClassVar[str] = "store"
    _log: ClassVar[logging.Logger] = logger

    sql: StoreSQL
    index: ResolvedIndex | None
    ttl: ResolvedTTL
    _warned_query_without_index: bool
    # Pooled batches run concurrently; the lock makes the warning appear once
    _warning_lock: threading.Lock

    def _configure(
        self,
        *,
        index: ResolvedIndex | None,
        ttl: Any,
        table_name: str,
        schema: str,
    ) -> None:
        """Set the shared state from validated settings."""
        self.index = index
        self.ttl = resolve_ttl_config(ttl)
        self.sql = StoreSQL(schema, table_name, index=index)
        self._warned_query_without_index = False
        self._warning_lock = threading.Lock()

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
                groups[(tuple(op.namespace), bool(op.refresh_ttl))].append((i, op))
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
            plan.search_statements, plan.search_to_embed = self._search_plan(plan.searches)
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
            if len(embedding) != self.index.dims:
                raise ValueError(
                    f"The embedding model returned a vector of {len(embedding)} dimensions, "
                    f"but the index has {self.index.dims}"
                )
            if not all(math.isfinite(value) for value in embedding):
                raise ValueError(
                    f"The embedding model returned a vector with NaN or infinite values "
                    f"for {prefix}/{key}, field {field_name}"
                )
            rows.append((prefix, key, field_name, list(embedding), ttls[(prefix, key)]))
        return rows

    def _search_plan(
        self, search_ops: Sequence[tuple[int, SearchOp]]
    ) -> tuple[list[tuple[str, list]], list[tuple[int, str]]]:
        """Build one statement per search op.

        Returns:
            The statements, in the order of ``search_ops``, and the
            (statement position, query text) pairs whose QUERY_VECTOR
            placeholders need the query's embedding.
        """
        statements: list[tuple[str, list]] = []
        to_embed: list[tuple[int, str]] = []
        omit_expired = self.ttl.omit_expired
        for position, (_, op) in enumerate(search_ops):
            if op.query and self.index is not None:
                over_fetch = op.limit * self.index.estimated_vectors * 2 + 1
                statements.append(
                    self.sql.semantic_search(
                        namespace_prefix=op.namespace_prefix,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        omit_expired=omit_expired,
                        over_fetch=over_fetch,
                        refresh=op.refresh_ttl,
                    )
                )
                to_embed.append((position, op.query))
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
                    refresh=op.refresh_ttl,
                )
            )
        return statements, to_embed

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
        """
        statements: list[tuple[str, list]] = []
        for prefix, keys in plan.reindexed.items():
            statements.append(self.sql.delete_vectors(prefix, keys))
        for start in range(0, len(plan.items), ITEMS_PER_STATEMENT):
            statements.append(
                self.sql.upsert_items(plan.items[start : start + ITEMS_PER_STATEMENT])
            )
        for start in range(0, len(vector_rows), VECTORS_PER_STATEMENT):
            statements.append(
                self.sql.upsert_vectors(vector_rows[start : start + VECTORS_PER_STATEMENT])
            )
        for prefix, keys in plan.deletes.items():
            statements.append(self.sql.delete_items(prefix, keys))
            if self.index is not None:
                statements.append(self.sql.delete_vectors(prefix, keys))
        return statements

    def _listing_statement(self, op: ListNamespacesOp) -> tuple[str, list]:
        """The statement for one list_namespaces op."""
        return self.sql.list_namespaces(
            match_conditions=[
                match_pattern(condition.path, condition.match_type)
                for condition in op.match_conditions or ()
            ],
            max_depth=op.max_depth,
            limit=op.limit,
            offset=op.offset,
            omit_expired=self.ttl.omit_expired,
        )
