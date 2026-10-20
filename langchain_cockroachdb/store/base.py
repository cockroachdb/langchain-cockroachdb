"""Code shared by the sync and async CockroachDB stores.

Everything here runs without a database: grouping and validating ops,
planning the rows a put batch writes, building search statements and
turning result rows into items. The store classes run the statements.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
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
from langchain_cockroachdb.store.sql import ItemRow, StoreSQL

logger = logging.getLogger(__name__)

# Rows in one multi-row item upsert. Each row takes 4 bound parameters, so a
# statement stays far below the limit of 65,535 parameters.
ITEMS_PER_STATEMENT = 500


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
        embed_requests: (prefix, key, field_name, text) for every text to embed.
        reindexed: Namespace text to the put keys whose vectors are deleted
            before new ones are written. Every put item when the store has an index.
    """

    deletes: dict[str, list[str]] = field(default_factory=dict)
    items: list[ItemRow] = field(default_factory=list)
    embed_requests: list[tuple[str, str, str, str]] = field(default_factory=list)
    reindexed: dict[str, list[str]] = field(default_factory=dict)


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
                    plan.embed_requests.extend(
                        (prefix, key, f"{path}.{i}", text) for i, text in enumerate(texts)
                    )
                elif texts:
                    plan.embed_requests.append((prefix, key, path, texts[0]))
        return plan

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
        self, plan: PutPlan, vector_rows: Sequence[Any] = ()
    ) -> list[tuple[str, list]]:
        """The statements of a put batch, in the order they must run.

        Old vectors of re-indexed items go first, then the item upserts, the
        vector upserts, and the deletes with their vectors.
        """
        statements: list[tuple[str, list]] = []
        for prefix, keys in plan.reindexed.items():
            statements.append(self.sql.delete_vectors(prefix, keys))
        for start in range(0, len(plan.items), ITEMS_PER_STATEMENT):
            statements.append(
                self.sql.upsert_items(plan.items[start : start + ITEMS_PER_STATEMENT])
            )
        for start in range(0, len(vector_rows), ITEMS_PER_STATEMENT):
            statements.append(
                self.sql.upsert_vectors(vector_rows[start : start + ITEMS_PER_STATEMENT])
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
