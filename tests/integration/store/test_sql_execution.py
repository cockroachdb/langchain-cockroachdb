"""The store's single-statement reads and deletes run on CockroachDB.

These run StoreSQL statements directly, with a vectors table, because the
store itself does not accept an index yet.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb._sql import vector_literal
from langchain_cockroachdb.store.config import resolve_index_config
from langchain_cockroachdb.store.sql import QUERY_VECTOR, StoreSQL


def make_sql(table_name: str) -> StoreSQL:
    index = resolve_index_config(
        {"dims": 3, "embed": DeterministicFakeEmbedding(size=3), "flat": True}
    )
    return StoreSQL("public", table_name, index=index)


def prepare(admin: Any, sql: StoreSQL) -> None:
    for statement in sql.migrations() + sql.vector_statements():
        admin.execute(statement)
    admin.execute(
        *sql.upsert_items(
            [
                ("a", "k1", '{"v": 1}', 1.0),
                ("a", "k2", '{"v": 2}', None),
                ("a.b", "k3", '{"v": 3}', 1.0),
                ("ab", "k4", '{"v": 4}', 1.0),
            ]
        )
    )
    admin.execute(
        *sql.upsert_vectors(
            [
                ("a", "k1", "$", [1.0, 0.0, 0.0], 1.0),
                ("a", "k2", "$", [0.0, 1.0, 0.0], None),
                ("a.b", "k3", "$", [0.0, 0.0, 1.0], 1.0),
                ("ab", "k4", "$", [1.0, 0.0, 0.0], 1.0),
            ]
        )
    )


def expiries(admin: Any, table: str) -> dict[tuple[str, str], Any]:
    rows = admin.execute(f"SELECT prefix, key, expires_at FROM {table}").fetchall()
    return {(row["prefix"], row["key"]): row["expires_at"] for row in rows}


def test_get_with_refresh_reads_and_refreshes_in_one_statement(admin, table_name) -> None:
    sql = make_sql(table_name)
    prepare(admin, sql)
    items_before, vectors_before = expiries(admin, sql.items), expiries(admin, sql.vectors)
    time.sleep(0.05)
    rows = admin.execute(
        *sql.get("a", ["k1", "k2", "missing"], omit_expired=True, refresh=True)
    ).fetchall()
    assert sorted(row["key"] for row in rows) == ["k1", "k2"]
    items_after, vectors_after = expiries(admin, sql.items), expiries(admin, sql.vectors)
    for after, before in ((items_after, items_before), (vectors_after, vectors_before)):
        assert after[("a", "k1")] > before[("a", "k1")]
        assert after[("a", "k2")] is None
        assert after[("a.b", "k3")] == before[("a.b", "k3")]


def test_searches_with_refresh_update_only_returned_rows(admin, table_name) -> None:
    sql = make_sql(table_name)
    prepare(admin, sql)
    items_before, vectors_before = expiries(admin, sql.items), expiries(admin, sql.vectors)
    time.sleep(0.05)
    statement, params = sql.search(
        namespace_prefix=["a"],
        filter={"v": {"$gte": 1}},
        limit=10,
        offset=0,
        omit_expired=True,
        refresh=True,
    )
    rows = admin.execute(statement, params).fetchall()
    assert {(row["prefix"], row["key"]) for row in rows} == {
        ("a", "k1"),
        ("a", "k2"),
        ("a.b", "k3"),
    }
    items_after = expiries(admin, sql.items)
    assert items_after[("a", "k1")] > items_before[("a", "k1")]
    assert items_after[("a.b", "k3")] > items_before[("a.b", "k3")]
    assert items_after[("ab", "k4")] == items_before[("ab", "k4")]

    time.sleep(0.05)
    items_before = expiries(admin, sql.items)
    statement, params = sql.semantic_search(
        namespace_prefix=["a"],
        filter=None,
        limit=1,
        offset=0,
        omit_expired=True,
        refresh=True,
    )
    query = vector_literal([1.0, 0.0, 0.0])
    params = [query if p is QUERY_VECTOR else p for p in params]
    rows = admin.execute(statement, params).fetchall()
    assert [(row["prefix"], row["key"]) for row in rows] == [("a", "k1")]
    assert rows[0]["score"] > 0.99
    items_after, vectors_after = expiries(admin, sql.items), expiries(admin, sql.vectors)
    assert items_after[("a", "k1")] > items_before[("a", "k1")]
    assert vectors_after[("a", "k1")] > vectors_before[("a", "k1")]
    assert items_after[("a.b", "k3")] == items_before[("a.b", "k3")]
    assert vectors_after[("ab", "k4")] == vectors_before[("ab", "k4")]


def test_delete_namespace_removes_items_and_vectors_in_one_statement(admin, table_name) -> None:
    sql = make_sql(table_name)
    prepare(admin, sql)
    rows = admin.execute(*sql.delete_namespace(["a"])).fetchall()
    assert sorted(row["key"] for row in rows) == ["k1", "k2", "k3"]
    assert set(expiries(admin, sql.items)) == {("ab", "k4")}
    assert set(expiries(admin, sql.vectors)) == {("ab", "k4")}
