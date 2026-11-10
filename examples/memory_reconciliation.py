"""Keeping an agent's memories true in CockroachDB.

Run it against a local cluster:

    COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable" \\
        python examples/memory_reconciliation.py

Semantic search finds memories that are similar, not memories that are still
true. This example keeps them true with the store's reconciliation methods:
a newer fact in the same slot supersedes an older one, a repeated fact
confirms the one already stored, and a fact that arrives late goes straight
into history. In the two-step flow a judge decides while no transaction is
open, and resolve applies the decision only if nothing changed in between.
History keeps every version with its provenance, confirm renews a memory's
expiry, and put_if keeps two writers from losing an update.

It needs no API key or network. KeywordEmbeddings, below, stands in for an
embedding model, and the judge is a scripted function that stands in for an
LLM or a person.

The store uses its own tables (memory_reconciliation and its _vectors,
_history and _migrations tables) and drops them at the end, even after a
failure.
"""

import os
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

import psycopg
from langchain_core.embeddings import Embeddings
from psycopg import sql

from langchain_cockroachdb import CockroachDBIndexConfig, CockroachDBStore, CockroachDBTTLConfig
from langchain_cockroachdb.store import (
    Action,
    ConditionalWrite,
    Conflicts,
    ConflictsChanged,
    Decision,
    HistoryConfig,
    MemoryRecord,
    ReconcileResult,
    first_of,
    skip_duplicates,
    supersede_same_slot,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)
# Raw psycopg needs a postgresql:// URL; cockroachdb:// is the same server
LIBPQ_URL = "postgresql://" + CONNECTION_STRING.split("://", 1)[1]
TABLE = "memory_reconciliation"
MEMORIES = ("user", "alice", "memories")
DAY = 60 * 24  # minutes


class KeywordEmbeddings(Embeddings):
    """Stands in for an embedding model: one dimension for each keyword a text mentions.

    Texts that mention the same keywords point the same way, so they score
    high against each other, as a real model scores texts about the same
    thing. Every vector also has a small constant part, so none is all zeros.
    """

    WEIGHTS = {
        "seat": 1.0,
        "window": 1.0,
        "aisle": 1.0,
        "peanut": 1.0,
        "no longer": 0.5,
        "please": 0.2,
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [weight if word in lowered else 0.0 for word, weight in self.WEIGHTS.items()] + [0.1]


# Embed each memory's "text" field. Use a real model, such as OpenAIEmbeddings,
# with its own dims, for scores that follow meaning.
INDEX: CockroachDBIndexConfig = {
    "dims": len(KeywordEmbeddings.WEIGHTS) + 1,
    "embed": KeywordEmbeddings(),
    "fields": ["text"],
}
# Memories expire 30 days after they were written or last confirmed. Reads do
# not push back the expiry: retrieval is not confirmation
TTL: CockroachDBTTLConfig = {"default_ttl": 30 * DAY, "refresh_on_read": False}
# Keep old versions for a year
HISTORY: HistoryConfig = {"ttl_days": 365}

# Skip a near duplicate and confirm the memory it repeats; otherwise let a
# newer fact in the same slot supersede the older ones
POLICY = first_of(skip_duplicates(), supersede_same_slot())


def outcome(result: ReconcileResult) -> str:
    """What reconcile or resolve did, in one line."""
    line = result.new if result.version is None else f"{result.new}, version {result.version}"
    for name in ("superseded", "deleted", "confirmed"):
        keys = getattr(result, name)
        if keys:
            line += f", {name} {list(keys)}"
    return line


def in_seat_slot(text: str) -> dict[str, Any]:
    """A memory about where Alice likes to sit. The "slot" field names the fact it states."""
    return {"slot": "seat", "text": text}


def slot_supersede(store: CockroachDBStore) -> None:
    """1. A newer fact in the same slot supersedes the older one."""
    print("1. A newer fact in the same slot supersedes the older one")
    result = store.reconcile(
        MEMORIES,
        "seat-window",
        in_seat_slot("I prefer window seats."),
        slot="slot",
        policy=POLICY,
        provenance={"source": "chat", "thread": "thread-1"},
        # Alice said it in January. Without valid_at, a fact is true from when it is written
        valid_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )
    print(f"   reconcile seat-window: {outcome(result)}")
    result = store.reconcile(
        MEMORIES,
        "seat-aisle",
        in_seat_slot("I prefer aisle seats now."),
        slot="slot",
        policy=POLICY,
        provenance={"source": "chat", "thread": "thread-2"},
    )
    print(f"   reconcile seat-aisle: {outcome(result)}")
    record = store.record(MEMORIES, "seat-aisle")
    assert record is not None
    print(f"   record seat-aisle: version {record.version}, provenance {record.provenance}")


def duplicate(store: CockroachDBStore) -> None:
    """2. A near duplicate is skipped, and confirms the memory it repeats."""
    print("2. A near duplicate is skipped, and confirms the memory it repeats")
    result = store.reconcile(
        MEMORIES,
        "seat-aisle-again",
        in_seat_slot("Aisle seats, please."),
        slot="slot",
        policy=POLICY,
        provenance={"source": "chat", "thread": "thread-3"},
    )
    print(f"   reconcile seat-aisle-again: {outcome(result)}")
    record = store.record(MEMORIES, "seat-aisle")
    assert record is not None
    print(f"   seat-aisle is confirmed {record.confirmations} time")


def late_fact(store: CockroachDBStore) -> None:
    """3. A fact that arrives late goes straight into history."""
    print("3. A fact that arrives late goes straight into history")
    # An agent reads an old conversation, where Alice said she prefers window seats
    result = store.reconcile(
        MEMORIES,
        "seat-window-2025",
        in_seat_slot("I prefer window seats."),
        slot="slot",
        policy=POLICY,
        provenance={"source": "chat", "thread": "thread-2025-03"},
        valid_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
    )
    print(f"   reconcile seat-window-2025, true as of 2025-03-01: {outcome(result)}")
    live = [item.key for item in store.search(MEMORIES, filter={"slot": "seat"})]
    print(f"   live memories in the seat slot: {live}")


def show_history(store: CockroachDBStore) -> None:
    """4. History keeps each old version, why it ended and where it came from."""
    print("4. History keeps each old version, why it ended and where it came from")
    for key in ("seat-window", "seat-window-2025"):
        for version in store.history(MEMORIES, key):
            assert version.invalid_at is not None
            print(f"   {key}, version {version.version}: {version.value['text']!r}")
            print(
                f"      true from {version.valid_at:%Y-%m-%d} to {version.invalid_at:%Y-%m-%d}, "
                f"{version.reason} by {version.superseded_by}, provenance {version.provenance}"
            )


def judge(conflicts: Conflicts) -> Decision:
    """Stands in for an LLM or a person deciding about the conflicts.

    A memory that says something is no longer true supersedes the similar
    memories it contradicts. Anything else is written next to them.
    """
    if "no longer" in conflicts.proposal.value["text"].lower():
        similar = [c.key for c in conflicts.candidates if c.reason == "similar"]
        return Decision(actions={key: Action.SUPERSEDE for key in similar})
    return Decision()


def find(store: CockroachDBStore, key: str, value: dict[str, Any]) -> Conflicts:
    """find_conflicts for a memory from the chat, printing its candidates."""
    conflicts = store.find_conflicts(
        MEMORIES, key, value, provenance={"source": "chat", "thread": "thread-4"}
    )
    for c in conflicts.candidates:
        score = "no score" if c.score is None else f"score {c.score:.2f}"
        print(f"   find_conflicts {key}: {c.key} ({c.reason}, {score}): {c.item.value['text']!r}")
    return conflicts


def two_steps(store: CockroachDBStore) -> None:
    """5. find_conflicts, a judge decides, and resolve applies the decision."""
    print("5. Two steps: find_conflicts, a judge decides, resolve applies the decision")
    # A plain put, as a tool that knows nothing of reconciliation makes
    store.put(MEMORIES, "allergy", {"text": "Allergic to peanuts."})
    new = {"text": "No longer allergic to peanuts."}
    conflicts = find(store, "no-peanut-allergy", new)
    decision = judge(conflicts)
    print(f"   judge: {[f'{action.value} {key}' for key, action in decision.actions.items()]}")
    # No transaction is open while the judge decides, so another writer can step in
    store.put(MEMORIES, "allergy", {"text": "Allergic to peanuts and cashews."})
    print("   meanwhile, another writer puts allergy again")
    try:
        store.resolve(conflicts, decision)
    except ConflictsChanged as changed:
        print(f"   resolve: ConflictsChanged for {list(changed.keys)}, so find them again")
    conflicts = find(store, "no-peanut-allergy", new)
    result = store.resolve(conflicts, judge(conflicts))
    print(f"   resolve: {outcome(result)}")


def renewal(record: MemoryRecord) -> float:
    """Minutes a confirmed memory stays: 30 days, doubled for each earlier confirmation.

    At most a year. ``record`` is the memory as confirm read it, before this
    confirmation counts.
    """
    return min(30 * DAY * 2.0**record.confirmations, 365 * DAY)


def expires_in_days(key: str) -> int:
    """Days until a memory expires. Items do not carry their expiry, so this reads the table."""
    query = sql.SQL("SELECT expires_at - now() AS left FROM {} WHERE prefix = %s AND key = %s")
    with psycopg.connect(LIBPQ_URL, autocommit=True) as conn:
        row = conn.execute(
            query.format(sql.Identifier(TABLE)), (".".join(MEMORIES), key)
        ).fetchone()
    assert row is not None
    days: float = row[0].total_seconds() / (24 * 3600)
    return round(days)


def confirmations(store: CockroachDBStore) -> None:
    """6. confirm renews a memory's expiry, for longer each time it is confirmed."""
    print("6. confirm renews a memory's expiry, for longer each time it is confirmed")
    record = store.record(MEMORIES, "seat-aisle")
    assert record is not None
    print(
        f"   seat-aisle: confirmed {record.confirmations} time, "
        f"expires in {expires_in_days('seat-aisle')} days"
    )
    for _ in range(2):
        confirmed = store.confirm(MEMORIES, "seat-aisle", ttl=renewal)
        record = store.record(MEMORIES, "seat-aisle")
        assert record is not None
        print(
            f"   confirm: {confirmed}, confirmed {record.confirmations} times, "
            f"expires in {expires_in_days('seat-aisle')} days"
        )
    print(f"   confirm a memory that does not exist: {store.confirm(MEMORIES, 'no-such-memory')}")


def airlines(names: Iterable[str]) -> dict[str, Any]:
    """A memory of the airlines Alice flies with."""
    names = sorted(set(names))
    return {"text": "Flies with " + ", ".join(names), "airlines": names}


def add_airline(
    store: CockroachDBStore, record: MemoryRecord | None, name: str, writer: str
) -> ConditionalWrite:
    """Add an airline to the memory read as ``record``, merging if another writer changed it."""
    while True:
        names = record.item.value["airlines"] if record else []
        result = store.put_if(MEMORIES, "airlines", airlines([*names, name]), expected=record)
        if result.applied:
            print(f"   {writer} adds {name}: applied, version {result.version}")
            return result
        # Another writer got there first: merge with the current memory, and expect it
        record = result.current
        current = "none" if record is None else f"version {record.version}"
        print(f"   {writer} adds {name}: conflict, the current memory is {current}")


def two_writers(store: CockroachDBStore) -> None:
    """7. Two writers update one memory with record() and put_if()."""
    print("7. Two writers update one memory with record() and put_if()")
    # expected=None writes only if no live memory has the key, so it creates once
    created = store.put_if(MEMORIES, "airlines", airlines(["TAP"]), expected=None)
    assert created.applied
    print(f"   put_if airlines, expected=None: applied, version {created.version}")
    again = store.put_if(MEMORIES, "airlines", airlines(["TAP"]), expected=None)
    assert not again.applied and again.current is not None
    print(f"   again: conflict, the current memory is version {again.current.version}")
    first = store.record(MEMORIES, "airlines")
    second = store.record(MEMORIES, "airlines")
    assert first is not None and second is not None
    print(f"   both writers read version {first.version}")
    add_airline(store, first, "Iberia", "writer 1")
    add_airline(store, second, "KLM", "writer 2")
    final = store.record(MEMORIES, "airlines")
    assert final is not None
    print(f"   airlines, version {final.version}: {final.item.value['airlines']}")


def forget(store: CockroachDBStore) -> None:
    """8. delete keeps history; purge removes a memory with its history."""
    print("8. Forgetting: delete keeps history, purge removes the memory and its history")
    store.delete(MEMORIES, "airlines")
    versions = [(v.version, v.reason) for v in store.history(MEMORIES, "airlines")]
    print(f"   delete airlines: get gives {store.get(MEMORIES, 'airlines')}, history {versions}")
    for key in ("airlines", "seat-window"):
        removed = store.purge(MEMORIES, key)
        print(f"   purge {key}: {removed}, history now {store.history(MEMORIES, key)}")


STEPS: list[Callable[[CockroachDBStore], None]] = [
    slot_supersede,
    duplicate,
    late_fact,
    show_history,
    two_steps,
    confirmations,
    two_writers,
    forget,
]


def drop_tables() -> None:
    """Drop the store's tables, so the next run starts clean."""
    with psycopg.connect(LIBPQ_URL, autocommit=True) as conn:
        for suffix in ("_vectors", "_history", "", "_migrations"):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(f"{TABLE}{suffix}"))
            )


def main() -> None:
    """Run every step, then drop the tables, even after a failure."""
    print("Keeping memories true with CockroachDB\n")
    print("=" * 50 + "\n")

    try:
        with CockroachDBStore.from_conn_string(
            CONNECTION_STRING, index=INDEX, ttl=TTL, history=HISTORY, table_name=TABLE
        ) as store:
            store.setup()
            print("setup() created the store tables, with history, a vector index and TTL\n")
            for step in STEPS:
                step(store)
                print()
    finally:
        drop_tables()
        print(f"Dropped {TABLE} and its _vectors, _history and _migrations tables\n")

    print("=" * 50)
    print("\nDone")


if __name__ == "__main__":
    main()
