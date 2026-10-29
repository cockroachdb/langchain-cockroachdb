# Memory Reconciliation Example

`examples/memory_reconciliation.py` keeps an agent's long-term memory true. A newer fact supersedes an older one in the same slot, a repeated fact confirms the memory it repeats, and a fact that arrives late goes straight into history. A judge decides about a conflict in two steps, history keeps every old version with its provenance, `confirm` renews a memory's expiry, and `put_if` keeps two writers from losing an update. It ends with the forget path.

## Running the Example

```bash
# Install the package
pip install langchain-cockroachdb

# Start CockroachDB
docker-compose up -d

# Run the memory reconciliation example
python examples/memory_reconciliation.py
```

It needs no API key. `KeywordEmbeddings`, defined in the script, stands in for an embedding model: it gives each text one dimension for each keyword it mentions, so texts that share keywords score high against each other, as a real model scores texts about the same thing. "I prefer aisle seats now." and "Aisle seats, please." score 0.99, and the window and aisle preferences score 0.50. The judge in the two-step flow is a scripted function that stands in for an LLM or a person.

The store uses its own tables (`memory_reconciliation`, `memory_reconciliation_vectors`, `memory_reconciliation_history` and `memory_reconciliation_migrations`) and drops them at the end, even after a failure. The script gives its memories readable keys. An agent usually passes `None` as the key, and the store picks a uuid4.

## The Store

History is on, so every write that ends a version keeps it. Expiry follows confirmations, not reads:

```python
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

with CockroachDBStore.from_conn_string(
    CONNECTION_STRING, index=INDEX, ttl=TTL, history=HISTORY, table_name=TABLE
) as store:
    store.setup()
```

Each memory about where Alice sits has `"slot": "seat"`. The policy skips a near duplicate and confirms the memory it repeats. Otherwise it lets a newer fact in the same slot supersede the older ones:

```python
POLICY = first_of(skip_duplicates(), supersede_same_slot())
```

## A Newer Fact Supersedes the Older One

Alice said she prefers window seats in January, and the agent passes that as the memory's valid time. Later she prefers aisle seats. The second `reconcile` finds the first memory in the same slot, and the policy supersedes it, in the transaction that writes the new one:

```python
store.reconcile(
    MEMORIES,
    "seat-window",
    in_seat_slot("I prefer window seats."),
    slot="slot",
    policy=POLICY,
    provenance={"source": "chat", "thread": "thread-1"},
    valid_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
)
store.reconcile(
    MEMORIES,
    "seat-aisle",
    in_seat_slot("I prefer aisle seats now."),
    slot="slot",
    policy=POLICY,
    provenance={"source": "chat", "thread": "thread-2"},
)
```

Then `record` reads the aisle preference with its version and provenance.

## A Near Duplicate Confirms the Memory It Repeats

"Aisle seats, please." scores 0.99 against the aisle preference, so `skip_duplicates` writes nothing and confirms the memory it repeats. Repeating a fact reinforces it instead of piling up copies:

```python
result = store.reconcile(
    MEMORIES,
    "seat-aisle-again",
    in_seat_slot("Aisle seats, please."),
    slot="slot",
    policy=POLICY,
    provenance={"source": "chat", "thread": "thread-3"},
)
# result.new == "skipped", result.confirmed == ("seat-aisle",)
```

## A Late Fact Goes Straight Into History

The agent reads an old conversation from March 2025, where Alice said she prefers window seats. The aisle preference became true later, so `supersede_same_slot` records the old fact as superseded by it. With history on, the fact goes straight into history and never becomes live:

```python
result = store.reconcile(
    MEMORIES,
    "seat-window-2025",
    in_seat_slot("I prefer window seats."),
    slot="slot",
    policy=POLICY,
    provenance={"source": "chat", "thread": "thread-2025-03"},
    valid_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
)
# result.new == "superseded"
```

A search of the seat slot still finds only the aisle preference.

## History

`history` returns the old versions of a memory, newest first. Each one says when its fact was true, why it ended, what replaced it and where it came from:

```python
for key in ("seat-window", "seat-window-2025"):
    for version in store.history(MEMORIES, key):
        print(key, version.version, version.value["text"])
        print(version.valid_at, version.invalid_at, version.reason, version.superseded_by)
        print(version.provenance)
```

## Two Steps With a Judge

A tool that knows nothing of reconciliation saved "Allergic to peanuts." with a plain `put`. Now Alice says she is no longer allergic. `find_conflicts` finds the old memory as a similar candidate (score 0.90), and the judge decides to supersede it. No transaction is open while the judge decides, so the script lets another writer change the old memory in between. `resolve` notices, raises `ConflictsChanged` and writes nothing. The script finds the conflicts again, asks the judge again, and `resolve` applies the decision:

```python
def judge(conflicts: Conflicts) -> Decision:
    """Stands in for an LLM or a person deciding about the conflicts.

    A memory that says something is no longer true supersedes the similar
    memories it contradicts. Anything else is written next to them.
    """
    if "no longer" in conflicts.proposal.value["text"].lower():
        similar = [c.key for c in conflicts.candidates if c.reason == "similar"]
        return Decision(actions={key: Action.SUPERSEDE for key in similar})
    return Decision()


store.put(MEMORIES, "allergy", {"text": "Allergic to peanuts."})
new = {"text": "No longer allergic to peanuts."}
conflicts = find(store, "no-peanut-allergy", new)
decision = judge(conflicts)
store.put(MEMORIES, "allergy", {"text": "Allergic to peanuts and cashews."})  # another writer
try:
    store.resolve(conflicts, decision)
except ConflictsChanged as changed:
    print(changed.keys)  # ('allergy',)
conflicts = find(store, "no-peanut-allergy", new)
result = store.resolve(conflicts, judge(conflicts))
```

`find` is a helper in the script that calls `find_conflicts` and prints the candidates. With an LLM in place of `judge`, the code is the same, as the [guide's example](../guides/long-term-memory.md#the-two-step-flow) shows.

## Confirmation-Driven Expiry

`confirm` renews a memory's expiry and counts the confirmation. Its `ttl` here is a function of the memory, so each confirmation keeps the memory longer:

```python
def renewal(record: MemoryRecord) -> float:
    """Minutes a confirmed memory stays: 30 days, doubled for each earlier confirmation.

    At most a year. ``record`` is the memory as confirm read it, before this
    confirmation counts.
    """
    return min(30 * DAY * 2.0**record.confirmations, 365 * DAY)


store.confirm(MEMORIES, "seat-aisle", ttl=renewal)
```

The aisle preference was confirmed once already, by the duplicate, so the next two confirmations keep it 60 and then 120 days. Items do not carry their expiry, so the script reads it from the table to show it. Confirming a memory that does not exist returns False.

## Two Writers

`put_if` with `expected=None` creates a memory only if no live memory has the key, so the second attempt to create it is a conflict. Then two writers read the same version with `record` and both add an airline. The first `put_if` applies. The second is a conflict, and its result carries the current memory, so the writer merges with it and writes again, expecting it:

```python
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


store.put_if(MEMORIES, "airlines", airlines(["TAP"]), expected=None)  # applied, version 1
store.put_if(MEMORIES, "airlines", airlines(["TAP"]), expected=None)  # a conflict
first = store.record(MEMORIES, "airlines")
second = store.record(MEMORIES, "airlines")
add_airline(store, first, "Iberia", "writer 1")
add_airline(store, second, "KLM", "writer 2")  # conflict, then merged
```

Neither update is lost: the memory ends at version 3 with both airlines.

## Forgetting

With history on, `delete` keeps the version it ends, and `get` no longer finds the memory. `purge` removes the memory with its vectors and its history. It also clears the history of a memory that was superseded:

```python
store.delete(MEMORIES, "airlines")
store.history(MEMORIES, "airlines")  # versions 3 (deleted), 2 and 1 (overwritten)
store.purge(MEMORIES, "airlines")  # True
store.purge(MEMORIES, "seat-window")  # True
```

## Output

```text
setup() created the store tables, with history, a vector index and TTL

1. A newer fact in the same slot supersedes the older one
   reconcile seat-window: written, version 1
   reconcile seat-aisle: written, version 1, superseded ['seat-window']
   record seat-aisle: version 1, provenance {'source': 'chat', 'thread': 'thread-2'}

2. A near duplicate is skipped, and confirms the memory it repeats
   reconcile seat-aisle-again: skipped, confirmed ['seat-aisle']
   seat-aisle is confirmed 1 time

3. A fact that arrives late goes straight into history
   reconcile seat-window-2025, true as of 2025-03-01: superseded
   live memories in the seat slot: ['seat-aisle']

4. History keeps each old version, why it ended and where it came from
   seat-window, version 1: 'I prefer window seats.'
      true from 2026-01-15 to 2026-10-29, superseded by seat-aisle, provenance {'source': 'chat', 'thread': 'thread-1'}
   seat-window-2025, version 1: 'I prefer window seats.'
      true from 2025-03-01 to 2026-10-29, superseded by seat-aisle, provenance {'source': 'chat', 'thread': 'thread-2025-03'}

5. Two steps: find_conflicts, a judge decides, resolve applies the decision
   find_conflicts no-peanut-allergy: allergy (similar, score 0.90): 'Allergic to peanuts.'
   judge: ['supersede allergy']
   meanwhile, another writer puts allergy again
   resolve: ConflictsChanged for ['allergy'], so find them again
   find_conflicts no-peanut-allergy: allergy (similar, score 0.90): 'Allergic to peanuts and cashews.'
   resolve: written, version 1, superseded ['allergy']

6. confirm renews a memory's expiry, for longer each time it is confirmed
   seat-aisle: confirmed 1 time, expires in 30 days
   confirm: True, confirmed 2 times, expires in 60 days
   confirm: True, confirmed 3 times, expires in 120 days
   confirm a memory that does not exist: False

7. Two writers update one memory with record() and put_if()
   put_if airlines, expected=None: applied, version 1
   again: conflict, the current memory is version 1
   both writers read version 1
   writer 1 adds Iberia: applied, version 2
   writer 2 adds KLM: conflict, the current memory is version 2
   writer 2 adds KLM: applied, version 3
   airlines, version 3: ['Iberia', 'KLM', 'TAP']

8. Forgetting: delete keeps history, purge removes the memory and its history
   delete airlines: get gives None, history [(3, 'deleted'), (2, 'overwritten'), (1, 'overwritten')]
   purge airlines: True, history now []
   purge seat-window: True, history now []
```

The dates in step 4 depend on the day you run it: the aisle preference is true from when it was written.

## The Async Version

`AsyncCockroachDBStore` has each of these methods as a coroutine with an `a` prefix: `areconcile`, `afind_conflicts`, `aresolve`, `arecord`, `aconfirm`, `aput_if`, `adelete_if`, `ahistory` and `apurge`. The policies and the TTL function stay plain functions.

## Next Steps

- [Keeping Memories True](../guides/long-term-memory.md#keeping-memories-true) - Candidates, policies, the two-step flow with an LLM, history and conditional writes
- [API Reference](../api/store.md) - Every method and type
- [Long-Term Memory Example](long-term-memory.md) - An agent with memory across threads
