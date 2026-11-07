# Virtual-cluster connections

A virtual cluster is a database target selected by the connection URL. Point
each client (the engine, chat history, and the checkpointers) at the SQL URL
for the virtual cluster you want, and the library preserves that routing. How
you select a virtual cluster depends on whether it is managed or self-hosted.

## Managed virtual clusters (Cockroach Continuum)

A Cockroach Continuum deployment is a host cluster, a Mission Critical cluster
with virtualization enabled, currently in preview. Each virtual cluster on the
host has its own SQL connection string, its own network security configuration,
and its own SQL users. Generate a virtual cluster's connection string from the
Cloud Console's Connect dialog or with the `ccloud` CLI, and use it as is:

```text
postgresql://app:PASSWORD@vc-endpoint.aws-us-east-2.cockroachlabs.cloud:26257/agents?sslmode=verify-full
```

Do not add a `-ccluster` option to a Continuum connection string; the generated
string already reaches its virtual cluster. A host cluster's console URL is a
management address, not a SQL connection string. Keep the supplied hostname,
port, database, and TLS parameters, including `sslrootcert` when needed, and do
not assume a virtual cluster is named `main` or build its connection string from
the host's identifier. An application connected to a virtual cluster cannot see
or reach the host or any other virtual cluster in the workspace. See
[Virtual clusters on a Continuum host](https://docs.cockroachlabs.com/docs/continuum/virtual-clusters).
This integration does not provision virtual clusters.

## Self-hosted virtual clusters

On a self-hosted deployment with cluster virtualization enabled, select the
application virtual cluster with the connection URL's `options` parameter:

```text
postgresql://app:PASSWORD@HOST:26257/agents?sslmode=verify-full&options=-ccluster%3Dmain
```

Here `main` selects the application virtual cluster, while `agents` selects a
database inside it. Create the database and grant the application user access
before connecting. Application workloads belong on the application virtual
cluster; the `system` virtual cluster is for administration and physical
cluster replication (PCR). CockroachDB documents this routing syntax in
[Work with Virtual Clusters](https://www.cockroachlabs.com/docs/stable/work-with-virtual-clusters).
The [self-hosted virtualization limitations](https://www.cockroachlabs.com/docs/stable/cluster-virtualization-overview#known-limitations)
currently allow one system virtual cluster and one application virtual cluster
per physical cluster, so testing two self-hosted application virtual clusters
requires two physical clusters, or one virtualized cluster that compares the
`main` and `system` virtual clusters. The engine, chat history and saver
examples below accept the supplied SQL URL either way.

## Engines and chat history

Store the complete connection URL in `COCKROACHDB_URL`:

```python
import os

from langchain_cockroachdb import CockroachDBChatMessageHistory, CockroachDBEngine

engine = CockroachDBEngine.from_connection_string(os.environ["COCKROACHDB_URL"])
history = CockroachDBChatMessageHistory("conversation-1", engine=engine)
```

The vector store uses the same engine. Chat history can also take the URL
directly through `connection_string=`. A shared engine reuses one connection
pool; close it when the application shuts down.

The library normalizes `postgresql://`, `cockroachdb://`, and
`cockroachdb+psycopg://` to the SQLAlchemy CockroachDB dialect with psycopg 3.
It preserves the rest of the URL, including virtual-cluster routing, TLS
parameters, and other connection settings.

## Checkpointers

Both saver factories accept the same URL:

```python
import os

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver

with CockroachDBSaver.from_conn_string(os.environ["COCKROACHDB_URL"]) as saver:
    saver.setup()


async def setup_async() -> None:
    async with AsyncCockroachDBSaver.from_conn_string(os.environ["COCKROACHDB_URL"]) as saver:
        await saver.setup()
```

The factories normalize the URL to `postgresql://` for psycopg and preserve its
query parameters. For concurrent agents, use a
[pooled checkpointer](checkpointer.md#connection-pool) and pass a
`postgresql://` URL directly to `psycopg_pool`. Pool constructors are provided
by psycopg and do not normalize `cockroachdb://` URLs.

## Long-term memory store

`CockroachDBStore` and `AsyncCockroachDBStore` accept the same URL through
`from_conn_string`, including the `pool_config` path, which builds the pool from
the normalized URL. Point them at the same virtual cluster:

```python
import os

from langchain_cockroachdb.store import CockroachDBStore

with CockroachDBStore.from_conn_string(os.environ["COCKROACHDB_URL"]) as store:
    store.setup()
```

See the [long-term memory store guide](long-term-memory.md) for indexes, TTL and
search.

## Combine connection options

Startup options such as the self-hosted `-ccluster` selector go in one
`options` value, separated by spaces. If a connection string already carries an
`options` value, append to it rather than replace it. For example, to select a
self-hosted `main` cluster and set a statement timeout:

```python
from urllib.parse import quote, urlencode

query = urlencode(
    {
        "sslmode": "verify-full",
        "options": "-ccluster=main -cstatement_timeout=5000",
    },
    quote_via=quote,
)
connection_url = f"postgresql://app:PASSWORD@HOST:26257/agents?{query}"
```

Encode query values once. `%3D` encodes `=` and `%20` encodes a space.
Use `quote_via=quote` when building a URL with `urlencode`: its default `+`
encoding for spaces is not decoded as a space by libpq. A literal `+` in a
query value should be encoded as `%2B` so both connection paths preserve it.
Avoid duplicate `options` query parameters. Driver keyword arguments
(`connect_args` on engines or `connect_kwargs` on saver factories) override
URL values, so an `options` keyword must contain the cluster selector too.

## Routing and isolation

The cluster selector chooses the database target for a connection. Vector
namespaces and checkpoint `thread_id` values scope application data inside
that target. They do not authorize access or make agents communicate. Apply
database privileges and application authorization for tenant access, and use
distinct thread IDs for independent conversations.

URL compatibility does not establish database failover or external tool
execution guarantees. See [Retries and Resilience](resilience.md) for
connection-loss handling.
