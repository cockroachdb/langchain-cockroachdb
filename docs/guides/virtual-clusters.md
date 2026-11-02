# Virtual-cluster connections

On a self-hosted CockroachDB deployment with cluster virtualization enabled,
choose the application virtual cluster with the connection URL's `options`
parameter:

```text
postgresql://app:PASSWORD@HOST:26257/agents?sslmode=verify-full&options=-ccluster%3Dmain
```

Here `main` selects the application virtual cluster, while `agents` selects a
database inside it. Create the database and grant the application user access
before connecting. Use the credentials and TLS parameters supplied for your
deployment, including `sslrootcert` when needed.

CockroachDB documents this routing syntax in
[Work with Virtual Clusters](https://www.cockroachlabs.com/docs/stable/work-with-virtual-clusters).
Application workloads belong on the application virtual cluster. The `system`
virtual cluster is for administration and physical cluster replication (PCR).
Consult the [self-hosted virtualization limitations](https://www.cockroachlabs.com/docs/stable/cluster-virtualization-overview#known-limitations)
for the supported topology. Those limits do not describe virtual clusters on
a Cockroach Continuum dedicated host. This guide assumes an existing
supported deployment.

## Managed virtual clusters

For a managed virtual cluster, use its complete SQL connection URL from the
Cloud console's connection dialog. Select the virtual cluster itself when
obtaining the URL. A dedicated host's console URL is a management address,
not a SQL connection string.

Keep the supplied hostname, port, database, TLS settings and any routing
parameters. Do not assume a managed virtual cluster is named `main` or
construct its connection string from the host cluster's identifier. The
engine, chat history and saver examples below accept the supplied SQL URL.

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

## Combine connection options

Use one `options` value containing all startup options, separated by spaces.
For example, to select `main` and set a statement timeout:

```python
from urllib.parse import quote, urlencode

query = urlencode({
    "sslmode": "verify-full",
    "options": "-ccluster=main -cstatement_timeout=5000",
}, quote_via=quote)
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
