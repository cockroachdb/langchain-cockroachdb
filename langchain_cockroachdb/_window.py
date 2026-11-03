"""When a search that filtered a window from a C-SPANN index must run again exactly.

CockroachDB serves a vector search from a C-SPANN index only when every
filter matches the index's prefix columns. So the vector store and the
memory store rank a window of the nearest rows from the index, with no
other filter, and apply the filter to that window. A row that fails the
filter still takes a place in the window, so a result shorter than asked for
may be missing matches that lie beyond it. Both stores decide with the rule
below whether to run the exact form of the search, which filters before it
ranks.
"""

from __future__ import annotations


def window_needs_exact_search(
    found: int, *, limit: int, window: int, window_size: int, scope_rows: int
) -> bool:
    """Whether a search that filtered a window from the index must run again exactly.

    A result of ``limit`` rows is complete. A shorter one is exact only when
    the index returned every row of the search's scope: the window is not
    full and its size equals the count of the scope. A full window may have
    left out rows beyond it, and the count cannot show otherwise, since it
    stops at the window's size. A window smaller than the count was cut short
    by the index itself, which a small beam size or small partitions can do.
    With no rows at all the sizes are unknown. All of these run again.

    Args:
        found: Rows the windowed search returned, after the filter.
        limit: Rows the search asked for.
        window: The LIMIT of the window.
        window_size: Rows the index returned into the window.
        scope_rows: Rows the search's scope holds (the namespace, or the
            whole table), counted up to ``window``.

    Returns:
        True when the exact form must run.
    """
    if found >= limit:
        return False
    if found == 0:
        return True
    return window_size >= window or window_size < scope_rows
