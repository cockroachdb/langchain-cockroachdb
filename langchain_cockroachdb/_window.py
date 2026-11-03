"""How large a window a search takes from a C-SPANN index, and when it runs again exactly.

CockroachDB serves a vector search from a C-SPANN index only when every
filter matches the index's prefix columns. So the vector store and the
memory store rank a window of the nearest rows from the index, with no
other filter, and apply the filter to that window. A row that fails the
filter still takes a place in the window, so a result shorter than asked for
may be missing matches that lie beyond it. Both stores size their windows
with window_for and decide with window_needs_exact_search whether to run
the exact form of the search, which filters before it ranks.
"""

from __future__ import annotations

# A window holds twice the rows a page needs, plus one so that a full window
# shows.
WINDOW_FACTOR = 2

# The fewest rows a window holds. On a 20,000-row table the index's lookup
# cost stayed flat up to a window of about 41 rows (272 rows read at 9, 299 at
# 41, 670 at 100). With a filter that 10% of rows pass, a window of 41 fills
# a page of 4 about 60% of the time, where a window of 9 does under 1%. A
# larger window never adds a fallback (see window_needs_exact_search).
WINDOW_FLOOR = 41


def window_for(rows: int) -> int:
    """The window for a page that needs ``rows`` rows: twice that plus one, at least WINDOW_FLOOR."""
    return max(WINDOW_FACTOR * rows + 1, WINDOW_FLOOR)


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
