"""Namespace labels with LIKE and regex metacharacters stay literal."""

from __future__ import annotations

from tests.integration.store.test_parity import as_set


def test_search_scope_with_metacharacter_labels(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    namespaces = [
        ("user_1",),
        ("userX1",),
        ("user_1", "child"),
        ("50%off",),
        ("50Xoff",),
        ("5000off",),
        ("back\\slash",),
        ("backXslash",),
        ("re+gex(*)",),
        ("regex",),
    ]
    for s in (store, ref):
        for ns in namespaces:
            s.put(ns, "k", {"ns": list(ns)})
    for prefix, expected in [
        (("user_1",), {("user_1",), ("user_1", "child")}),
        (("50%off",), {("50%off",)}),
        (("back\\slash",), {("back\\slash",)}),
        (("re+gex(*)",), {("re+gex(*)",)}),
    ]:
        mine, theirs = store.search(prefix), ref.search(prefix)
        assert as_set(mine) == as_set(theirs), prefix
        assert {item.namespace for item in mine} == expected, prefix
    for case in [
        {"prefix": ("user_1",)},
        {"prefix": ("50%off",)},
        {"suffix": ("re+gex(*)",)},
        {"prefix": ("back\\slash",)},
    ]:
        assert store.list_namespaces(**case) == ref.list_namespaces(**case), case
    assert store.delete_namespace(("user_1",)) == 2
    assert store.get(("userX1",), "k") is not None


def test_list_namespaces_wildcard_matches_exactly_one_label(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    namespaces = [
        ("a", "x", "c"),
        ("a", "x", "c", "d"),
        ("a", "x", "y", "c"),
        ("a", "c"),
        ("a", "xx", "c"),
        ("b", "x", "c"),
    ]
    for s in (store, ref):
        for ns in namespaces:
            s.put(ns, "k", {"v": 1})
    cases = [
        ({"prefix": ("a", "*", "c")}, [("a", "x", "c"), ("a", "x", "c", "d"), ("a", "xx", "c")]),
        (
            {"suffix": ("*", "c")},
            [("a", "c"), ("a", "x", "c"), ("a", "x", "y", "c"), ("a", "xx", "c"), ("b", "x", "c")],
        ),
        ({"suffix": ("x", "*")}, [("a", "x", "c"), ("b", "x", "c")]),
        (
            {"prefix": ("*", "x", "*")},
            [("a", "x", "c"), ("a", "x", "c", "d"), ("a", "x", "y", "c"), ("b", "x", "c")],
        ),
        (
            {"prefix": ("*",), "suffix": ("*", "*", "*", "*")},
            [("a", "x", "c", "d"), ("a", "x", "y", "c")],
        ),
    ]
    for case, expected in cases:
        mine = store.list_namespaces(**case)
        assert mine == ref.list_namespaces(**case), case
        assert mine == expected, case


# Listings with a prefix read only the key span of its literal labels. These
# show the range leaves out nothing InMemoryStore lists.
RANGE_NAMESPACES = [
    ("user",),
    ("user", "al"),
    ("user", "al", "notes"),
    ("user", "alice"),
    ("user", "alice", "memories"),
    ("user", "alice", "memories", "deep"),
    ("user", "alice", "profile"),
    ("user", "bob", "memories"),
    ("user", "bob", "notes", "memories"),
    ("users", "al"),
    ("a", "b"),
    ("ab", "c"),
]
RANGE_CASES: list[dict] = [
    {"prefix": ("user", "al")},
    {"prefix": ("user", "alice")},
    {"prefix": ("user", "a")},
    {"prefix": ("a",)},
    {"prefix": ("user", "*", "memories")},
    {"prefix": ("user", "*", "memories"), "max_depth": 2},
    {"prefix": ("user", "al"), "max_depth": 2},
    {"prefix": ("user",), "max_depth": 2},
    {"prefix": ("user",), "max_depth": 2, "limit": 2, "offset": 1},
    {"suffix": ("memories",)},
    {"suffix": ("*", "memories")},
    {"prefix": ("user", "alice"), "suffix": ("memories",)},
    {"prefix": ("user", "*"), "suffix": ("notes",)},
    {"prefix": ("*", "al")},
    {"prefix": ("user", "alice", "memories", "deep")},
]


def test_list_namespaces_prefix_range_hides_nothing(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        for ns in RANGE_NAMESPACES:
            s.put(ns, "k", {"v": 1})
    for case in RANGE_CASES:
        assert store.list_namespaces(**case) == ref.list_namespaces(**case), case
    assert store.list_namespaces(prefix=("user", "al")) == [("user", "al"), ("user", "al", "notes")]


def test_list_namespaces_prefix_range_with_bytes_around_the_dot(
    store_factory, reference, mode
) -> None:
    # "-", "#" and " " sort before ".", "/" right after it, "é" far above.
    # The store sorts by the dotted text, so where InMemoryStore's tuple
    # order differs these compare as sets
    store, ref = store_factory(mode), reference()
    namespaces = [
        ("user", "al"),
        ("user", "al", "x"),
        ("user", "al-x"),
        ("user", "al#x"),
        ("user", "al x"),
        ("user", "al/x"),
        ("user", "al/x", "y"),
        ("user", "alé"),
        ("user", "a"),
    ]
    for s in (store, ref):
        for ns in namespaces:
            s.put(ns, "k", {"v": 1})
    for case in [
        {"prefix": ("user", "al")},
        {"prefix": ("user", "al/x")},
        {"prefix": ("user", "al-x")},
        {"prefix": ("user", "a")},
        {"prefix": ("user", "*", "x")},
        {"prefix": ("user", "al"), "max_depth": 2},
    ]:
        assert sorted(store.list_namespaces(**case)) == sorted(ref.list_namespaces(**case)), case
    assert sorted(store.list_namespaces(prefix=("user", "al"))) == [
        ("user", "al"),
        ("user", "al", "x"),
    ]
