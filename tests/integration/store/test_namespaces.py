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
