"""The async store's methods keep the bodies of their sync twins, apart from async and await.

CockroachDBStore and AsyncCockroachDBStore plan, check and decide with the
same code in base.py, so each pair of methods below differs only in its I/O:
the async one awaits where the sync one blocks. Their attempt bodies still
carry rules that only their order keeps, such as the sticky committed flag
set right after the writes go out, or the policy run under the mark. A fix
made in one store and not in the other would break parity with no decision
code changing, so these tests compare each pair's syntax trees.

Before comparing, both trees lose their docstrings, decorators and type
annotations; async functions, ``async with`` and ``await`` become their
plain forms; and the async store's names map back to the sync store's:
the coroutine forms of its helpers (``_areconcile_columns``) and the method
names its errors give (``"aput_if"``).
"""

from __future__ import annotations

import ast
import difflib
import inspect
from types import ModuleType

import pytest

from langchain_cockroachdb.store import aio, store

# (sync method, its async twin): the methods the async store gained for
# reconciliation, history and conditional writes, the writes and the schema
# check they share with plain puts, and the helpers that their attempts go
# through
TWINS = [
    ("history", "ahistory"),
    ("purge", "apurge"),
    ("find_conflicts", "afind_conflicts"),
    ("record", "arecord"),
    ("confirm", "aconfirm"),
    ("reconcile", "areconcile"),
    ("resolve", "aresolve"),
    ("put_if", "aput_if"),
    ("delete_if", "adelete_if"),
    ("_run_conditional", "_arun_conditional"),
    ("_send", "_send"),
    ("_cursor", "_cursor"),
    ("_read_all", "_read_all"),
    ("_embed_put", "_embed_put"),
    ("setup", "setup"),
    ("_wait_for_migration", "_wait_for_migration"),
    ("_set_ttl", "_set_ttl"),
    ("_run_puts", "_run_puts"),
    ("delete_namespace", "adelete_namespace"),
    ("_schema", "_aschema"),
    ("_read_schema", "_aread_schema"),
    ("_reconcile_columns", "_areconcile_columns"),
    ("_history_schema", "_ahistory_schema"),
    ("_history_table_exists", "_ahistory_table_exists"),
]

# Async names, as names, attributes or string constants, and the sync names
# they stand for: the store methods above, and the names that errors give
RENAMED = {
    "aembed_documents": "embed_documents",
    "alist_namespaces": "list_namespaces",
    **{async_name: sync_name for sync_name, async_name in TWINS if async_name != sync_name},
}

Method = ast.FunctionDef | ast.AsyncFunctionDef


def methods(module: ModuleType, class_name: str) -> dict[str, Method]:
    """The methods that ``class_name`` defines in ``module``, by name, as syntax trees."""
    tree = ast.parse(inspect.getsource(module))
    [found] = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    return {n.name: n for n in found.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


class Plain(ast.NodeTransformer):
    """Turns a method into the plain form that the comparison uses."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        # A docstring is the body's first statement, a string
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]  # type: ignore[attr-defined]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = RENAMED.get(node.name, node.name)
        node.decorator_list = []
        node.returns = None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        plain = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
            **({"type_params": node.type_params} if hasattr(node, "type_params") else {}),
        )
        return self.visit_FunctionDef(ast.copy_location(plain, node))

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        plain = ast.With(items=node.items, body=node.body, type_comment=node.type_comment)
        return self.generic_visit(ast.copy_location(plain, node))

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        # A typed assignment becomes a plain one; a bare declaration goes
        if node.value is None:
            return None
        plain = ast.Assign(targets=[node.target], value=node.value)
        return self.generic_visit(ast.copy_location(plain, node))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = RENAMED.get(node.id, node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = RENAMED.get(node.attr, node.attr)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            node.value = RENAMED.get(node.value, node.value)
        return node


def plain(method: Method) -> ast.AST:
    return ast.fix_missing_locations(Plain().visit(method))


def drift(sync_method: Method, async_method: Method) -> str:
    """How the plain forms of two methods differ, as a unified diff, or "" if they match."""
    mine, theirs = plain(sync_method), plain(async_method)
    if ast.dump(mine) == ast.dump(theirs):
        return ""
    return "\n".join(
        difflib.unified_diff(
            ast.unparse(mine).splitlines(),
            ast.unparse(theirs).splitlines(),
            "sync",
            "async",
            lineterm="",
        )
    )


SYNC = methods(store, "CockroachDBStore")
ASYNC = methods(aio, "AsyncCockroachDBStore")


@pytest.mark.parametrize(("sync_name", "async_name"), TWINS, ids=[name for _, name in TWINS])
def test_the_async_method_keeps_the_body_of_its_sync_twin(sync_name: str, async_name: str) -> None:
    difference = drift(SYNC[sync_name], ASYNC[async_name])
    assert not difference, (
        f"AsyncCockroachDBStore.{async_name} has drifted from CockroachDBStore.{sync_name}: "
        "beyond async and await, their bodies differ. Make the same change in both.\n"
        f"{difference}"
    )


def test_the_comparison_sees_more_than_async_and_await() -> None:
    def parse(source: str) -> Method:
        node = ast.parse(source).body[0]
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        return node

    sync_form = parse(
        'def confirm(self, key: str) -> bool:\n    """Doc."""\n    ok: bool = self.run(key)\n'
        "    return ok"
    )
    # The same body as a coroutine: only the async forms and the names differ
    same = parse("async def aconfirm(self, key):\n    ok = await self.run(key)\n    return ok")
    assert drift(sync_form, same) == ""
    # A statement more, another argument or another constant is a drift
    for changed in (
        "async def aconfirm(self, key):\n    ok = await self.run(key)\n    self.log()\n    return ok",
        "async def aconfirm(self, key):\n    ok = await self.run(key, True)\n    return ok",
        "async def aconfirm(self, key):\n    ok = await self.run('k')\n    return ok",
    ):
        assert drift(sync_form, parse(changed)) != ""
