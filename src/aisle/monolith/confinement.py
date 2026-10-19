"""In-process confinement of a monolithic orchestration module (SPEC 440,
MON-6/MON-7 — the broker-side half; the OS-level half is issue #353).

The agent module is executed by the broker inside a guarded window: its
imports go through `guarded_import`, which refuses the trusted AISLE assets
(verifier, reset, bridge, guard, driver, harness, registry), the dora
runtime, and every process/socket/network/FFI/introspection route the
MON-7 conformance list names. `open` is refused too, so the module has no
filesystem path of its own (the #353 profile closes the paths a module can
reach without Python). A refusal raises `ConfinementViolation`, which the
broker turns into an infrastructure-invalid record BEFORE any command or
score (MON-7: "denied or produce an infrastructure-invalid record").

`TrustedIntegrity` snapshots the code objects of the in-process trusted
callables (turn stamping, the pinned primitives) when the broker starts and
re-verifies them after every agent callback: a monkeypatch that swaps one
of them is caught before the next command leaves the broker.

This is not a sandbox: a module that reaches `builtins` through a literal
type walk can defeat an in-process hook. It is the fail-closed record layer
MON-7 asks for; the confinement policy of #353 is what makes bypass
impossible rather than recorded.
"""

from __future__ import annotations

import builtins
import hashlib
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: module names (and prefixes) a monolithic module may never import; the
#: MON-7 list — trusted assets, runtime, process, socket, network, FFI,
#: filesystem and introspection routes
DENIED_IMPORTS: tuple[str, ...] = (
    # trusted AISLE assets (MON-6)
    "aisle.verifier",
    "aisle.reset",
    "aisle.harness",
    "aisle.nodes.dora_genesis",
    "aisle.nodes.budget_guard",
    "aisle.nodes.so101_driver",
    "aisle.nodes.g1_driver",
    "aisle.nodes.turn_barrier",
    "aisle.monolith.confinement",
    "aisle.monolith.launcher",
    "aisle.turn_node",
    "aisle.topics",
    "aisle.registry",
    # the typed arm's runtime and facilities (MON-3)
    "dora",
    "yaml",
    # process / socket / network
    "subprocess",
    "multiprocessing",
    "socket",
    "ssl",
    "http",
    "urllib",
    "asyncio",
    "concurrent",
    "threading",
    "signal",
    # FFI, introspection, loaders, filesystem
    "ctypes",
    "cffi",
    "gc",
    "inspect",
    "importlib",
    "pkgutil",
    "runpy",
    "sys",
    "os",
    "io",
    "pathlib",
    "shutil",
    "tempfile",
    "glob",
    "builtins",
    "code",
    "codeop",
    "marshal",
    "pickle",
    "shelve",
    "sqlite3",
)

#: the marker the broker plants in the module's globals; only frames whose
#: globals carry it are subject to the hook (the broker's own imports and
#: the primitives' lazy imports are untouched)
MARKER = "__aisle_monolith__"


class ConfinementViolation(RuntimeError):
    """A denied route was attempted; the broker records it as
    infrastructure-invalid and stops before any command."""

    def __init__(self, route: str, detail: str) -> None:
        super().__init__(f"{route}: {detail}")
        self.route = route
        self.detail = detail


def denied(name: str) -> bool:
    return any(name == d or name.startswith(d + ".") for d in DENIED_IMPORTS)


def check_import(name: str, fromlist: tuple | None, level: int) -> None:
    """Raise for a denied absolute import (or a `from pkg import child`
    whose child resolves into a denied module); relative imports have no
    package in a single-file module and are denied outright."""
    if level and level > 0:
        raise ConfinementViolation("import", f"relative import in a single-file module: {name!r}")
    if denied(name):
        raise ConfinementViolation("import", f"denied module {name!r}")
    for child in fromlist or ():
        if isinstance(child, str) and denied(f"{name}.{child}"):
            raise ConfinementViolation("import", f"denied module {name}.{child!r}")


_real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """builtins.__import__ replacement: the MON-7 denial list applies to
    frames that carry the monolith marker in their globals — the agent
    module and anything it defines — and to nothing else."""
    if isinstance(globals, dict) and globals.get(MARKER):
        check_import(name, tuple(fromlist or ()), level)
    return _real_import(name, globals, locals, fromlist, level)


def strict_import(name, globals=None, locals=None, fromlist=(), level=0):
    """The `__import__` installed in the module's builtins: every import
    statement compiled into the module — and any bare `__import__(...)`
    call it makes — resolves here, so the denial list applies regardless
    of which globals the call carries."""
    check_import(name, tuple(fromlist or ()), level)
    return _real_import(name, globals, locals, fromlist, level)


def _denied_open(*args, **kwargs):
    raise ConfinementViolation("filesystem", "open() is not available to a monolithic module")


def restricted_builtins() -> dict:
    """The builtins namespace an agent module executes with: the real
    builtins minus filesystem/loader routes, plus the guarded importer."""
    table = dict(vars(builtins))
    table["__import__"] = strict_import
    table["open"] = _denied_open
    for name in ("exit", "quit", "breakpoint", "input", "help"):
        table.pop(name, None)
    return table


def load_module(path: str, source: str) -> dict:
    """Execute one monolithic module's source in a marked namespace with the
    restricted builtins; returns the namespace (MON-3: one ordinary module,
    ordinary syntax/import/runtime errors surface as-is)."""
    namespace: dict = {
        "__name__": "aisle_monolith_module",
        "__file__": path,
        "__builtins__": restricted_builtins(),
        MARKER: True,
    }
    code = compile(source, path, "exec")
    exec(code, namespace)  # noqa: S102 — the module IS the deliverable under test
    return namespace


@contextmanager
def guarded():
    """Install the guarded importer for the duration of an agent callback so
    imports the module performs lazily (inside a function body) are checked
    exactly like top-level ones."""
    previous = builtins.__import__
    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = previous


def _code_digest(obj: Any) -> str:
    code = getattr(obj, "__code__", None)
    if code is None and hasattr(obj, "__func__"):
        code = obj.__func__.__code__
    if code is None:
        return hashlib.sha256(repr(obj).encode()).hexdigest()
    return hashlib.sha256(
        code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode()
    ).hexdigest()


@dataclass
class TrustedIntegrity:
    """Snapshot-and-verify of in-process trusted callables (MON-7
    monkeypatching / replacement of trusted modules)."""

    targets: dict[str, Callable[[], Any]]
    baseline: dict[str, tuple[int, str]] = field(default_factory=dict)

    def snapshot(self) -> None:
        self.baseline = {
            name: (id(getter()), _code_digest(getter())) for name, getter in self.targets.items()
        }

    def verify(self) -> None:
        for name, getter in self.targets.items():
            current = getter()
            expected_id, expected_digest = self.baseline[name]
            if id(current) != expected_id or _code_digest(current) != expected_digest:
                raise ConfinementViolation("monkeypatch", f"trusted callable {name} was replaced")


def default_integrity() -> TrustedIntegrity:
    """The broker's trusted-callable set: turn stamping (the guard's trust
    boundary) and the pinned primitives of both perception rungs."""
    from aisle import topics, turn_node
    from aisle.monolith import primitives
    from aisle.nodes import grasp_topdown, ik_trajectory, l2_pose, segmented_pose
    from aisle.verifier import stages

    return TrustedIntegrity(
        {
            "primitives.Primitives.pose_session": lambda: primitives.Primitives.pose_session,
            "primitives.Primitives.plan_grasp": lambda: primitives.Primitives.plan_grasp,
            "primitives.Primitives.staged_plan": lambda: primitives.Primitives.staged_plan,
            "primitives.Primitives.streamer": lambda: primitives.Primitives.streamer,
            "topics.stamp": lambda: topics.stamp,
            "turn_node.Node.send_output": lambda: turn_node.Node.send_output,
            "segmented_pose.estimate_pose": lambda: segmented_pose.estimate_pose,
            "segmented_pose.L1Session._estimate": lambda: segmented_pose.L1Session._estimate,
            "primitives.Primitives.l2_pose_session": lambda: primitives.Primitives.l2_pose_session,
            "l2_pose.L2Session._estimate": lambda: l2_pose.L2Session._estimate,
            "l2_pose.pick_target_detection": lambda: l2_pose.pick_target_detection,
            "grasp_topdown.plan_grasp": lambda: grasp_topdown.plan_grasp,
            "ik_trajectory.StagedPlan.__init__": lambda: ik_trajectory.StagedPlan.__init__,
            "ik_trajectory.StageStreamer.step": lambda: ik_trajectory.StageStreamer.step,
            "verifier.stages.backproject_overhead": lambda: stages.backproject_overhead,
        }
    )
