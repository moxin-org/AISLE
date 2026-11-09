"""Physics-engine selection for the bridge and the harness (ADR-67).

The scene contract (SPEC 020) and the bridge contract (SPEC 030) are engine
neutral at the object level: the bridge talks to `robot`, entity, link and
camera objects through the small duck-typed surface Genesis exposes
natively. This package names the supported engines, resolves the attested
backend for each, and dispatches scene construction:

- ``genesis``: the frozen `aisle.scenes` builders, unchanged (CON-7).
- ``nexus``: `aisle.sim.nexus_backend`, which rebuilds the same scenes from
  the frozen pure functions (layout, placements, textures) on the Nexus
  GPU engine behind the same object surface.
- ``rapier``: `aisle.sim.rapier_backend` (ADR-68), the same scenes stepped by
  rapier on the CPU. It borrows the Nexus viewer as its renderer, so it needs
  both wheels, and gains a deterministic single-threaded step in exchange.

Nothing here imports a simulator at module level (CON-12): the engine name
alone is a plain string decision, and every simulator import happens inside
the builder that needs it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ENGINES: tuple[str, ...] = ("genesis", "nexus", "rapier")
DEFAULT_ENGINE = "genesis"
ENGINE_ENV_VAR = "AISLE_SIM_ENGINE"

# Backend names each engine accepts through AISLE_SIM_BACKEND (BRG-6).
# rapier steps on the CPU and takes no other backend; its renderer is the
# Nexus viewer, whose adapter choice is not the physics backend (ADR-68).
ENGINE_BACKENDS: dict[str, tuple[str, ...]] = {
    "genesis": ("cpu", "metal", "cuda"),
    "nexus": ("webgpu", "metal", "cuda", "cpu"),
    "rapier": ("cpu",),
}

# The importable module each engine needs. rapier needs two: the solver and
# the Nexus viewer it renders through.
ENGINE_MODULES: dict[str, tuple[str, ...]] = {
    "genesis": ("genesis",),
    "nexus": ("nexus3d",),
    "rapier": ("rapier3d", "nexus3d"),
}


def normalize_engine(name: str | None) -> str:
    """Validate an engine name; None means the default. Unknown names are
    refused rather than defaulted, for the same reason TC-9 refuses an
    unknown perception rung: a typo must not silently attest another engine."""
    if name is None:
        return DEFAULT_ENGINE
    engine = name.strip().lower()
    if engine not in ENGINES:
        raise ValueError(f"unknown simulation engine {name!r}; expected one of {ENGINES}")
    return engine


def select_engine(env: Mapping[str, str] | None = None) -> str:
    """The engine named by ``AISLE_SIM_ENGINE`` (default ``genesis``)."""
    env = os.environ if env is None else env
    return normalize_engine(env.get(ENGINE_ENV_VAR))


def select_nexus_backend(sim_extra: str, platform_name: str, cuda_available: bool = False) -> str:
    """Nexus counterpart of `select_genesis_backend`: the portable ``sim``
    selection maps to native Metal on macOS, which the locked
    dimforge-nexus3d wheel is built with there (ADR-70), and WebGPU
    elsewhere; ``cuda`` is the Linux-only explicit opt-in and fails closed
    without a device."""
    if sim_extra == "sim":
        return "metal" if platform_name == "Darwin" else "webgpu"
    if sim_extra != "cuda":
        raise ValueError(f"unknown simulation extra {sim_extra!r}; expected 'sim' or 'cuda'")
    if platform_name != "Linux":
        raise ValueError("the locked CUDA simulation extra is supported only on Linux")
    if not cuda_available:
        raise ValueError("the CUDA simulation extra requires an available CUDA device")
    return "cuda"


def select_rapier_backend(sim_extra: str, platform_name: str, cuda_available: bool = False) -> str:
    """rapier steps on the CPU on every platform (ADR-68), so the portable
    extra resolves to ``cpu`` and the CUDA extra has nothing to select."""
    if sim_extra == "sim":
        return "cpu"
    if sim_extra != "cuda":
        raise ValueError(f"unknown simulation extra {sim_extra!r}; expected 'sim' or 'cuda'")
    raise ValueError("the rapier engine is CPU only; the CUDA extra selects no rapier backend")


def select_sim_backend(
    engine: str, sim_extra: str, platform_name: str, cuda_available: bool = False
) -> str:
    """Resolve (engine, extra, platform) to one attested backend name."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.pharmacy import select_genesis_backend

        return select_genesis_backend(sim_extra, platform_name, cuda_available)
    if engine == "rapier":
        return select_rapier_backend(sim_extra, platform_name, cuda_available)
    return select_nexus_backend(sim_extra, platform_name, cuda_available)


def validate_backend(engine: str, backend: str | None) -> str | None:
    """Refuse a backend name the engine does not know (bridge config)."""
    engine = normalize_engine(engine)
    if backend is not None and backend not in ENGINE_BACKENDS[engine]:
        raise ValueError(
            f"unknown simulation backend {backend!r} for engine {engine!r}; "
            f"expected one of {ENGINE_BACKENDS[engine]}"
        )
    return backend


def engine_available(engine: str) -> bool:
    """Whether every Python package the engine needs is importable
    (collection-safe: uses find_spec, never imports the simulator). rapier
    needs its renderer too, so a missing nexus3d refuses it here rather than
    at the first render (ADR-68)."""
    import importlib.util

    modules = ENGINE_MODULES[normalize_engine(engine)]
    return all(importlib.util.find_spec(module) is not None for module in modules)


def engine_version(engine: str) -> str:
    """The installed simulator's version string, for `bridge_info`. The
    rapier engine reports its solver's version; its renderer, the locked
    dimforge-nexus3d wheel, is attested by the lock (ADR-68, ADR-70)."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        import genesis

        return str(genesis.__version__)
    from importlib.metadata import PackageNotFoundError, version

    dist, module = (
        ("rapier3d", "rapier3d") if engine == "rapier" else ("dimforge-nexus3d", "nexus3d")
    )
    try:
        return version(dist)
    except PackageNotFoundError:
        import importlib

        return str(getattr(importlib.import_module(module), "__version__", "unknown"))


def build_scene(engine: str, *args: Any, **kwargs: Any):
    """`aisle.scenes.pharmacy.build_scene` on the chosen engine."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.pharmacy import build_scene as genesis_build_scene

        return genesis_build_scene(*args, **kwargs)
    if engine == "rapier":
        from aisle.sim.rapier_backend import build_scene as rapier_build_scene

        return rapier_build_scene(*args, **kwargs)
    from aisle.sim.nexus_backend import build_scene as nexus_build_scene

    return nexus_build_scene(*args, **kwargs)


def build_store(engine: str, *args: Any, **kwargs: Any):
    """`aisle.scenes.store.build_store` on the chosen engine."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.store import build_store as genesis_build_store

        return genesis_build_store(*args, **kwargs)
    if engine == "rapier":
        from aisle.sim.rapier_backend import build_store as rapier_build_store

        return rapier_build_store(*args, **kwargs)
    from aisle.sim.nexus_backend import build_store as nexus_build_store

    return nexus_build_store(*args, **kwargs)
