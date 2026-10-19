"""Dataflow validator (SPEC 060 VAL-1..7, CON-8).

Loads a dora dataflow YAML plus every registry manifest and rejects graphs
that cannot run safely: unresolved node ids, duplicate ids, missing
producers, schema mismatches against the CAP-2 vocabulary, oracle leaks
(VAL-6), ungated motion (VAL-5), motion nodes without evalcards, and
pip:-sourced nodes whose distribution is not installed (INSTALL_MISSING).
Hints are the research agent's learning signal: every error names a registry
capability or a concrete fix. No genesis or dora imports (unit territory).
"""

import difflib
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

import yaml

from aisle.harness.registry import (
    MOTION_SINK_PORTS,
    _path_source_valid,
    _pip_dist,
    _pip_installed,
    load_capability_schema,
    load_manifests,
    load_vocabulary,
    manifest_schema_errors,
)

# MOTION_SINK_PORTS lives in registry.py (shared with the ADR-5 lint rule):
# base_cmd is a motion sink too (SPEC 210 MOB-3) — a mobile base command
# reaching the bridge MUST traverse the budget guard.
GUARD_ID = "budget-guard"
RATE_BAND = 0.2  # TC-4: rates are contracts within ±20%
# ADR-29: the guard's stats tick doubles as the watchdog wall-net sweep, so
# a mobile graph must wire it from a real dora timer no slower than this
# (the checked-in graphs use 5000 ms; the limits.toml wall-net latency
# story assumes it)
GUARD_TICK_MAX_MS = 5000
# MOB-4: each embodiment profile resolves to an ARM kind. `mobile` is the
# franka arm on a differential-drive base, so franka-arm capabilities work
# unchanged under `mobile`; only base-requiring nodes distinguish them.
EMBODIMENT_ARM = {
    "franka": "franka",
    "so101": "so101",
    "mobile": "franka",
    "g1_right": "g1_right",  # G1-E: harness accepts; G1-F owns kinematics/limits
}


def _entry(code: str, where: dict, detail: str, hint: str) -> dict:
    return {"code": code, **where, "detail": detail, "hint": hint}


def _closest(name: str, candidates: list[str]) -> str:
    # cutoff 0.75: a weak match ("warp-drive" ~ "arm-driver-sim") is a
    # misleading hint, worse than none
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.75)
    return matches[0] if matches else ""


def _input_source(raw) -> str | None:
    """Unwrap a graph input to its source string: dora's extended form
    ({source: ..., queue_size: N}) or the plain string. None for a
    missing/empty/non-string source. Issue #160 item 3: this idiom had
    FIVE hand-rolled copies that were starting to disagree."""
    source = raw.get("source") if isinstance(raw, dict) else raw
    return source if isinstance(source, str) and source else None


def _backward_sources(node: dict) -> list[str | None]:
    """Backward-edge sources of a node; None for timers/dora/malformed."""
    sources: list[str | None] = []
    for raw in (node.get("inputs") or {}).values():
        source = _input_source(raw)
        if source is not None and not source.startswith("dora/"):
            sources.append(source)
        else:
            sources.append(None)
    return sources


def _effective_manifest_port(port: str, declared: dict) -> str | None:
    """Resolve dora's indexed fan-in convention against one manifest map."""
    if port in declared:
        return port
    base, _, suffix = port.rpartition("_")
    return base if suffix.isdigit() and base in declared else None


def _clock_topology(nodes: list[dict], manifests: dict[str, dict]) -> dict:
    """Return the participant set and edge classifier shared by validation/compiler."""
    graph_nodes = {str(node.get("id", "")): node for node in nodes}
    bridge_ids = {
        node_id
        for node_id in graph_nodes
        if "sim_bridge" in ((manifests.get(node_id) or {}).get("provides") or [])
    }
    barrier_ids = {
        node_id
        for node_id in graph_nodes
        if "turn_barrier" in ((manifests.get(node_id) or {}).get("provides") or [])
    }

    def manifest_for(node_id: str) -> dict:
        manifest = manifests.get(node_id)
        if manifest is not None:
            return manifest
        stamped = re.fullmatch(r"(?P<base>.+)-a\d+", node_id)
        return manifests.get(stamped.group("base"), {}) if stamped else {}

    def edge_kind(consumer: str, port: str) -> str:
        declared = manifest_for(consumer).get("inputs", {})
        effective = _effective_manifest_port(port, declared)
        return (declared.get(effective, {}) if effective else {}).get("turn_edge", "forward")

    participants: set[str] = set()
    frontier: list[str] = []
    for bridge_id in sorted(bridge_ids):
        bridge = graph_nodes[bridge_id]
        for port, raw in (bridge.get("inputs") or {}).items():
            base_port = port.rpartition("_")[0] if port.rpartition("_")[2].isdigit() else port
            if base_port not in MOTION_SINK_PORTS | {"reset"}:
                continue
            source = _input_source(raw)
            if source and not source.startswith("dora/"):
                frontier.append(source.partition("/")[0])
    while frontier:
        node_id = frontier.pop()
        if node_id in participants or node_id in bridge_ids or node_id in barrier_ids:
            continue
        node = graph_nodes.get(node_id)
        if node is None:
            continue
        participants.add(node_id)
        for port, raw in (node.get("inputs") or {}).items():
            declared = manifest_for(node_id).get("inputs", {})
            effective = _effective_manifest_port(port, declared)
            if effective and declared[effective].get("is_clock") is True:
                continue
            source = _input_source(raw)
            if source and not source.startswith("dora/"):
                frontier.append(source.partition("/")[0])
    return {
        "graph_nodes": graph_nodes,
        "bridge_ids": bridge_ids,
        "barrier_ids": barrier_ids,
        "participants": participants,
        "manifest_for": manifest_for,
        "edge_kind": edge_kind,
    }


def compile_turn_plan(nodes: list[dict], manifests: dict[str, dict]) -> dict:
    """Compile a validated graph into the terminal barrier's runtime plan."""
    topology = _clock_topology(nodes, manifests)
    bridges = sorted(topology["bridge_ids"])
    barriers = sorted(topology["barrier_ids"])
    if len(bridges) != 1 or len(barriers) != 1:
        raise ValueError("a lockstep runtime plan requires one bridge and one turn barrier")
    graph_nodes = topology["graph_nodes"]
    participants: dict[str, dict] = {}
    for node_id in sorted(topology["participants"]):
        inputs = {}
        manifest = topology["manifest_for"](node_id)
        declared = manifest.get("inputs", {})
        for port, raw in sorted((graph_nodes[node_id].get("inputs") or {}).items()):
            effective = _effective_manifest_port(port, declared)
            if effective and declared[effective].get("is_clock") is True:
                continue
            source = _input_source(raw)
            if source is None or source.startswith("dora/"):
                continue
            producer, _, output = source.partition("/")
            if producer not in topology["participants"] | topology["bridge_ids"]:
                continue
            inputs[port] = {
                "source": producer,
                "output": output,
                "edge": topology["edge_kind"](node_id, port),
            }
        participants[node_id] = {
            "inputs": inputs,
            "outputs": sorted(graph_nodes[node_id].get("outputs") or []),
            "verdict_bearing": node_id == "verifier-realistic",
        }

    barrier = graph_nodes[barriers[0]]
    done_ports = {}
    for port, raw in sorted((barrier.get("inputs") or {}).items()):
        source = _input_source(raw)
        producer, _, output = str(source or "").partition("/")
        if producer in participants and output == "turn_done":
            done_ports[port] = producer
    bridge_inputs = {}
    bridge = graph_nodes[bridges[0]]
    bridge_manifest = topology["manifest_for"](bridges[0])
    declared_bridge_inputs = bridge_manifest.get("inputs", {})
    for port, raw in sorted((bridge.get("inputs") or {}).items()):
        effective = _effective_manifest_port(port, declared_bridge_inputs)
        if effective and declared_bridge_inputs[effective].get("is_clock") is True:
            continue
        source = _input_source(raw)
        if source is None or source.startswith("dora/"):
            continue
        producer, _, output = source.partition("/")
        if producer in participants:
            bridge_inputs[port] = {"source": producer, "output": output}
    return {
        "bridge": bridges[0],
        "bridge_outputs": sorted(graph_nodes[bridges[0]].get("outputs") or []),
        "bridge_inputs": bridge_inputs,
        "barrier": barriers[0],
        "participants": participants,
        "done_ports": done_ports,
    }


def _clock_errors(nodes: list[dict], manifests: dict[str, dict]) -> list[dict]:
    """Validate ADR-30 clock participation and stratified topology (VAL-2).

    Clock validation applies to simulation graphs (those containing a
    ``sim_bridge`` provider).  Registry-only subgraphs remain independently
    compilable; once connected to a bridge, every causal path to reset or a
    motion input must join the closed-turn protocol.
    """
    errors: list[dict] = []
    topology = _clock_topology(nodes, manifests)
    graph_nodes = topology["graph_nodes"]
    bridge_ids = topology["bridge_ids"]
    if not bridge_ids:
        return []
    barrier_ids = topology["barrier_ids"]
    manifest_for = topology["manifest_for"]
    edge_kind = topology["edge_kind"]

    # Participant reachability follows all causal edges (including episodic
    # back-edges); episodic classification removes those edges only from the
    # within-turn cycle check.  This keeps client/verifier service loops in
    # the barrier while measurement taps with no path to bridge actuation stay
    # exempt by construction.
    participants = topology["participants"]

    def add(code: str, where: dict, detail: str, hint: str) -> None:
        errors.append(_entry(code, where, detail, hint))

    # Every declared structural clock is transported honestly and comes from
    # the one source class allowed at that point in the protocol.
    done_sources: set[str] = set()
    for node_id, node in sorted(graph_nodes.items()):
        manifest = manifest_for(node_id)
        declared = manifest.get("inputs", {})
        for port, raw in sorted((node.get("inputs") or {}).items()):
            effective = _effective_manifest_port(port, declared)
            spec = declared.get(effective, {}) if effective else {}
            if spec.get("is_clock") is not True:
                continue
            source = _input_source(raw)
            edge = {"edge": f"{source or '<missing>'} -> {node_id}/{port}"}
            queue_ok = (
                isinstance(raw, dict)
                and isinstance(raw.get("queue_size"), int)
                and not isinstance(raw.get("queue_size"), bool)
                and raw["queue_size"] > 0
                and raw.get("queue_policy") == "backpressure"
            )
            if not queue_ok:
                add(
                    "CLOCK_DROPPED",
                    edge,
                    f"clock input {node_id}/{port} is not an explicit positive backpressure queue",
                    "set queue_size to a positive integer and queue_policy: backpressure",
                )
            producer, _, output = str(source or "").partition("/")
            valid = False
            if node_id in participants:
                valid = producer in barrier_ids and output == "turn"
            elif node_id in barrier_ids and effective == "sim_turn":
                valid = producer in bridge_ids and output == "sim_turn"
            elif node_id in barrier_ids and effective == "done":
                valid = producer in participants and output == "turn_done"
                if valid:
                    done_sources.add(producer)
            elif node_id in bridge_ids and effective == "turn_commit":
                valid = producer in barrier_ids and output == "turn_commit"
            if not valid:
                add(
                    "CLOCK_SOURCE_INVALID",
                    edge,
                    f"{node_id}/{port} clock source {source!r} is not valid for its "
                    "ADR-30 protocol role",
                    "wire participant clocks from turn-barrier/turn, barrier sim_turn "
                    "from the bridge, done inputs from participant/turn_done, and the "
                    "bridge commit from turn-barrier/turn_commit",
                )

    for node_id in sorted(participants):
        node = graph_nodes[node_id]
        manifest = manifest_for(node_id)
        clock_ports = [
            port
            for port, spec in (manifest.get("inputs") or {}).items()
            if isinstance(spec, dict) and spec.get("is_clock") is True
        ]
        wired = [port for port in clock_ports if port in (node.get("inputs") or {})]
        graph_outputs = set(node.get("outputs") or [])
        manifest_outputs = set(manifest.get("outputs") or {})
        if (
            len(wired) != 1
            or "turn_done" not in graph_outputs
            or "turn_done" not in manifest_outputs
        ):
            add(
                "CLOCK_PATH_INCOMPLETE",
                {"node": node_id},
                f"participant {node_id!r} must wire exactly one is_clock input and "
                "declare turn_done in both graph and manifest",
                "wire turn from turn-barrier/turn with backpressure and add turn_done output",
            )
        env = node.get("env") or {}
        configured_outputs = {
            item for item in str(env.get("AISLE_TURN_OUTPUTS", "")).split(",") if item
        }
        configured_node = str(env.get("AISLE_TURN_NODE", ""))
        lockstep = str(env.get("AISLE_LOCKSTEP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not lockstep or configured_node != node_id or configured_outputs != graph_outputs:
            add(
                "CLOCK_PATH_INCOMPLETE",
                {"node": node_id},
                f"participant {node_id!r} lockstep env must name the node and enumerate "
                f"exact graph outputs; configured={sorted(configured_outputs)}, "
                f"graph={sorted(graph_outputs)}",
                "set AISLE_LOCKSTEP=1, AISLE_TURN_NODE to the node id, and "
                "AISLE_TURN_OUTPUTS to every graph output",
            )
        if node_id not in done_sources:
            add(
                "CLOCK_PATH_INCOMPLETE",
                {"node": node_id},
                f"participant {node_id!r} has no turn_done edge into the terminal barrier",
                "wire participant/turn_done to one indexed turn-barrier done input",
            )

    # The bridge accepts one and only one commit edge.  Indexed aliases count
    # too, so duplicating the input cannot evade the cardinality check.
    for bridge_id in sorted(bridge_ids):
        bridge = graph_nodes[bridge_id]
        bridge_env = bridge.get("env") or {}
        bridge_outputs = set(bridge.get("outputs") or [])
        configured_outputs = {
            item for item in str(bridge_env.get("AISLE_TURN_OUTPUTS", "")).split(",") if item
        }
        if configured_outputs != bridge_outputs:
            add(
                "CLOCK_PATH_INCOMPLETE",
                {"node": bridge_id},
                f"bridge {bridge_id!r} AISLE_TURN_OUTPUTS must enumerate exact graph "
                f"outputs; configured={sorted(configured_outputs)}, "
                f"graph={sorted(bridge_outputs)}",
                "set AISLE_TURN_OUTPUTS to every bridge graph output",
            )
        bridge_inputs = bridge.get("inputs") or {}
        commits = []
        for port, raw in bridge_inputs.items():
            base, _, suffix = port.rpartition("_")
            if port == "turn_commit" or (base == "turn_commit" and suffix.isdigit()):
                commits.append(_input_source(raw))
        valid_commits = [
            source
            for source in commits
            if source
            and source.partition("/")[0] in barrier_ids
            and source.partition("/")[2] == "turn_commit"
        ]
        if len(commits) != 1 or len(valid_commits) != 1:
            add(
                "CLOCK_COMMIT_COUNT",
                {"node": bridge_id},
                f"bridge {bridge_id!r} has {len(commits)} commit inputs and "
                f"{len(valid_commits)} valid terminal sources; expected exactly one",
                "wire exactly one turn_commit from turn-barrier/turn_commit",
            )

    # Forward-only causal graph must be acyclic.  Bridge and barrier edges
    # are turn boundaries, not within-turn dependencies.
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in participants}
    for consumer in sorted(participants):
        for port, raw in (graph_nodes[consumer].get("inputs") or {}).items():
            if edge_kind(consumer, port) == "episodic":
                continue
            declared = manifest_for(consumer).get("inputs", {})
            effective = _effective_manifest_port(port, declared)
            if effective and declared[effective].get("is_clock") is True:
                continue
            source = _input_source(raw)
            producer = source.partition("/")[0] if source else ""
            if producer in participants:
                adjacency[producer].add(consumer)

    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] | None = None

    def visit(node_id: str, path: list[str]) -> None:
        nonlocal cycle
        if cycle is not None or node_id in visited:
            return
        if node_id in visiting:
            start = path.index(node_id)
            cycle = path[start:] + [node_id]
            return
        visiting.add(node_id)
        for downstream in sorted(adjacency[node_id]):
            visit(downstream, path + [downstream])
        visiting.discard(node_id)
        visited.add(node_id)

    for node_id in sorted(adjacency):
        visit(node_id, [node_id])
    # An episodic edge must be a genuine BACK edge (issue #220). The rule
    # above uses `turn_edge` only to DELETE edges from the cycle check, so
    # nothing asked whether the producer is actually scheduled AFTER the
    # consumer. On a FORWARD edge the label is not a harmless mislabel: the
    # consumer resolves an episodic input from the PREVIOUS turn's
    # watermark while dora delivers the current turn's message, so the first
    # time the producer emits, ParticipantTurn sees an excess input with a
    # wrong stamp and raises — the barrier watchdog then kills the dataflow
    # about a second in. `expert_s1` and `expert_t4` both shipped that way.
    # Forward-DAG layer: the barrier releases a participant once its forward
    # inputs are satisfied, so a strictly LOWER layer means "emits earlier in
    # this same turn". Equal-or-higher is safe: the producer has not run yet
    # when the consumer opens (a genuine back edge), or the two are unordered
    # and the barrier never places the producer first. Verified against the
    # corpus: this marks `expert_s1` and `expert_t4` and leaves `expert_t0`
    # through `t3` alone, which matches what actually runs.
    layer_of: dict[str, int] = {}

    def forward_layer(node_id: str, stack: tuple[str, ...] = ()) -> int:
        if node_id in layer_of:
            return layer_of[node_id]
        if node_id in stack:  # defensive: CLOCK_CYCLE reports the real fault
            return 0
        upstream = [p for p, outs in adjacency.items() if node_id in outs]
        depth = 1 + max((forward_layer(p, stack + (node_id,)) for p in upstream), default=0)
        layer_of[node_id] = depth
        return depth

    for consumer in sorted(participants):
        for port, raw in sorted((graph_nodes[consumer].get("inputs") or {}).items()):
            if edge_kind(consumer, port) != "episodic":
                continue
            source = _input_source(raw)
            producer = source.partition("/")[0] if source else ""
            if producer not in participants or producer == consumer:
                continue
            if forward_layer(producer) < forward_layer(consumer):
                add(
                    "EPISODIC_NOT_A_BACK_EDGE",
                    {"node": consumer, "input": port, "source": source},
                    f"{consumer}/{port} is declared turn_edge: episodic, but {producer!r} is "
                    f"not downstream of {consumer!r} — an episodic input is resolved from the "
                    "previous turn while this one arrives in the current turn, so the "
                    "participant raises on the first message (ADR-30, issue #220)",
                    f"declare {port!r} forward on the manifest that carries it, and regenerate "
                    "the committed turn plans; keep turn_edge: episodic for reply/verdict/"
                    "result back-edges only",
                )

    if cycle is not None:
        add(
            "CLOCK_CYCLE",
            {"node": cycle[0]},
            f"forward-edge cycle has no episodic break: {' -> '.join(cycle)}",
            "declare a reply/verdict/result/violation back-edge turn_edge: episodic",
        )
    return errors


def _dialogue_blinding_errors(nodes: list[dict]) -> list[dict]:
    """DIALOGUE_GOAL_LEAK (ADR-32 §1): in a T4 graph the policy learns its
    task from dialogue, and the TC-7 goal — which carries the FINAL
    corrected target — may reach only `verifier-*` nodes. A graph is T4
    when any node declares `AISLE_TASK_TIER: T4` (the same graph-attested
    declaration pattern as VAL-8's rung); every such node must consume
    `human_msg`, else the blinding is advisory — the exact failure VAL-6
    exists to prevent for oracle_state."""

    def declared_tier(node: dict) -> str:
        env = node.get("env")
        raw = env.get("AISLE_TASK_TIER", "") if isinstance(env, dict) else ""
        return str(raw).strip().upper()

    t4_nodes = [n for n in nodes if declared_tier(n) == "T4"]
    if not t4_nodes:
        return []
    errors: list[dict] = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        for port, raw in sorted((node.get("inputs") or {}).items()):
            source = _input_source(raw)
            if source is None or "/" not in source:
                continue
            if source.split("/", 1)[1] == "episode_goal" and not node_id.startswith("verifier-"):
                errors.append(
                    _entry(
                        "DIALOGUE_GOAL_LEAK",
                        {"node": node_id, "input": port, "source": source},
                        f"episode_goal consumed by {node_id!r} in a T4 graph — the goal "
                        "carries the final corrected target; the policy must learn its "
                        "task from dialogue (ADR-32 §1)",
                        "route episode_goal to verifier-* nodes only; feed the task "
                        "state machine from human-sim/human_msg instead",
                    )
                )
    for node in t4_nodes:
        sources = [_input_source(raw) for raw in (node.get("inputs") or {}).values()]
        if not any(s is not None and s.endswith("/human_msg") for s in sources):
            node_id = str(node.get("id", ""))
            errors.append(
                _entry(
                    "DIALOGUE_GOAL_LEAK",
                    {"node": node_id},
                    f"{node_id!r} declares AISLE_TASK_TIER T4 but consumes no "
                    "human_msg — a T4 policy with no dialogue input cannot learn "
                    "its task (ADR-32 §1)",
                    "wire human-sim/human_msg into the T4 node's inputs",
                )
            )
    return errors


def _guard_resolved(out_port: str, graph_nodes: dict, manifests: dict) -> bool:
    """The guard hop counts only when fully resolved: a budget-guard graph
    node AND manifest exist, and the referenced output is declared by both
    — a manifest alone (or a phantom output) is not a gate."""
    node = graph_nodes.get(GUARD_ID)
    manifest = manifests.get(GUARD_ID)
    return (
        node is not None
        and manifest is not None
        and out_port in (node.get("outputs") or [])
        and out_port in (manifest.get("outputs") or {})
    )


def _gated_source(
    source: str | None, graph_nodes: dict, manifests: dict, memo: dict, stack: set
) -> bool:
    """VAL-5 traversal semantics: True iff EVERY backward path from this
    source reaches the fully resolved budget-guard before terminating at a
    root, timer, or unresolvable source. Conservative dataflow assumption:
    all of a node's inputs feed its outputs, so one unguarded input taints
    the node; cycles without a guard on them are ungated."""
    if source is None:
        return False
    src_id, _, out_port = source.partition("/")
    if src_id == GUARD_ID:
        return _guard_resolved(out_port, graph_nodes, manifests)
    if src_id in memo:
        return memo[src_id]
    if src_id in stack:
        return False
    node = graph_nodes.get(src_id)
    if node is None or not (node.get("inputs") or {}):
        memo[src_id] = False  # unresolvable source or root: path ends unguarded
        return False
    stack.add(src_id)
    result = all(
        _gated_source(upstream, graph_nodes, manifests, memo, stack)
        for upstream in _backward_sources(node)
    )
    stack.discard(src_id)
    memo[src_id] = result
    return result


def load_graph(path: Path, graph_snapshot: bytes | None = None) -> tuple[list | None, list[dict]]:
    """Parse a graph path or captured bytes; return ``(nodes, errors)``.

    ``path`` remains the graph's provenance even when ``graph_snapshot`` is
    supplied. In particular, validate() uses its parent as VAL-2's one and
    only relative-path base; the snapshot is content, not a staged identity.
    """
    where = {"node": str(path)}

    def invalid(detail: str, hint: str) -> tuple[None, list[dict]]:
        return None, [_entry("GRAPH_INVALID", where, detail, hint)]

    try:
        # encoding pinned: graphs carry em dashes in their header comments, so
        # the locale default (LC_ALL=C in a minimal container or cron) turned
        # every graph into "cannot read graph: 'ascii' codec can't decode"
        text = (
            graph_snapshot.decode("utf-8")
            if graph_snapshot is not None
            else path.read_text(encoding="utf-8")
        )
        data = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError) as exc:
        return invalid(f"cannot read graph: {exc}", "pass a readable UTF-8 dataflow YAML path")
    except yaml.YAMLError as exc:
        return invalid(f"unparseable YAML: {exc}", "fix the YAML syntax; see graphs/ for examples")
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
        return invalid(
            "graph must be a mapping with a `nodes` list of mappings",
            "structure the file as {nodes: [{id, inputs, outputs}, ...]}",
        )
    if not nodes:
        return invalid(
            "the `nodes` list is empty",
            "add at least one node; a graph that runs nothing never validates",
        )
    structural: list[dict] = []
    for index, node in enumerate(nodes):
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            structural.append(
                _entry(
                    "GRAPH_INVALID",
                    {"node": f"nodes[{index}]"},
                    f"node id must be a non-empty string, got {node_id!r}",
                    "give every node a string id matching a registry manifest",
                )
            )
        inputs = node.get("inputs")
        if inputs is not None and not isinstance(inputs, dict):
            structural.append(
                _entry(
                    "GRAPH_INVALID",
                    {"node": str(node_id)},
                    f"inputs must be a mapping of port -> source, got {type(inputs).__name__}",
                    "write inputs as {port: producer-id/output}",
                )
            )
        outputs = node.get("outputs")
        bad_outputs = not isinstance(outputs, list) or not all(
            isinstance(o, str) and o for o in outputs
        )
        if outputs is not None and bad_outputs:
            structural.append(
                _entry(
                    "GRAPH_INVALID",
                    {"node": str(node_id)},
                    f"outputs must be a list of non-empty strings, got {outputs!r}",
                    "write outputs as a YAML list of output port names",
                )
            )
        elif outputs is not None and len(set(outputs)) != len(outputs):
            duplicates = sorted({o for o in outputs if outputs.count(o) > 1})
            structural.append(
                _entry(
                    "GRAPH_INVALID",
                    {"node": str(node_id)},
                    f"duplicate output ports {duplicates} — each output may appear once",
                    "remove the repeated entries from the outputs list",
                )
            )
        env = node.get("env")
        if env is not None and not isinstance(env, dict):
            # PR #80 re-review: downstream checks read env with .get(); a
            # scalar/list here must be a structural refusal, not a crash
            structural.append(
                _entry(
                    "GRAPH_INVALID",
                    {"node": str(node_id)},
                    f"env must be a mapping of variable -> value, got {type(env).__name__}",
                    "write env as {VAR: value} pairs",
                )
            )
    if structural:
        return None, structural
    return nodes, []


def _manifest_launchable(
    manifest: dict,
    arm_kind: str,
    root: Path | None = None,
    embodiment: str | None = None,
    allow_unproven: bool = False,
) -> bool:
    """A candidate INSTALL_MISSING alternative must survive EVERY
    manifest-level check of the NEXT compile (VAL-2's never-fail-the-
    next-compile MUST; PR #63/#64 reviews): an installed distribution —
    or a path source that is a real file under the root (else
    SOURCE_INVALID) — AND arm-compatible AND base-compatible (a
    base-requiring peer on a fixed-base graph is EMBODIMENT_MISMATCH)
    AND, for motion manifests, evalcarded unless the caller allows
    unproven motion (else EVAL_MISSING_FOR_MOTION)."""
    dist = _pip_dist(manifest)
    if dist is not None and not _pip_installed(dist):
        return False
    source = manifest.get("source")
    if (
        dist is None
        and root is not None
        and isinstance(source, str)
        and ":" not in source
        and not _path_source_valid(source, root)
    ):
        return False
    arms = manifest.get("embodiment", {}).get("arm", [])
    if arms and arm_kind not in arms:
        return False
    base = manifest.get("embodiment", {}).get("base", [])
    if base and embodiment is not None and embodiment not in base:
        return False
    if (
        manifest.get("safety_class") == "motion"
        and manifest.get("eval") is None
        and not allow_unproven
    ):
        return False
    return True


def validate_nodes(
    nodes: list[dict],
    manifests: dict[str, dict],
    vocabulary: set[str],
    embodiment: str,
    allow_unproven: bool,
    graph_dir: Path | None = None,
    root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    errors: list[dict] = []
    warnings: list[dict] = []
    for node_id, count in sorted(Counter(n["id"] for n in nodes).items()):
        if count > 1:
            errors.append(
                _entry(
                    "NODE_ID_DUPLICATE",
                    {"node": node_id},
                    f"node id {node_id!r} appears {count} times",
                    "give each node instance a unique id matching one manifest",
                )
            )

    graph_nodes = {n["id"]: n for n in nodes}
    # MOB-4: an embodiment resolves to an ARM kind; `mobile` runs the
    # franka arm on a base, so a franka-arm graph validates unchanged
    # under `mobile`. Arm nodes are checked against the resolved arm.
    arm_kind = EMBODIMENT_ARM.get(embodiment, embodiment)
    # TC-9/VAL-8: the perception rung is a property of the GRAPH, read once
    rung, bridge_ids, rung_errors = graph_perception_rung(nodes, manifests)
    errors.extend(rung_errors)
    # ADR-67: the engine is declared on the bridge like the rung, so a typo
    # must be caught here rather than at launch
    errors.extend(graph_sim_engine_errors(nodes, bridge_ids))
    # ADR-32 §1: T4 goal blinding is likewise a property of the graph
    errors.extend(_dialogue_blinding_errors(nodes))
    for node in nodes:
        node_id = node["id"]
        # ADR-25 (issue #71, PR #80 review): the bridge's reset-less
        # bring-up opt-out must never reach a validated graph — the rollout
        # runner scrubs it from os.environ, but node env in graph YAML
        # bypasses that scrub and would restore the pre-reset startup race
        # in a measured rollout. Fail closed on the truthy spellings the
        # bridge itself accepts.
        bringup = str((node.get("env") or {}).get("AISLE_STEP_WITHOUT_RESET", ""))
        if bringup.strip().lower() in ("1", "true", "yes"):
            errors.append(
                _entry(
                    "BRINGUP_ENV_FORBIDDEN",
                    {"node": node_id},
                    f"node {node_id!r} sets AISLE_STEP_WITHOUT_RESET={bringup!r} in graph env",
                    "remove it: pre-reset free-running is a bring-up mode for "
                    "reset-less debug graphs run directly via `dora run`, never "
                    "for validated rollouts (CON-5, ADR-25)",
                )
            )
        manifest = manifests.get(node_id)
        if manifest is None:
            # fleet stamping (design doc 8.4.3): `<id>-a<int>` is agent
            # <int>'s instance of manifest `<id>` — same schema, same
            # capabilities, one env pin apart
            stamped = re.fullmatch(r"(?P<base>.+)-a\d+", node_id)
            if stamped:
                manifest = manifests.get(stamped.group("base"))
        if manifest is None:
            close = _closest(node_id, list(manifests))
            suggestion = (
                f"rename the node to {close!r}"
                if close
                else "no similar manifest id exists; find one with: "
                "python -m aisle.harness.registry search --provides <capability> "
                "--installed (issue #39: only launchable nodes)"
            )
            errors.append(
                _entry(
                    "MANIFEST_MISSING",
                    {"node": node_id},
                    f"no manifest for node id {node_id!r}",
                    suggestion,
                )
            )
            # VAL-6 is manifest-based: a node WITHOUT a manifest is never an
            # authorized verifier, so oracle consumption must still surface
            # and not hide behind MANIFEST_MISSING.
            for port, raw in (node.get("inputs") or {}).items():
                source = _input_source(raw)
                if source is None:
                    continue
                if source.endswith("/oracle_state"):
                    errors.append(
                        _entry(
                            "ORACLE_LEAK",
                            {"edge": f"{source} -> {node_id}/{port}"},
                            f"oracle_state consumed by {node_id!r}, which has no "
                            "manifest and so cannot be an authorized verifier (VAL-6)",
                            "only verifier-* manifests may read ground truth",
                        )
                    )
                # VAL-8 needs the same fallback for the same reason: the rung
                # binds the GRAPH, not the registry, so an unregistered node —
                # a hand-added one, or anything a harness injects — must not
                # be exempt from it. Measured before this was added: at L1 an
                # unregistered consumer of `poses` reported MANIFEST_MISSING
                # alone, where the byte-identical oracle_state graph reported
                # MANIFEST_MISSING + ORACLE_LEAK.
                producer, _, out_port = source.partition("/")
                if producer in bridge_ids and out_port in FORBIDDEN_BY_RUNG.get(rung, ()):
                    errors.append(
                        _rung_entry(
                            {"edge": f"{source} -> {node_id}/{port}"}, out_port, node_id, rung
                        )
                    )
            continue
        # VAL-2 SOURCE_INVALID (issue #35): unknown schemes and empty pip
        # dists are CAP-1 pattern violations (schema/lint + the registry
        # screen above), so the remaining dodge channel a schema cannot
        # check is a path-form source naming NO FILE under the root — the
        # manifest's launch claim is unverifiable: error out and skip the
        # source-derived checks for this node.
        source_val = manifest.get("source")
        source_invalid = False
        if isinstance(source_val, str) and root is not None:
            if ":" not in source_val and not _path_source_valid(source_val, root):
                source_invalid = True
                errors.append(
                    _entry(
                        "SOURCE_INVALID",
                        {"node": node_id},
                        f"manifest source {source_val!r} is not a regular "
                        "file contained by the root — the graph would "
                        "validate but never launch (or launch code outside "
                        "the vetted tree)",
                        "point the manifest source at the node's real file "
                        "(or register the node with harness skill register)",
                    )
                )
        # VAL-2 INSTALL_MISSING (H1-discovered): a pip:-sourced capability
        # that is not installed validates into a graph that cannot launch —
        # the agent's only signal is this error, so the hint must name an
        # installed same-capability alternative when one exists
        dist = None if source_invalid else _pip_dist(manifest)
        if dist is not None and not _pip_installed(dist):
            provided = set(manifest.get("provides") or [])
            # a usable alternative must FULLY cover the missing node's
            # capabilities — a partial provider is not a replacement
            alternatives = sorted(
                other_id
                for other_id, other in manifests.items()
                if other_id != node_id
                and provided <= set(other.get("provides") or [])
                and _manifest_launchable(other, arm_kind, root, embodiment, allow_unproven)
            )
            if alternatives:
                hint = f"use an installed provider of {sorted(provided)}: " + ", ".join(
                    alternatives
                )
            elif dist:
                hint = (
                    f"install {dist!r} (env-change PR) or author a node "
                    f"providing {sorted(provided)}"
                )
            else:
                # `install ''` is a nonsense instruction (red-team, PR #34)
                hint = (
                    "fix the manifest source: `pip:` carries no distribution "
                    f"name; name one or author a node providing {sorted(provided)}"
                )
            detail = (
                f"manifest source {manifest.get('source')!r}: distribution "
                + (f"{dist!r} is not installed in this environment" if dist else "name is empty")
                + " — the graph would validate but never launch"
            )
            errors.append(_entry("INSTALL_MISSING", {"node": node_id}, detail, hint))
        # VAL-2 PATH_MANIFEST_MISMATCH (issue #36; H3 campaign 2 live case):
        # dora launches the graph node's `path`, not the manifest's
        # `source` — an approved id with a divergent path executes unvetted
        # code under a vetted identity, and every topology check passes
        node_path = node.get("path")
        source = manifest.get("source")
        if (
            isinstance(node_path, str)
            and isinstance(source, str)
            and not source_invalid
            and graph_dir is not None
            and root is not None
        ):
            pip_name = _pip_dist(manifest)
            if pip_name is not None:
                # graphs reference pip nodes as the manifest source
                # verbatim or the bare NORMALIZED distribution name (PR
                # #62 review P2: reuse _pip_dist so decorated/case-varied
                # sources match the INSTALL_MISSING contract) — anything
                # else launches other code
                canonical_path = re.sub(r"[-_.]+", "-", node_path.strip()).lower()
                matches = node_path == source or canonical_path == pip_name
            else:
                # exactly ONE base: the graph's own directory — the base
                # dora resolves against. A graphs-dir fallback approved
                # tmpdir-staged graphs whose paths resolve elsewhere at
                # runtime (PR #62 review P1: the live-swap bypass);
                # staged copies must carry absolute paths instead.
                matches = (graph_dir / node_path).resolve() == (root / source).resolve()
            if not matches:
                errors.append(
                    _entry(
                        "PATH_MANIFEST_MISMATCH",
                        {"node": node_id},
                        f"path {node_path!r} does not resolve to the manifest "
                        f"source {source!r} — dora would launch code the "
                        f"registry never vetted under the id {node_id!r}",
                        f"point path at the manifest source ({source!r}), or "
                        "register a manifest for what path actually launches "
                        "(harness skill register)",
                    )
                )
        # VAL-4: every schema name a graph node's manifest references must be
        # in the vocabulary — including unwired ports; never silently passed
        for direction in ("inputs", "outputs"):
            for port, spec in (manifest.get(direction) or {}).items():
                schema = spec.get("schema") if isinstance(spec, dict) else None
                if schema is not None and schema not in vocabulary:
                    errors.append(
                        _entry(
                            "SCHEMA_UNKNOWN",
                            {"node": node_id},
                            f"{direction}/{port}: schema {schema!r} is not in "
                            "registry/schema/schemas.toml",
                            "add it via a Class C schema-vocabulary change (CAP-2) or fix the name",
                        )
                    )
        # every graph-declared output must exist in the manifest, consumed
        # or not — the graph cannot invent ports the typed contract lacks
        manifest_outputs = manifest.get("outputs") or {}
        for out in node.get("outputs") or []:
            if out not in manifest_outputs:
                errors.append(
                    _entry(
                        "SCHEMA_MISMATCH",
                        {"node": node_id},
                        f"graph declares output {out!r} but {node_id}'s manifest does not",
                        f"use one of the manifest outputs {sorted(manifest_outputs)}, "
                        "or extend the manifest (Class B change)",
                    )
                )
        arms = manifest.get("embodiment", {}).get("arm", [])
        if arms and arm_kind not in arms:
            errors.append(
                _entry(
                    "EMBODIMENT_MISMATCH",
                    {"node": node_id},
                    f"{node_id} supports arms {arms}, graph targets "
                    f"{embodiment!r} (arm {arm_kind!r})",
                    f"swap in a capability supporting {arm_kind!r} or change --embodiment",
                )
            )
        # MOB-4: a base-requiring node lists the base-providing embodiments
        # it needs; on a fixed-base graph (no base) that is a mismatch.
        base = manifest.get("embodiment", {}).get("base", [])
        if base and embodiment not in base:
            errors.append(
                _entry(
                    "EMBODIMENT_MISMATCH",
                    {"node": node_id},
                    f"{node_id} requires a base profile {base}, graph targets {embodiment!r}",
                    f"target one of {base} (a mobile base profile), "
                    "or drop the base-requiring node",
                )
            )
        if manifest.get("safety_class") == "motion" and manifest.get("eval") is None:
            entry = _entry(
                "EVAL_MISSING_FOR_MOTION",
                {"node": node_id},
                f"{node_id} is safety_class=motion with no evalcard (CAP-6)",
                "attach an evalcard from its eval suite before motion use",
            )
            (warnings if allow_unproven else errors).append(entry)

        # SPEC 210 MOB-3: on a mobile graph the guard (it outputs
        # base_cmd_safe) MUST also wire base_pose (keep-out feedback AND the
        # sim-time watchdog's clock, ADR-29) and tick (BG-5 stats AND the
        # watchdog's fail-closed wall-net sweep) — the validator otherwise
        # does not require every manifest input. Port NAMES are not enough
        # (PR #156 review): both inputs are now the watchdog's clocks, so
        # their SOURCES are checked too — tick must be a real dora timer at
        # a bounded period (a never-firing or slow source silently disables
        # the wall net), and base_pose must come from a sim_bridge node (an
        # arbitrary producer could feed forged or absent stamps and defeat
        # staleness AND keep-out at once).
        if embodiment == "mobile" and "base_cmd_safe" in (manifest.get("outputs") or {}):
            guard_inputs = node.get("inputs") or {}
            missing = {"base_pose", "tick"} - set(guard_inputs)
            if missing:
                errors.append(
                    _entry(
                        "MOBILE_GUARD_INCOMPLETE",
                        {"node": node_id},
                        f"{node_id} guards the base on a mobile graph but does not "
                        f"wire {sorted(missing)}",
                        "wire base_pose (MOB-3 keep-out + the watchdog's sim clock) "
                        "and tick (BG-5 stats + the ADR-29 wall-net sweep) into the guard",
                    )
                )
            if "tick" in guard_inputs:
                tick_src = _input_source(guard_inputs["tick"])
                # the shared timer parser, not an ad-hoc regex (issue #160
                # item 3): the regex accepted millis/0, which _parse_timer_hz
                # rejects everywhere else — a zero-period timer is not a
                # bounded sweep cadence, it is malformed
                tick_hz = _parse_timer_hz(str(tick_src or ""))
                if tick_hz is None or tick_hz < 1000.0 / GUARD_TICK_MAX_MS:
                    errors.append(
                        _entry(
                            "MOBILE_GUARD_INCOMPLETE",
                            {"node": node_id},
                            f"{node_id} tick is {tick_src!r}, not a dora timer at "
                            f"<= {GUARD_TICK_MAX_MS} ms",
                            "wire tick from dora/timer/millis/<N> with N <= "
                            f"{GUARD_TICK_MAX_MS} so the ADR-29 wall-net sweep "
                            "actually runs at a bounded cadence",
                        )
                    )
            if "base_pose" in guard_inputs:
                pose_src = _input_source(guard_inputs["base_pose"])
                producer = str(pose_src or "").partition("/")[0]
                if producer not in bridge_ids:
                    errors.append(
                        _entry(
                            "MOBILE_GUARD_INCOMPLETE",
                            {"node": node_id},
                            f"{node_id} base_pose comes from {producer!r}, which does "
                            "not provide sim_bridge",
                            "wire base_pose from the sim bridge — the watchdog's "
                            "staleness clock and the keep-out geometry both trust "
                            "its stamps (ADR-29)",
                        )
                    )

        for port, source in (node.get("inputs") or {}).items():
            _validate_edge(
                node,
                manifest,
                port,
                source,
                graph_nodes,
                manifests,
                vocabulary,
                errors,
                warnings,
                rung,
                bridge_ids,
            )
    errors += _refusal_routing_errors(nodes)
    # ADR-30/VAL-2: every otherwise-valid simulator graph must prove its
    # closed-turn topology. Graphs already refused for another concern do not
    # get secondary CLOCK noise; once that concern is fixed, CLOCK failures
    # surface before the graph can validate or run. REFUSAL_UNROUTED is
    # deliberately ABOVE this: an unroutable refusal is the more specific
    # diagnosis, and the two would otherwise both fire on the same graph.
    if not errors:
        errors.extend(_clock_errors(nodes, manifests))
        errors.extend(_turn_plan_errors(nodes, manifests, graph_dir))
    return errors, warnings


def _turn_plan_errors(nodes: list[dict], manifests: dict, graph_dir: Path | None) -> list[dict]:
    """ADR-30 / issue #213: the COMMITTED turn plan must match this graph.

    `graphs/turn_plans/<stem>.json` is not documentation — the barrier
    (`nodes/turn_barrier.py::load_plan`) reads the committed file at runtime
    and `turns.py::_check_output_set` refuses any participant whose
    watermark output set disagrees with it. The plan is also FROZEN
    (`tools/env_hash.py`) precisely because manifests are not: compiling it
    at run time instead would let an unfrozen manifest edit change the
    scheduler topology without moving `env_hash`. So the artifact has to
    stay committed, and staleness has to be caught somewhere else.

    Without this rule it is caught nowhere. A graph edit that skips the
    regeneration passes validate and `env_hash` (which hashes the plan as an
    INPUT, not as something derived), then kills the barrier on the first
    watermark — visible only as every episode hitting the ADR-23 wall clamp,
    relaunching, and the run reporting `pass1: 0.0` about twenty minutes
    later, with the actual ProtocolError buried in the barrier's per-node
    dora log. Measured in the round-2 review of #208.

    Runs only on an otherwise-valid graph (the caller's `if not errors`
    gate), so `compile_turn_plan` cannot be handed a malformed one."""
    errors: list[dict] = []
    for node in nodes:
        env = node.get("env")
        raw = str((env or {}).get("AISLE_TURN_PLAN", "")).strip() if isinstance(env, dict) else ""
        if not raw or not raw.endswith(".json"):
            continue  # inline JSON or no plan: the barrier owns those verdicts
        node_id = str(node.get("id", ""))
        # Only an AUTHORED graph — one sitting beside its own plan — is
        # comparable. A RELOCATED copy (an instrumented run graph, or the
        # staged copy `harness swap` validates) legitimately differs from the
        # committed plan: instrumentation adds the trace recorder, and A7
        # rewrites the topology and compiles its own plan into the run dir.
        # Comparing those against the authored plan would refuse correct
        # graphs. The barrier resolves a missing relative path against
        # <repo>/graphs, so a relocated copy still runs the committed plan —
        # which the authored graph's own validation already covers.
        path = Path(raw)
        if not path.is_file():
            if graph_dir is None:
                continue
            path = Path(graph_dir) / raw
            if not path.exists():
                continue  # not beside the graph: see above
        where = {"node": node_id, "plan": raw}
        try:
            committed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                _entry(
                    "TURN_PLAN_STALE",
                    where,
                    f"cannot read the committed turn plan {raw!r}: {exc}",
                    "regenerate it with compile_turn_plan(graph_nodes, manifests); the "
                    "barrier loads this file at runtime",
                )
            )
            continue
        expected = compile_turn_plan(nodes, manifests)
        if committed != expected:
            differing = sorted(
                k for k in set(committed) | set(expected) if committed.get(k) != expected.get(k)
            ) or ["(identical keys, different nesting)"]
            errors.append(
                _entry(
                    "TURN_PLAN_STALE",
                    where,
                    f"the committed turn plan {raw!r} does not match this graph; "
                    f"differing sections: {differing}",
                    "regenerate it with compile_turn_plan(graph_nodes, manifests) — the "
                    "barrier loads the COMMITTED file, so a stale one kills it on the "
                    "first watermark and every episode wall-clamps (issue #213)",
                )
            )
    return errors


def _refusal_routing_errors(nodes: list[dict]) -> list[dict]:
    """TC-6/ADR-34: a graph carrying a reset service MUST route refusals.

    dora DROPS a `send_output` to an output the graph does not declare — one
    warning to the node's stderr, `Ok(())` to the caller (dora node
    `validate_output`). So a graph whose reset node omits `reset_refused`
    does not fail loudly; the refusal evaporates and the requester waits in
    `awaiting_reset` until the wall clamp. That is strictly worse than the
    silent degradation this split removed, and it is invisible to every
    other rule here: VAL-4 checks that declared outputs exist in the
    manifest, never that a manifest output is declared.

    Two halves, because either alone still hangs the run: the service must
    DECLARE the output, and something must CONSUME it (round-2 review of
    #208)."""
    errors: list[dict] = []
    reset_nodes = [
        n
        for n in nodes
        if isinstance(n, dict)
        and str(n.get("path") or "").replace("\\", "/").endswith(RESET_SERVICE_SOURCE)
    ]
    for node in reset_nodes:
        node_id = str(node.get("id", ""))
        outputs = node.get("outputs")
        if "reset_refused" not in (outputs if isinstance(outputs, list) else []):
            errors.append(
                _entry(
                    "REFUSAL_UNROUTED",
                    {"node": node_id, "output": "reset_refused"},
                    f"{node_id!r} runs the reset service but declares no `reset_refused` "
                    "output; dora would DROP every refusal with a stderr warning and the "
                    "requester would wait forever (TC-6, ADR-34)",
                    "add `reset_refused` to the node's outputs and wire it to the node "
                    "that sends `reset` (see graphs/expert_t0.yaml)",
                )
            )
            continue
        consumers = [
            other["id"]
            for other in nodes
            if isinstance(other, dict)
            and isinstance(other.get("inputs"), dict)
            and any(
                (_input_source(raw) or "") == f"{node_id}/reset_refused"
                for raw in other["inputs"].values()
            )
        ]
        if not consumers:
            errors.append(
                _entry(
                    "REFUSAL_UNROUTED",
                    {"node": node_id, "output": "reset_refused"},
                    f"{node_id}/reset_refused reaches no consumer; a refused reset would "
                    "be published to nobody and the requester would wait forever "
                    "(TC-6, ADR-34)",
                    "wire `reset_refused` into the node that sends `reset`",
                )
            )
    return errors


#: What a reset node must be wired for to actually PERFORM a behavioral
#: reset (RST-2): the perception it plans from, the joint feedback it
#: closes the loop on, and the command topics it drives the arm with.
#: A node without these still ACCEPTS mode 1 -- it just burns every attempt
#: on `latest_rgb is None` and falls back to teleport (issue #196).
BEHAVIORAL_RESET_INPUTS = ("bridge_info", "rgb_overhead", "depth_overhead", "joint_state")
BEHAVIORAL_RESET_OUTPUTS = ("reset_joint_cmd", "reset_gripper_cmd")
#: The reset service is identified by the SOURCE it runs, not by node id:
#: `graphs/` happens to id it `reset` but `tests/conftest.py` ids the same
#: node `reset-service`, and the repo's other cross-node check
#: (test_episode_boundary_wiring._node_sources) already resolves it by path.
RESET_SERVICE_SOURCE = "reset/service.py"


def behavioral_reset_gaps(nodes: list) -> list[str]:
    """RST-2 (issue #196): what this graph is missing to serve a behavioral
    reset, or [] if it can.

    Most graphs wire the reset node with `{reset, reset_done}` only.
    `--reset behavioral` on one of them is accepted, pays a ~2 s model load
    inside the first reset request (the `bridge_info` pre-warm is
    unreachable without that input), exhausts all three attempts on a
    missing frame, and falls back to teleport -- reporting
    `fallback: true, behavioral_attempts: 3` every episode. An A6 ablation
    run that way is a teleport run that measures nothing, with no error
    anywhere.

    Declaring the ports is not the same as being wired for them, so the
    command outputs are checked for a CONSUMER too: the reset has no
    private channel to the arm (VAL-5), so `reset_joint_cmd` that reaches
    no budget guard is exactly as inert as one that was never declared,
    and `validate` has no rule against an unconsumed output. The input
    side needs no such check -- VAL-2/VAL-3 already refuse an edge whose
    source does not resolve, and this runs after that gate.

    Pure so the refusal is unit-testable without a rollout. Tolerant of
    malformed shapes rather than raising: `load_graph` owns those verdicts
    (GRAPH_INVALID), and a TypeError out of here would leave the CLI with
    no JSON to print (CON-8)."""
    nodes = [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []
    resets = [
        n
        for n in nodes
        if str(n.get("path") or "").replace("\\", "/").endswith(RESET_SERVICE_SOURCE)
    ] or [n for n in nodes if n.get("id") == "reset"]
    if not resets:
        return [f"no node running {RESET_SERVICE_SOURCE} in the graph"]
    # a graph with more than one is served if ANY of them can serve it
    return min((_reset_node_gaps(n, nodes) for n in resets), key=len)


def _reset_node_gaps(node: dict, nodes: list) -> list[str]:
    """`behavioral_reset_gaps` for one candidate reset node."""
    inputs = node.get("inputs")
    have_in = set(inputs) if isinstance(inputs, dict) else set()
    outputs = node.get("outputs")
    have_out = {o for o in outputs if isinstance(o, str)} if isinstance(outputs, list) else set()
    consumed = set()
    for other in nodes:
        if other is node or not isinstance(other.get("inputs"), dict):
            continue
        for raw in other["inputs"].values():
            source = _input_source(raw) or ""
            producer, _, topic = source.partition("/")
            if topic and producer == node.get("id"):
                consumed.add(topic)
    gaps = [f"input {name!r}" for name in BEHAVIORAL_RESET_INPUTS if name not in have_in]
    gaps += [f"output {name!r}" for name in BEHAVIORAL_RESET_OUTPUTS if name not in have_out]
    gaps += [
        f"output {name!r} reaches no consumer"
        for name in BEHAVIORAL_RESET_OUTPUTS
        if name in have_out and name not in consumed
    ]
    return gaps


def graph_perception_rung(nodes: list, manifests: dict) -> tuple[str, list[str], list[dict]]:
    """The graph's perception rung, sim-bridge ids, and VAL-8 errors (TC-9).

    The rung rides the GRAPH so the graph hash attests which pose source a
    result used — the same reasoning as ADR-25's bring-up scrub and ADR-11
    clause 14's capture declaration. A graph that declares nothing is L0,
    which is what every pre-TC-9 graph is: the check below then permits
    `poses` exactly as before.

    Two ways of reading this wrong both END IN A SILENT PASS, because an
    unrecognized rung makes FORBIDDEN_BY_RUNG.get(...) empty and forbids
    nothing, so both are errors rather than best-effort guesses:

    * a value this table does not know (`L3`, a stray space in `"L1 "`).
      Measured before this was fixed: `AISLE_PERCEPTION: "L1 "` on a graph
      wiring ground-truth `poses` validated ok=true, exit 0. The bridge
      normalizes with .strip().upper() and REFUSES an unknown rung, so the
      un-stripped read also disagreed with the runtime about the same text.
    * two bridge processes with DIFFERENT effective rungs. Node order in a
      dataflow YAML is arbitrary, and an omitted key defaults that bridge to
      L0 at runtime, so neither first-wins nor applying one explicit value
      graph-wide describes what the processes actually run.
    """
    errors: list[dict] = []
    declared = {}
    for node in nodes:
        env = node.get("env") or {}
        if "AISLE_PERCEPTION" not in env:
            continue
        # a PRESENT but blank value is an unknown rung, not an absent one:
        # skipping it fell through to the L0 default and permitted ground
        # truth, so `AISLE_PERCEPTION: "   "` validated ok=true. Membership
        # rather than `is None`, because `AISLE_PERCEPTION:` with no value
        # parses from YAML as None and is still a graph DECLARING a rung.
        declared[node["id"]] = str(env.get("AISLE_PERCEPTION") or "").strip().upper()
    # TC-9: the rung is declared on the SIM BRIDGE, because dora injects a
    # node's env into that node's process alone. A rung on any other node is
    # read by nobody at runtime: the validator would forbid `poses` graph-wide
    # while the bridge stayed at its default, published ground truth and
    # rendered no segmentation — an L1 run starved of every pose source.
    bridge_ids = sorted(
        node["id"]
        for node in nodes
        if "sim_bridge" in ((manifests.get(node["id"]) or {}).get("provides") or [])
    )
    for node_id in sorted(set(declared) - set(bridge_ids)):
        errors.append(
            _entry(
                "PERCEPTION_RUNG_VIOLATION",
                {"node": node_id},
                f"{node_id!r} declares AISLE_PERCEPTION but does not provide sim_bridge (VAL-8)",
                (
                    f"move the declaration to {bridge_ids[0]!r}"
                    if bridge_ids
                    else "declare the rung on the sim_bridge node; this graph has none"
                )
                + " — dora passes env to that node's process only, so a rung declared "
                "elsewhere never reaches the bridge that must act on it",
            )
        )
        declared.pop(node_id)
    unknown = {n: r for n, r in declared.items() if r not in FORBIDDEN_BY_RUNG}
    for node_id, rung in sorted(unknown.items()):
        errors.append(
            _entry(
                "PERCEPTION_RUNG_VIOLATION",
                {"node": node_id},
                f"unknown perception rung {rung!r} declared by {node_id!r} (VAL-8)",
                f"set AISLE_PERCEPTION to one of {sorted(FORBIDDEN_BY_RUNG)} — an "
                "unrecognized rung would forbid nothing and silently pass the check",
            )
        )
    # Every bridge has an EFFECTIVE runtime rung: an omitted key is L0, not a
    # vote of abstention. This matters for the future multi-bridge case — one
    # explicit L1 bridge plus one omitted/default-L0 bridge runs two different
    # contracts and cannot be represented by one graph-level attestation.
    effective = {bridge_id: declared.get(bridge_id, "L0") for bridge_id in bridge_ids}
    distinct = sorted(set(effective.values()))
    if len(distinct) > 1:
        errors.append(
            _entry(
                "PERCEPTION_RUNG_VIOLATION",
                {"node": sorted(effective)[0]},
                f"conflicting perception rungs {distinct} across sim bridges "
                f"{sorted(effective)} (VAL-8)",
                "declare the same AISLE_PERCEPTION on every sim-bridge node — an "
                "omitted key defaults that process to L0, and a graph with two "
                "effective rungs attests neither",
            )
        )
    # on any bad declaration, enforce the STRICTEST rung the table has rather
    # than the most permissive: the graph is already rejected, and a forbidden
    # edge in it should still be named in the same report instead of surfacing
    # only after the declaration is fixed
    if errors:
        strictest = max(FORBIDDEN_BY_RUNG, key=lambda r: len(FORBIDDEN_BY_RUNG[r]))
        return strictest, bridge_ids, errors
    rung = distinct[0] if distinct else "L0"
    graph_nodes = {node["id"]: node for node in nodes}
    for bridge_id in bridge_ids:
        bridge_env = graph_nodes[bridge_id].get("env") or {}
        scene = bridge_env.get("AISLE_SCENE", "pharmacy")
        # Env values are not structurally restricted to strings. Do not let a
        # malformed list/map turn this CON-8 JSON diagnostic into a TypeError;
        # the bridge only recognizes the exact string ``store`` too.
        scene_key = scene if isinstance(scene, str) else ""
        if rung in UNSUPPORTED_RUNGS_BY_SCENE.get(scene_key, ()):
            errors.append(
                _entry(
                    "PERCEPTION_RUNG_VIOLATION",
                    {"node": bridge_id},
                    f"perception rung {rung} is not supported for AISLE_SCENE={scene!r} "
                    f"on bridge {bridge_id!r} (VAL-8)",
                    "use AISLE_PERCEPTION=L0 for the store's supported pose path, or "
                    "teach the estimated-pose consumer to query the store namespace "
                    "before selecting L1/L2",
                )
            )
    return rung, bridge_ids, errors


def graph_sim_engine_errors(nodes: list, bridge_ids: list[str]) -> list[dict]:
    """VAL-8 companion for ADR-67: a sim bridge declaring AISLE_SIM_ENGINE
    must name an engine the harness knows.

    Deliberately minimal next to the rung: the engine forbids no topics, so
    the only thing a graph can get silently wrong is the name. An unknown one
    refuses at bridge config time hours later, or worse, defaults elsewhere;
    naming it here costs one dictionary lookup."""
    from aisle.sim import ENGINES, normalize_engine

    errors: list[dict] = []
    by_id = {node["id"]: node for node in nodes}
    for bridge_id in sorted(set(bridge_ids)):
        env = by_id.get(bridge_id, {}).get("env") or {}
        if "AISLE_SIM_ENGINE" not in env:
            continue
        declared = env.get("AISLE_SIM_ENGINE")
        try:
            normalize_engine(str(declared or ""))
        except ValueError:
            errors.append(
                _entry(
                    "SIM_ENGINE_UNKNOWN",
                    {"node": bridge_id},
                    f"unknown simulation engine {declared!r} declared by {bridge_id!r} (ADR-67)",
                    f"set AISLE_SIM_ENGINE to one of {list(ENGINES)}, or drop the key to "
                    "let the runner choose",
                )
            )
    return errors


FORBIDDEN_BY_RUNG = {
    "L0": (),
    # TC-9: at L1 pose must be ESTIMATED from segmentation + depth, so the
    # ground-truth shortcut is closed. At L2 segmentation goes too.
    "L1": ("poses",),
    "L2": ("poses", "seg_overhead"),
}

# Issue #130: the bridge already refuses these combinations at config time.
# Keep the same compatibility gate in validation so an author gets an
# actionable error before paying for Genesis startup and a zero-episode run.
# L1's id-map query and L2's detector vocabulary currently use desk med names;
# store graspables use item ids such as ``slot#2`` and ``bin#category``.
UNSUPPORTED_RUNGS_BY_SCENE = {"store": ("L1", "L2")}

# VAL-2/VAL-3: a hint MUST NOT name an alternative that fails the NEXT
# compile. The L1 remedy is segmentation + depth, but seg_overhead is itself
# forbidden at L2, so a single shared hint sent an L2 author to a topic their
# own rung rejects — measured: following it produced a second
# PERCEPTION_RUNG_VIOLATION. The remedy is per-rung for that reason.
RUNG_REMEDY = {
    "L1": "estimate pose from seg_overhead + depth_overhead",
    "L2": (
        "derive identity from rgb_overhead/rgb_wrist; depth_overhead may supply "
        "same-stamp metric geometry — L2 forbids semantic segmentation"
    ),
}


def _rung_entry(edge: dict, out_port: str, node_id: str, rung: str) -> dict:
    """One PERCEPTION_RUNG_VIOLATION, phrased the same wherever it is raised.

    The hint deliberately does NOT offer "or declare L0 to use ground truth".
    SPEC 060's opening line is that these messages are the research agent's
    learning signal, and the rung is self-declared in the graph the agent
    authors — so naming the downgrade would hand it a one-token way to make
    the error disappear without changing what the run actually measures.
    Lowering a rung is a claim about a result, not a lint fix."""
    return _entry(
        "PERCEPTION_RUNG_VIOLATION",
        edge,
        f"{out_port!r} consumed by {node_id!r} under perception rung {rung} (VAL-8)",
        f"at {rung}, {RUNG_REMEDY.get(rung, 'do not consume ground truth')}",
    )


def _parse_timer_hz(source: str) -> float | None:
    """Rate of a well-formed dora/timer/millis/<N> source, else None.

    The `dora` prefix is part of the contract, not decoration: only dora's
    own timer is guaranteed to fire. Without this check a source shaped
    like `some-node/timer/millis/10` parsed as a 100 Hz timer, which let a
    mobile graph wire the guard's wall-net sweep tick (ADR-29) to a node
    that may never emit and still pass MOBILE_GUARD_INCOMPLETE — the
    fail-closed net silently disabled (PR #177 review; the ad-hoc regex
    this helper replaced did anchor the prefix)."""
    parts = source.split("/")
    if len(parts) == 4 and parts[0] == "dora" and parts[1] == "timer" and parts[2] == "millis":
        if parts[3].isdigit() and int(parts[3]) > 0:
            return 1000.0 / int(parts[3])
    return None


def _validate_edge(
    node,
    manifest,
    port,
    source,
    graph_nodes,
    manifests,
    vocabulary,
    errors,
    warnings,
    rung,
    bridge_ids,
) -> None:
    node_id = node["id"]
    # keep the RAW value for the message: VAL-3 makes these hints the
    # research agent's learning signal, and reporting the unwrapped None
    # instead of the offending value ("got 42") tells the author nothing
    # about what they wrote (PR #177 review)
    raw_source = source
    source = _input_source(source)  # dora extended input form {source: ..., queue_size: N}
    if source is None:
        errors.append(
            _entry(
                "GRAPH_INVALID",
                {"edge": f"{node_id}/{port}"},
                f"input source must be a string or {{source: ...}} mapping, got {raw_source!r}",
                "write the source as producer-id/output or dora/timer/millis/<N>",
            )
        )
        return
    edge = {"edge": f"{source} -> {node_id}/{port}"}
    declared_inputs = manifest.get("inputs", {})
    is_dora_source = source.startswith("dora/")
    src_id = None if is_dora_source else source.partition("/")[0]

    # VAL-5 first: every backward path into a motion sink must traverse the
    # RESOLVED budget-guard (topological, per the spec's "every path"; a
    # same-named node with no manifest is spoofing; timers and unresolvable
    # sources are ungated). Never let a later check's early return hide this.
    # The guard itself is motion under the ratified ADR-5 boundary (it emits
    # the *_safe actuation streams) but is exempt as a sink: it IS the gate
    # every path must traverse — raw commands are exactly what it consumes.
    if (
        manifest.get("safety_class") == "motion"
        and port in MOTION_SINK_PORTS
        and node_id != GUARD_ID
    ):
        gated = not is_dora_source and _gated_source(source, graph_nodes, manifests, {}, set())
        if not gated:
            errors.append(
                _entry(
                    "MOTION_UNGATED",
                    edge,
                    f"a path into {node_id}/{port} does not traverse {GUARD_ID} (VAL-5)",
                    f"route every command path through the {GUARD_ID} node (SPEC 080)",
                )
            )

    # fleet fan-in convention (BRG-5, design doc 8.4.3): `name_<int>` is
    # an INSTANCE of declared port `name` — a shared node (guard, reset
    # service) takes one suffixed input per agent, schema unchanged.
    # Dora needs distinct input ids per source; duplicating every
    # instance in the manifest would make the agent count a manifest
    # property, which it is not.
    base_port, _, suffix = port.rpartition("_")
    effective_port = port
    if port not in declared_inputs and suffix.isdigit() and base_port in declared_inputs:
        effective_port = base_port
    port_declared = effective_port in declared_inputs
    if not port_declared:
        errors.append(
            _entry(
                "SCHEMA_MISMATCH",
                edge,
                f"{node_id} has no declared input port {port!r}",
                f"rename the input to one of {node_id}'s declared ports "
                f"{sorted(declared_inputs)}, or extend its manifest (Class B change)",
            )
        )

    if is_dora_source:
        timer_hz = _parse_timer_hz(source)
        if timer_hz is None:
            errors.append(
                _entry(
                    "INPUT_NO_PRODUCER",
                    edge,
                    f"{source!r} is not a valid dora builtin source",
                    "only dora/timer/millis/<N> (N > 0) is supported",
                )
            )
            return
        rate_declared = declared_inputs.get(port, {}).get("rate_hz")
        if rate_declared and abs(timer_hz - rate_declared) > RATE_BAND * rate_declared:
            warnings.append(
                _entry(
                    "RATE_INCOMPATIBLE",
                    edge,
                    f"timer drives {port} at {timer_hz:g} Hz; manifest declares "
                    f"{rate_declared} Hz (±20% band, TC-4)",
                    f"use dora/timer/millis/{round(1000 / rate_declared)}",
                )
            )
        return

    out_port = source.partition("/")[2]
    producer = graph_nodes.get(src_id)
    declared_outputs = (producer or {}).get("outputs") or []
    if producer is None or out_port not in declared_outputs:
        if producer is None:
            close = _closest(src_id, list(graph_nodes))
            hint = (
                f"change the edge source to {close!r} (closest graph node id)"
                if close
                else f"add a node producing {out_port!r} or point the edge at one of "
                f"the graph's nodes: {sorted(graph_nodes)}"
            )
        else:
            hint = (
                f"wire from one of {src_id}'s declared outputs {declared_outputs}, "
                f"or add {out_port!r} to that node's outputs list"
            )
        errors.append(
            _entry(
                "INPUT_NO_PRODUCER",
                edge,
                f"no producer for {source!r}",
                hint,
            )
        )
        return

    # VAL-8 before any schema-level return, for the same reason as VAL-6: a
    # rung violation must not hide behind SCHEMA_* on the same edge. Without
    # this the ladder is advisory — a graph could keep consuming ground-truth
    # pose while its results were reported as L1, which would silently
    # invalidate every L1 number published from it (TC-9).
    # the rung forbids the BRIDGE's ground-truth topics, not every port that
    # happens to share their name: an estimated pose published on a port
    # called `poses` by some other node is exactly what L1 asks for, and
    # name-only matching rejected it (Codex review). TC-9's wording is
    # "the bridge's `poses` topic" — this makes the code say that.
    if src_id in bridge_ids and out_port in FORBIDDEN_BY_RUNG.get(rung, ()):
        errors.append(_rung_entry(edge, out_port, node_id, rung))

    # VAL-6 before any schema-level return: an oracle leak must never be
    # hidden behind SCHEMA_UNKNOWN/SCHEMA_MISMATCH on the same edge.
    if out_port == "oracle_state" and not node_id.startswith("verifier-"):
        errors.append(
            _entry(
                "ORACLE_LEAK",
                edge,
                f"oracle_state consumed by non-verifier node {node_id!r} (VAL-6)",
                # PR #34 red-team: naming pose-estimator here sent agents to
                # an uninstalled pip node — the next compile's INSTALL_MISSING
                "only verifier-* nodes may read ground truth; use an installed "
                "object_pose provider (e.g. oracle-pose) for perception",
            )
        )

    producer_manifest = manifests.get(src_id)
    manifest_outputs = (producer_manifest or {}).get("outputs") or {}
    if producer_manifest is not None and out_port not in manifest_outputs:
        errors.append(
            _entry(
                "INPUT_NO_PRODUCER",
                edge,
                f"{src_id}'s manifest declares no output {out_port!r} — the graph "
                "outputs list cannot invent ports the typed contract lacks",
                f"{src_id} manifest outputs: {sorted(manifest_outputs)}",
            )
        )
        return
    if not port_declared:
        return

    consumer_schema = declared_inputs[effective_port].get("schema")
    producer_schema = (
        manifest_outputs.get(out_port, {}).get("schema") if producer_manifest else None
    )
    # unknown names were already reported per node by the VAL-4 sweep;
    # a mismatch verdict against an unknown name would be noise
    if any(s is not None and s not in vocabulary for s in (producer_schema, consumer_schema)):
        return
    if producer_schema is not None and producer_schema != consumer_schema:
        errors.append(
            _entry(
                "SCHEMA_MISMATCH",
                edge,
                f"{src_id}/{out_port} produces {producer_schema}; "
                f"{node_id}/{port} expects {consumer_schema}",
                f"{src_id}/{out_port} carries {producer_schema}; feed {port} from "
                f"an output with schema {consumer_schema} instead",
            )
        )


def validate(
    graph_path: Path,
    root: Path,
    embodiment: str,
    allow_unproven: bool,
    graph_snapshot: bytes | None = None,
) -> dict:
    report = {"ok": False, "graph": str(graph_path), "errors": [], "warnings": [], "dist_state": {}}
    # ADR-24 D5 (PR #69 review F4): the diagnostic is computed from the
    # REGISTRY alone, before any graph parsing — early graph errors still
    # carry the three registry mappings
    try:
        import importlib.metadata as _md

        for _, m_early in load_manifests(root)[0]:
            dist_early = _pip_dist(m_early) if isinstance(m_early, dict) else None
            if dist_early:
                try:
                    report["dist_state"][dist_early] = _md.version(dist_early)
                except _md.PackageNotFoundError:
                    report["dist_state"][dist_early] = None
        report["dist_state"] = dict(sorted(report["dist_state"].items()))
    except Exception:  # noqa: BLE001 — a broken registry surfaces via its own errors
        pass
    nodes, errors = load_graph(graph_path, graph_snapshot)
    if nodes is None:
        report["errors"] = errors
        return report

    manifest_list, manifest_errors = load_manifests(root)
    if manifest_errors:
        report["errors"] = [
            _entry(
                "GRAPH_INVALID",
                {"node": "(registry)"},
                e["message"],
                "fix the registry before validating graphs (harness/registry.py lint)",
            )
            for e in manifest_errors
        ]
        return report
    try:
        capability_schema = load_capability_schema(root)
        vocabulary = set(load_vocabulary(root))
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        report["errors"] = [
            _entry(
                "GRAPH_INVALID",
                {"node": "(registry)"},
                f"cannot load registry schema files: {exc}",
                "restore registry/schema/ (capability.schema.json, schemas.toml)",
            )
        ]
        return report

    # full CAP-1 schema screen: malformed registry data becomes a structured
    # error before graph validation, never a TypeError mid-check (CON-8).
    # The vocabulary check is deliberately NOT part of this screen — unknown
    # schema NAMES are the validator's own SCHEMA_UNKNOWN concern (VAL-4).
    malformed = [
        _entry(
            "GRAPH_INVALID",
            {"node": "(registry)"},
            f"malformed manifest {path.name}: {message}",
            "fix the registry before validating graphs (harness/registry.py lint)",
        )
        for path, m in manifest_list
        for message in manifest_schema_errors(capability_schema, m)
    ]
    if malformed:
        report["errors"] = malformed
        return report
    manifests = {m["id"]: m for _, m in manifest_list}

    errors, warnings = validate_nodes(
        nodes,
        manifests,
        vocabulary,
        embodiment,
        allow_unproven,
        graph_dir=graph_path.parent,
        root=root,
    )
    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors
    return report


def write_turn_plan(graph_path: Path, root: Path) -> dict:
    """Rewrite a graph's committed ADR-30 turn plan from its own topology.

    Issue #227. The barrier loads the COMMITTED plan at runtime, and #214's
    TURN_PLAN_STALE refuses a graph whose plan disagrees — correctly, since a
    stale plan kills the barrier on the first watermark rather than failing
    at the gate. But `compile_turn_plan` was reachable only from Python and
    `harness/CLAUDE.research.md` never mentioned turn plans, so a research
    agent editing its own `graphs/agent_campaign.yaml` had no documented way
    to recover and burned metered budget (ADR-21) rediscovering the rule.

    Refuses rather than guesses when the graph is not a lockstep participant
    graph or declares no plan path: writing a plan for a graph that never
    asked for one would put an unread artifact in the tree, and silently
    writing nothing would look like success (CON-8).
    """
    report: dict = {"ok": False, "graph": str(graph_path), "errors": [], "warnings": []}
    nodes, load_errors = load_graph(graph_path)
    if nodes is None:
        report["errors"] = load_errors
        return report
    manifest_list, manifest_errors = load_manifests(root)
    if manifest_errors:
        report["errors"] = manifest_errors
        return report
    manifests = {m["id"]: m for _, m in manifest_list}

    declared = [
        str((node.get("env") or {}).get("AISLE_TURN_PLAN", "")).strip()
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("env"), dict)
    ]
    plans = [raw for raw in declared if raw.endswith(".json")]
    if not plans:
        report["errors"] = [
            _entry(
                "TURN_PLAN_ABSENT",
                {"node": str(graph_path)},
                "no node declares an AISLE_TURN_PLAN path, so this graph has no committed "
                "plan to rewrite",
                "run this on a lockstep graph; a free-run graph needs no turn plan",
            )
        ]
        return report

    try:
        plan = compile_turn_plan(nodes, manifests)
    except ValueError as exc:
        report["errors"] = [
            _entry(
                "TURN_PLAN_ABSENT",
                {"node": str(graph_path)},
                f"cannot compile a turn plan for this graph: {exc}",
                "a lockstep plan needs exactly one sim bridge and one turn barrier",
            )
        ]
        return report

    written = []
    for raw in sorted(set(plans)):
        path = Path(raw)
        if not path.is_file():
            path = graph_path.parent / raw
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n")
        written.append(str(path))
    report["ok"] = True
    report["written"] = written
    return report
