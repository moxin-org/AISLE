"""ADR-30 terminal run-to-quiescence barrier node."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from aisle.turns import ProtocolError, TurnBarrier, TurnWatchdog


def load_plan(environ) -> dict:
    raw = environ.get("AISLE_TURN_PLAN", "").strip()
    if not raw:
        raise SystemExit("turn-barrier config refused: AISLE_TURN_PLAN is required")
    try:
        if raw.endswith(".json"):
            path = Path(raw)
            if not path.is_file():
                path = Path(__file__).resolve().parents[3] / "graphs" / raw
            plan = json.loads(path.read_text(encoding="utf-8"))
        else:
            plan = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"turn-barrier config refused: cannot load AISLE_TURN_PLAN: {exc}"
        ) from exc
    return plan


#: The DEFAULT ordinary-turn watchdog (ADR-30): planning that runs
#: synchronously inside a turn must finish well inside this. Named so
#: planners and tests read the constraint from source instead of
#: hardcoding it (G1-F8); the AISLE_TURN_WATCHDOG_S override is unchanged.
ORDINARY_TURN_WATCHDOG_S = 10.0


def main() -> None:  # pragma: no cover - exercised by graph acceptance
    import pyarrow as pa
    from dora import Node

    from aisle.topics import make_sender

    plan = load_plan(os.environ)
    node = Node()
    send = make_sender(node)
    barrier = TurnBarrier(plan)
    watchdog = TurnWatchdog(
        # Includes host scheduler stalls and first-use model warmup while
        # remaining far below episode wall clamps.  It is liveness-only and
        # can never advance physics.
        ordinary_s=float(os.environ.get("AISLE_TURN_WATCHDOG_S", str(ORDINARY_TURN_WATCHDOG_S))),
        verdict_s=float(os.environ.get("AISLE_VERDICT_TURN_WATCHDOG_S", "15")),
        clock=time.monotonic,
    )

    def dispatch(ready: dict[str, dict[str, int]]) -> None:
        if barrier.current is None or not ready:
            return
        ready_plan = {}
        for target, expected in sorted(ready.items()):
            if barrier.is_verdict_bearing(target):
                watchdog.mark_verdict_bearing()
            names = sorted(expected)
            input_stamps = barrier.input_stamps(target)
            ready_plan[target] = {
                "expected_inputs": names,
                "expected_counts": [expected[name] for name in names],
                "expected_turn_epochs": [input_stamps[name].epoch for name in names],
                "expected_turn_ids": [input_stamps[name].turn_id for name in names],
                "expected_sim_time_ns": [input_stamps[name].sim_time_ns for name in names],
            }
        # One dora output already fans out to every participant. Sending one
        # targeted copy per ready node makes each causal layer O(N^2) at the
        # transport. Broadcast the whole ready layer once; wrappers select
        # their own exact declaration.
        send(
            "turn",
            pa.array([barrier.current.turn_id], type=pa.uint64()),
            {**barrier.current.metadata(), "ready_plan": json.dumps(ready_plan, sort_keys=True)},
        )

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        try:
            if event["id"] == "sim_turn":
                watchdog.open()
                dispatch(barrier.open_bridge(metadata))
            elif event["id"] == "tick":
                if watchdog.expired and not barrier.complete:
                    current = barrier.current
                    raise ProtocolError(
                        f"turn watchdog expired ({watchdog.turn_type}) at "
                        f"{current.epoch}:{current.turn_id}"
                        if current
                        else "turn watchdog expired"
                    )
            else:
                source_node = metadata.get("source_node")
                if not isinstance(source_node, str) or not source_node:
                    raise ProtocolError(f"{event['id']} missing source_node")
                dispatch(barrier.close(source_node, metadata))
                if barrier.complete:
                    assert barrier.current is not None
                    expected = barrier.bridge_expected_inputs()
                    names = sorted(expected)
                    send(
                        "turn_commit",
                        pa.array([barrier.current.turn_id], type=pa.uint64()),
                        {
                            **barrier.current.metadata(),
                            "expected_inputs": names,
                            "expected_counts": [expected[name] for name in names],
                            **({"shutdown": True} if barrier.shutdown_requested else {}),
                        },
                    )
                    if barrier.shutdown_requested:
                        return
        except ProtocolError as exc:
            print(f"ADR-30 protocol error: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
