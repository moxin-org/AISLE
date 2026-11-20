"""Isolated P3 generated workers exercising public T4 requests on synthetic transport.

No simulator, provider or live guard path is launched. These are boundary checks,
not physical outcomes or measured P3 sessions. Existing qualified recordings are
used only through the same arm-bound policy evidence interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from p3_evidence import digest
from p3_isolated_mock import capture_generated_source, launch_client
from p3_mock_session import write

from aisle.harness.matched_runtime import capture_runtime
from aisle.harness.p3_task_boundary import TaskSession

RPC_SOURCE = b"""
import json,sys
from pathlib import Path
def exact(size):
    data=bytearray()
    while len(data)<size:
        chunk=sys.stdin.buffer.read(size-len(data))
        if not chunk: raise RuntimeError("controller pipe ended")
        data.extend(chunk)
    return data
def send(value):
    data=json.dumps(value).encode()
    sys.stdout.buffer.write(len(data).to_bytes(4,"big")+data)
    sys.stdout.buffer.flush()
def call(operation,arguments):
    send({"kind":"call","request":{"operation":operation,"arguments":arguments}})
    return json.loads(exact(int.from_bytes(exact(4),"big")))
for operation,arguments in [("journal",{"limit":2}),
                            ("raw",{"endpoint":"ik-trajectory/joint_cmd","limit":2})]:
    if not call(operation,arguments)["ok"]: raise RuntimeError("raw evidence refused")
call("history",{})
"""

WORKER_SOURCE = (
    RPC_SOURCE
    + b"""
assert not call("act",{"topic":"joint_cmd"})["ok"]
steps=[]
while True:
    response=call("next",{})
    if not response["ok"]: raise RuntimeError("task input refused")
    event=response["event"]
    if event is None: break
    assert "PRIVATE-SENTINEL" not in json.dumps(event)
    state=event["state"]
    if event["kind"]=="reset":
        steps=[]
    if event["kind"] in ("request","correction"):
        request=state["request"]
        steps=([("return",request["return_med"])] if request["return_med"] else [])
        steps.append(("deliver",request["med"]))
        reply=call("act",{"revision":state["revision"],"kind":"confirm","med":request["med"]})
        if not reply["ok"]: raise RuntimeError("confirmation refused")
    if steps and state["confirmed"] and state["active"] is None:
        kind,med=steps.pop(0)
        reply=call("act",{"revision":state["revision"],"kind":kind,"med":med})
        if not reply["ok"]: raise RuntimeError("fixed-library request refused")
if steps: raise RuntimeError("task input ended before planned requests")
send({"kind":"finished"})
"""
)


def participant_source():
    return (
        RPC_SOURCE
        + b'Path("worker.py").write_bytes('
        + repr(WORKER_SOURCE).encode()
        + b')\nsend({"kind":"finished"})\n'
    )


class ScriptedTaskTransport:
    """Deterministic public T4 fixture, deliberately not a physical skill executor."""

    def __init__(self):
        self.sent = []
        self.feedback = []
        self.sequence = 0
        self.events = self._events()

    def _event(self, topic, payload, **metadata):
        self.sequence += 1
        return (
            topic,
            payload,
            dict(
                env_id=0,
                sim_time_ns=self.sequence,
                seq=self.sequence,
                private_seed="PRIVATE-SENTINEL",
                **metadata,
            ),
        )

    def _last(self, topic, payload):
        if not self.sent or self.sent[-1]["topic"] != topic or self.sent[-1]["payload"] != payload:
            raise ValueError("mock worker request sequence differs")
        return self.sent[-1]["metadata"]

    def _events(self):
        yield self._event("reset_done", [1])
        yield self._event(
            "human_msg",
            dict(kind="request", med="a", text="bring a", private_goal="PRIVATE-SENTINEL"),
            goal_id="initial",
        )
        self._last("robot_msg", dict(kind="confirm", med="a", text="you want the a, correct?"))
        yield self._event(
            "human_msg", dict(kind="confirm_reply", med="a", text="yes"), goal_id="initial"
        )
        initial = self._last("target_request", dict(target_med="a"))
        # The public recovery request precedes the initial executor's home-stage completion.
        yield self._event(
            "human_msg",
            dict(kind="request", med="b", return_med="a", text="return a, bring b"),
            goal_id="recovery",
        )
        self._last("robot_msg", dict(kind="confirm", med="b", text="you want the b, correct?"))
        yield self._event(
            "human_msg", dict(kind="confirm_reply", med="b", text="yes"), goal_id="recovery"
        )
        if len(self.sent) != 3:
            raise ValueError("mock worker overlapped the initial motion")
        yield self._event("plan_done", {"ok": True}, goal_id="initial", request_id="stale")
        yield self._event("plan_done", {"ok": True}, **initial)
        returned = self._last("return_request", dict(return_med="a"))
        yield self._event("plan_done", {"ok": True}, **initial)
        yield self._event("plan_done", {"ok": True}, **returned)
        delivered = self._last("target_request", dict(target_med="b"))
        yield self._event("plan_done", {"ok": True}, **delivered)
        if len(self.sent) != 5:
            raise ValueError("mock worker issued unexpected task requests")

    def next_event(self):
        return next(self.events)

    def send(self, topic, payload, metadata):
        destination = self.feedback if topic == "episode_feedback" else self.sent
        destination.append(copy.deepcopy(dict(topic=topic, payload=payload, metadata=metadata)))


def run_mock(bundle, output):
    bundle, output = Path(bundle).resolve(strict=True), Path(output).absolute()
    if output.resolve() != output or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("mock output must be canonical and separate from evidence")
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    runtime = capture_runtime([Path(sys.base_prefix).resolve()])
    manifest_sha256 = digest(bundle / "manifest.json")
    slots = [
        dict(id=f"task-{arm}-{role}", arm=arm, role=role, status="unstarted")
        for arm in ("history", "trace")
        for role in ("participant", "generated-worker")
    ]
    write(output / "slots.json", slots)
    generated, common, task_responses = {}, {}, None
    for slot in slots:
        slot["status"] = "started"
        write(output / "slots.json", slots)
        controller, workspace = private / slot["id"], output / slot["id"]
        try:
            is_worker = slot["role"] == "generated-worker"
            source = generated[slot["arm"]] if is_worker else participant_source()
            transport = ScriptedTaskTransport()

            def factory(evidence, path, source_hash, bound_transport=transport):
                return TaskSession(
                    evidence=evidence,
                    transport=bound_transport,
                    output=path,
                    source_sha256=source_hash,
                    medicines=["a", "b"],
                )

            result = launch_client(
                source=source,
                workspace=workspace,
                controller=controller,
                bundle=bundle,
                manifest_sha256=manifest_sha256,
                arm=slot["arm"],
                session_id=slot["id"],
                runtime=runtime,
                hidden_roots=(Path(__file__).resolve().parents[1], private),
                service_factory=factory if is_worker else None,
                timeout_s=120,
            )
            if not result["ok"]:
                raise ValueError("isolated task client failed; see retained result")
            events = [
                json.loads(line)
                for line in (controller / "evidence/events.jsonl").read_text().splitlines()
            ]
            responses = [r["response"] for r in events if r["kind"] == "response"]
            if (
                len(responses) != 3
                or not all(r["ok"] for r in responses[:2])
                or responses[-1]["ok"] != (slot["arm"] == "history")
            ):
                raise ValueError("task client evidence grant differs")
            if slot["arm"] == "history":
                common[slot["role"]] = responses[:2]
            elif responses[:2] != common[slot["role"]]:
                raise ValueError("paired raw evidence differs")
            if not is_worker:
                generated[slot["arm"]] = capture_generated_source(workspace / "home")
                if generated[slot["arm"]] != WORKER_SOURCE:
                    raise ValueError("generated task source differs")
            else:
                task_log = [
                    json.loads(line)
                    for line in (controller / "task/events.jsonl").read_text().splitlines()
                ]
                # The history query is the only treatment difference. Opaque public
                # revision/motion IDs are deterministic and identical in this fixture.
                public = [r["response"] for r in task_log if r["kind"] == "response"][3:]
                if slot["arm"] == "history":
                    task_responses = public
                elif public != task_responses:
                    raise ValueError("paired task responses differ")
                summary = json.loads((controller / "task/summary.json").read_text())
                if (
                    not summary["ok"]
                    or summary["motion_requests"] != 3
                    or summary["source_sha256"] != result["source_sha256"]
                ):
                    raise ValueError("task execution receipt differs")
                write(controller / "mock-dispatches.json", transport.sent)
            slot.update(status="mock_checks_passed", source_sha256=result["source_sha256"])
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            slot.update(status="mock_failure", error=str(exc))
            break
        finally:
            write(output / "slots.json", slots)
    report = dict(
        ok=all(s["status"] == "mock_checks_passed" for s in slots),
        mode="isolated_synthetic_task_transport",
        slots=slots,
        manifest_sha256=manifest_sha256,
        launcher_sha256=digest(__file__),
        boundary_sha256=digest(
            Path(__file__).resolve().parents[1] / "src/aisle/harness/p3_task_boundary.py"
        ),
        measured_sessions=0,
        provider_calls=0,
        provider_spend_dollars=0,
        hard_dollar_ceiling=200,
        guard_traversal_verified=False,
        physical_score=None,
        p3_complete=False,
        limitation=(
            "Real isolated participant/generated-worker processes; synthetic public T4 task "
            "transport only. No physical execution, guard traversal, live task admission "
            "or measured treatment claim."
        ),
    )
    write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_mock(args.bundle, args.out)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result = dict(ok=False, error=str(exc), provider_calls=0, measured_sessions=0)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
