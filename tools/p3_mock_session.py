"""Controller-owned P3 evidence dispatch and mock-only paired interface checks.

This is not a sandbox for participant code, a robot policy runner, or a live
provider adapter. No provider/client is imported and no subprocess is launched.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from p3_evidence import (
    digest,
    list_artifacts,
    query_journal,
    query_raw,
    read_artifact,
    read_policy,
    summarize_history,
)

OPERATIONS = {
    "journal": {"offset", "limit"},
    "raw": {"endpoint", "offset", "limit"},
    "artifacts": set(),
    "artifact": {"path", "offset", "limit"},
    "history": set(),
}


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def write(path, value):
    Path(path).write_text(encoded(value))


class EvidenceSession:
    """The trusted controller binds arm, session, bundle and limits exactly once.

    Only call(request) is a participant-facing protocol. Do not expose this
    Python object, its filesystem, or these tools to participant code.
    """

    def __init__(self, *, bundle, manifest_sha256, arm, session_id, output, max_calls=32):
        if arm not in {"history", "trace"} or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", session_id):
            raise ValueError("invalid controller session binding")
        if type(max_calls) is not int or not 1 <= max_calls <= 100:
            raise ValueError("invalid call budget")
        self.bundle = Path(bundle)
        self.manifest_sha256 = manifest_sha256
        self._verify_bundle()
        self.arm, self.session_id = arm, session_id
        self.output = Path(output)
        if self.output.resolve().is_relative_to(
            self.bundle.resolve()
        ) or self.bundle.resolve().is_relative_to(self.output.resolve()):
            raise ValueError("session retention and evidence must be separate")
        self.output.mkdir(parents=True, exist_ok=False)
        self.calls = self.refused = 0
        self.max_calls, self.closed, self.budget_stopped = max_calls, False, False
        self.journal = (self.output / "events.jsonl").open("x")
        write(
            self.output / "binding.json",
            dict(
                session_id=session_id,
                arm=arm,
                manifest_sha256=manifest_sha256,
                max_calls=max_calls,
                mode="mock_only",
            ),
        )

    def _verify_bundle(self):
        if self.bundle.is_symlink() or (self.bundle / "manifest.json").is_symlink():
            raise ValueError("unsafe evidence binding")
        if digest(self.bundle / "manifest.json") != self.manifest_sha256:
            raise ValueError("bound manifest changed")
        read_policy(self.bundle)

    def _retain(self, event):
        self.journal.write(encoded(event))
        self.journal.flush()

    def call(self, request):
        if self.closed:
            raise ValueError("session is closed")
        if self.calls >= self.max_calls:
            self.budget_stopped = True
            self._retain({"kind": "stop", "reason": "call_budget_exhausted"})
            self.close()
            raise ValueError("session call budget exhausted")
        self.calls += 1
        try:
            raw = encoded(request)
            if len(raw.encode()) > 4096:
                raise ValueError("oversized request")
            retained_request = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            retained_request = {"invalid_request": True}
        self._retain(dict(kind="request", call=self.calls, request=retained_request))
        try:
            if (
                type(request) is not dict
                or retained_request.get("invalid_request") is True
                or set(request) != {"operation", "arguments"}
            ):
                raise ValueError("invalid request")
            operation, args = request["operation"], request["arguments"]
            if (
                type(operation) is not str
                or operation not in OPERATIONS
                or type(args) is not dict
                or not set(args) <= OPERATIONS[operation]
            ):
                raise ValueError("invalid operation or arguments")
            if operation == "history" and self.arm != "history":
                raise ValueError("history is not granted")
            self._verify_bundle()
            if operation == "journal":
                data = {"records": query_journal(self.bundle, **args)}
            elif operation == "raw":
                data = {"rows": query_raw(self.bundle, **args)}
            elif operation == "artifacts":
                data = {"artifacts": list_artifacts(self.bundle)}
            elif operation == "artifact":
                data = {
                    "artifact": read_artifact(
                        self.bundle, args["path"], **{k: v for k, v in args.items() if k != "path"}
                    )
                }
            else:
                data = {"invocations": summarize_history(read_policy(self.bundle))}
            response = {"ok": True, **data}
            if len(encoded(response).encode()) > 2 * 1024 * 1024:
                raise ValueError("response exceeds interface budget")
        except (OSError, ValueError, TypeError, KeyError):
            self.refused += 1
            # Do not return controller filesystem paths in backend errors.
            response = {"ok": False, "error": "evidence_request_refused"}
        self._retain(dict(kind="response", call=self.calls, response=response))
        return response

    def close(self):
        if not self.closed:
            self.closed = True
            self.journal.close()
            write(
                self.output / "summary.json",
                dict(
                    session_id=self.session_id,
                    calls=self.calls,
                    refused_calls=self.refused,
                    budget_stopped=self.budget_stopped,
                    measured_pilot=False,
                    provider_calls=0,
                    provider_spend_dollars=0,
                ),
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_mock(bundle, output, *, blocks=6):
    """Fixed synthetic interface exercises; not pilot assignments or acceptance scores."""
    if type(blocks) is not int or not 1 <= blocks <= 6:
        raise ValueError("mock block count must be 1..6")
    bundle, output = Path(bundle), Path(output)
    manifest_sha256 = digest(bundle / "manifest.json")
    read_policy(bundle)
    if output.resolve().is_relative_to(bundle.resolve()) or bundle.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("mock output and evidence must be separate")
    output.mkdir(parents=True, exist_ok=False)
    slots = [
        dict(session_id=f"mock-{block}-{arm}", block=block, arm=arm, status="unstarted")
        for block in range(1, blocks + 1)
        for arm in ("history", "trace")
    ]
    write(output / "slots.json", slots)
    common = [
        {"operation": "journal", "arguments": {"limit": 2}},
        {"operation": "artifacts", "arguments": {}},
    ]
    request_history = {"operation": "history", "arguments": {}}
    raw_responses = {}
    for slot in slots:
        slot["status"] = "started"
        write(output / "slots.json", slots)
        try:
            with EvidenceSession(
                bundle=bundle,
                manifest_sha256=manifest_sha256,
                arm=slot["arm"],
                session_id=slot["session_id"],
                output=output / slot["session_id"],
            ) as controller:
                raw = [controller.call(request) for request in common]
                if not all(result["ok"] for result in raw):
                    raise ValueError("common evidence request failed")
                history = controller.call(request_history)
                if history["ok"] != (slot["arm"] == "history"):
                    raise ValueError("history grant differs from assigned treatment")
                if slot["arm"] == "history":
                    raw_responses[slot["block"]] = raw
                elif raw != raw_responses[slot["block"]]:
                    raise ValueError("raw evidence differs between matched mock sessions")
            slot["status"] = "mock_checks_passed"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            slot.update(status="mock_failure", error=str(exc))
            write(output / "slots.json", slots)
            break
        write(output / "slots.json", slots)
    result = dict(
        ok=all(s["status"] == "mock_checks_passed" for s in slots),
        mode="mock_only",
        mock_sessions=sum(s["status"] != "unstarted" for s in slots),
        measured_sessions=0,
        provider_calls=0,
        provider_spend_dollars=0,
        hard_dollar_ceiling=200,
        api_invocation_authorized=False,
        p3_complete=False,
        p4_on_hold=True,
        slots=slots,
        manifest_sha256=manifest_sha256,
        limitation=(
            "No participant/worker isolation, physics, provider or treatment-effect measurement."
        ),
    )
    write(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        result = run_mock(args.bundle, args.out, blocks=args.blocks)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "error": str(exc), "mode": "mock_only", "provider_calls": 0}
    print(encoded(result), end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
