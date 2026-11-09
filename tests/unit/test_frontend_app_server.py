"""MON-8/MON-12/MON-13: app-server requests are bound to the active thread and turn."""

import copy

import pytest

pytestmark = pytest.mark.unit


def _call():
    return {
        "id": 7,
        "method": "item/tool/call",
        "params": {
            "threadId": "thread",
            "turnId": "turn",
            "callId": "call",
            "namespace": "harness",
            "tool": "check",
            "arguments": {},
        },
    }


def test_dynamic_call_has_exact_frontend_identity():
    """MON-12: preserve the app-server call identity before creating a request."""
    from aisle.harness.frontend_app_server import parse_dynamic_call

    assert parse_dynamic_call(_call(), thread_id="thread", turn_id="turn") == {
        "turn_id": "turn",
        "call_id": "call",
        "tool_name": "harness.check",
    }


@pytest.mark.parametrize(
    "change", ["thread", "turn", "namespace", "tool", "arguments", "extra", "id"]
)
def test_dynamic_call_cannot_change_the_admitted_request(change):
    """MON-8/MON-13: foreign calls and participant-selected run parameters fail closed."""
    from aisle.harness.frontend_app_server import parse_dynamic_call

    call = copy.deepcopy(_call())
    if change in ("thread", "turn"):
        call["params"][change + "Id"] = "foreign"
    elif change == "arguments":
        call["params"]["arguments"] = {"seed": 123}
    elif change == "extra":
        call["params"]["authorization_id"] = "invented"
    elif change == "id":
        call["id"] = True
    else:
        call["params"][change] = "foreign"
    with pytest.raises(ValueError):
        parse_dynamic_call(call, thread_id="thread", turn_id="turn")


@pytest.mark.parametrize("failure", ["foreign_call", "timeout", "eof", "malformed_then_call"])
def test_owned_transport_failure_never_calls_controller_and_reaps_process(tmp_path, failure):
    """MON-12/MON-13: an invalid or unavailable frontend cannot invoke an authorized tool."""
    import os
    import sys

    from aisle.harness.frontend_app_server import run_app_server

    script = tmp_path / "server.py"
    pidfile = tmp_path / "pid"
    script.write_text("""import json,os,pathlib,sys,time
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
for raw in sys.stdin:
    row=json.loads(raw)
    if 'id' not in row:
        continue
    result={}
    if row['id']=='thread':
        result={'thread':{'id':'thread'}}
    if row['id']=='turn':
        result={'turn':{'id':'turn'}}
    print(json.dumps({'id':row['id'],'result':result}),flush=True)
    if row['id']=='turn':
        if sys.argv[2]=='malformed_then_call':
            print('{}',flush=True)
            print(json.dumps({'id':7,'method':'item/tool/call','params':{
                'threadId':'thread','turnId':'turn','callId':'call',
                'namespace':'harness','tool':'check','arguments':{}}}),flush=True)
        elif sys.argv[2]=='foreign_call':
            print(json.dumps({'id':7,'method':'item/tool/call','params':{
                'threadId':'foreign','turnId':'turn','callId':'call',
                'namespace':'harness','tool':'check','arguments':{}}}),flush=True)
        elif sys.argv[2]=='eof':
            break
        time.sleep(60)
""")
    called = []
    references = []
    with pytest.raises((ValueError, TimeoutError)):
        run_app_server(
            [sys.executable, "-I", str(script), str(pidfile), failure],
            cwd=tmp_path,
            env={"PATH": os.defpath},
            output=tmp_path / "protocol",
            thread_params={},
            input_items=[],
            handle_call=lambda *args: called.append(args),
            timeout_s=0.5,
            on_reference=references.append,
        )
    assert not called
    assert (tmp_path / "protocol/failure.json").is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    import hashlib

    assert len(references) == 1
    reference = references[0]
    assert reference["stream_complete"] is False
    assert reference["failure"] is not None
    assert "failure.json" in reference["artifacts"]
    for name, digest in reference["artifacts"].items():
        assert hashlib.sha256((tmp_path / "protocol" / name).read_bytes()).hexdigest() == digest


def _usage(input_tokens=10, cached=3, output=2):
    totals = {
        "inputTokens": input_tokens,
        "cachedInputTokens": cached,
        "outputTokens": output,
        "reasoningOutputTokens": 0,
        "totalTokens": input_tokens + output,
    }
    return {
        "threadId": "thread",
        "turnId": "turn",
        "tokenUsage": {"total": totals, "last": dict(totals)},
    }


def test_cancelled_frontend_preserves_primary_error_when_reference_callback_fails(
    tmp_path, monkeypatch
):
    """MON-12/MON-13: cancellation keeps its identity even if evidence finalization fails."""
    import asyncio

    from aisle.harness import frontend_app_server as frontend

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError("session cancelled")

    references = []

    def retain(reference):
        references.append(reference)
        raise OSError("reference storage unavailable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cancelled)
    with pytest.raises(asyncio.CancelledError, match="session cancelled") as caught:
        frontend.run_app_server(
            ["unused"],
            cwd=tmp_path,
            env={},
            output=tmp_path / "protocol",
            thread_params={},
            input_items=[],
            handle_call=lambda *args: None,
            timeout_s=1,
            on_reference=retain,
        )
    assert len(references) == 1
    assert references[0]["stream_complete"] is False
    assert references[0]["thread_id"] is None
    assert references[0]["failure"]["error_type"] == "CancelledError"
    assert "failure.json" in references[0]["artifacts"]
    assert any("reference storage unavailable" in note for note in caught.value.__notes__)


def test_app_server_usage_counts_cumulative_updates_once():
    """MON-8/MON-12: repeated cumulative usage cannot double-count shared token budgets."""
    from aisle.harness.frontend_app_server import AppServerUsage

    meter = AppServerUsage("thread", "turn")
    meter.feed(_usage())
    meter.feed(_usage())
    assert meter.report() == {"tokens": 9, "tokens_generated": 2}
    meter.feed(_usage(input_tokens=15, cached=5, output=4))
    assert meter.report() == {"tokens": 14, "tokens_generated": 4}


@pytest.mark.parametrize(
    "drift", ["missing", "foreign", "negative", "boolean", "cache", "backwards"]
)
def test_app_server_usage_rejects_missing_or_invalid_accounting(drift):
    """MON-8/MON-13: missing or inconsistent usage cannot become a zero-token session."""
    from aisle.harness.frontend_app_server import AppServerUsage

    meter = AppServerUsage("thread", "turn")
    value = _usage()
    if drift == "foreign":
        value["threadId"] = "foreign"
    elif drift == "negative":
        value["tokenUsage"]["total"]["outputTokens"] = -1
    elif drift == "boolean":
        value["tokenUsage"]["total"]["outputTokens"] = True
    elif drift == "cache":
        value["tokenUsage"]["total"]["cachedInputTokens"] = 11
    elif drift == "backwards":
        meter.feed(_usage(input_tokens=20))
    with pytest.raises(ValueError):
        if drift != "missing":
            meter.feed(value)
        meter.report()


def test_cached_usage_cannot_reduce_already_observed_spend():
    """MON-8/MON-13: growing cumulative cache counts cannot refund observed token spend."""
    from aisle.harness.frontend_app_server import AppServerUsage

    meter = AppServerUsage("thread", "turn")
    meter.feed(_usage())
    with pytest.raises(ValueError):
        meter.feed(_usage(cached=8))


def test_last_usage_cannot_exceed_cumulative_usage():
    """MON-12/MON-13: a last response cannot exceed the same thread's cumulative totals."""
    from aisle.harness.frontend_app_server import AppServerUsage

    value = _usage()
    value["tokenUsage"]["last"] = _usage(input_tokens=20)["tokenUsage"]["total"]
    with pytest.raises(ValueError):
        AppServerUsage("thread", "turn").feed(value)


def test_async_transport_cancellation_reaps_owned_frontend(tmp_path):
    """MON-13: a failed nested authority can cancel the frontend while its pipe is idle."""
    import asyncio
    import os
    import sys

    from aisle.harness.frontend_app_server import run_app_server_async

    async def exercise():
        pidfile = tmp_path / "pid"
        script = tmp_path / "idle.py"
        script.write_text(
            "import os,pathlib,sys,time\n"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        task = asyncio.create_task(
            run_app_server_async(
                [sys.executable, "-I", str(script), str(pidfile)],
                cwd=tmp_path,
                env={"PATH": os.defpath},
                output=tmp_path / "protocol",
                thread_params={},
                input_items=[],
                handle_call=lambda *_: pytest.fail("unexpected tool"),
                timeout_s=60,
            )
        )
        try:
            async with asyncio.timeout(2):
                while not pidfile.exists():
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            with pytest.raises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)
            assert (tmp_path / "protocol/failure.json").is_file()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_cancellation_during_reaping_cannot_publish_reference_early(tmp_path, monkeypatch):
    """MON-12/MON-13: relay cancellation must not interrupt a frontend's ongoing cleanup."""
    import asyncio
    import os
    import signal
    import sys

    from aisle.harness.frontend_app_server import run_app_server_async

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        real_spawn = asyncio.create_subprocess_exec
        process = None
        original_wait = None
        reaped = False
        references = []

        async def spawn(*args, **kwargs):
            nonlocal process, original_wait
            process = await real_spawn(*args, **kwargs)
            original_wait = process.wait

            async def wait():
                nonlocal reaped
                entered.set()
                await release.wait()
                result = await original_wait()
                reaped = True
                return result

            process.wait = wait
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            run_app_server_async(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import time; print('invalid', flush=True); time.sleep(60)",
                ],
                cwd=tmp_path,
                env={"PATH": os.defpath},
                output=tmp_path / "protocol",
                thread_params={},
                input_items=[],
                handle_call=lambda *_: pytest.fail("unexpected tool"),
                timeout_s=5,
                on_reference=lambda reference: references.append((reaped, reference)),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert references == [], "failure reference was published before process cleanup"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert references and references[0][0] is True
            assert process.returncode is not None
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await original_wait()

    asyncio.run(exercise())


def test_controller_environment_reaches_the_child_but_not_the_declared_hash(tmp_path):
    """TRT-7/MON-13 (ADR-69 §6): a controller-only secret is delivered to the
    frontend process, while the retained invocation hashes only the declared
    environment and names the added keys, never their values."""
    import hashlib
    import json
    import os
    import sys

    from aisle.harness.frontend_app_server import run_app_server

    script = tmp_path / "server.py"
    seen = tmp_path / "seen"
    script.write_text(
        "import os,pathlib,sys\n"
        "pathlib.Path(sys.argv[1]).write_text(os.environ.get('AISLE_RELAY_KEY',''))\n"
    )
    declared = {"PATH": os.defpath}
    secret = "s" * 43
    with pytest.raises((ValueError, TimeoutError)):
        run_app_server(
            [sys.executable, "-I", str(script), str(seen)],
            cwd=tmp_path,
            env=declared,
            controller_env={"AISLE_RELAY_KEY": secret},
            output=tmp_path / "protocol",
            thread_params={},
            input_items=[],
            handle_call=lambda *args: None,
            timeout_s=2,
        )
    assert seen.read_text() == secret
    invocation = json.loads((tmp_path / "protocol" / "invocation.json").read_text())
    expected = hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()
    assert invocation["environment_sha256"] == expected
    assert invocation["controller_environment_keys"] == ["AISLE_RELAY_KEY"]
    retained = b"".join(p.read_bytes() for p in (tmp_path / "protocol").iterdir())
    assert secret.encode() not in retained


def test_controller_environment_cannot_override_declared_keys(tmp_path):
    """TRT-7: the controller may add a variable, never replace a declared one."""
    import os

    from aisle.harness.frontend_app_server import run_app_server

    with pytest.raises(ValueError, match="controller environment"):
        run_app_server(
            ["/usr/bin/true"],
            cwd=tmp_path,
            env={"PATH": os.defpath},
            controller_env={"PATH": "/tmp"},
            output=tmp_path / "protocol",
            thread_params={},
            input_items=[],
            handle_call=lambda *args: None,
            timeout_s=1,
        )
