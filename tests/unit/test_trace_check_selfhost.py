"""Unit tests for tools/trace_check.py (SPEC 070 HAR-9, CON-8).

HAR-9: trace_check scans specs for MUST requirement IDs and tests for
docstring citations; exits nonzero listing uncovered MUSTs.
"""

import json
from pathlib import Path

import pytest
from cli_helpers import run_tool

pytestmark = pytest.mark.unit


def check(*args: str) -> tuple[int, dict]:
    """Run trace_check; return (exit code, parsed JSON report)."""
    proc = run_tool("trace_check.py", *args)
    return proc.returncode, json.loads(proc.stdout)


def make_fixture(tmp_path: Path, spec_text: str, test_text: str | None = None) -> Path:
    """Build a minimal repo root with one spec and optionally one test file."""
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "999-fixture.md").write_text(spec_text)
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    if test_text is not None:
        (tmp_path / "tests" / "unit" / "test_fixture.py").write_text(test_text)
    return tmp_path


def test_selfhost_passes():
    """HAR-9: trace_check runs on this repository and exits 0 with ok=true.

    Every MUST-bearing requirement ID in specs/ is either cited by a test
    docstring or explicitly waived in tools/trace_waivers.toml.
    """
    code, report = check()
    assert code == 0, report
    assert report["ok"] is True
    assert report["uncovered"] == []


def test_coverage_map_lists_citing_tests(tmp_path):
    """HAR-9: the report carries a deterministic MUST-ID -> citing-test
    mapping ("relpath::test_name"), both self-hosted and on fixtures."""
    _, report = check()
    har9_tests = report["coverage_map"]["HAR-9"]
    assert "tests/unit/test_trace_check_selfhost.py::test_selfhost_passes" in har9_tests
    assert har9_tests == sorted(har9_tests)
    _, again = check()
    assert again["coverage_map"] == report["coverage_map"]

    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_heartbeat():\n    """ZZ-1: heartbeats are published."""\n',
    )
    _, fixture_report = check("--root", str(root))
    assert fixture_report["coverage_map"] == {
        "ZZ-1": ["tests/unit/test_fixture.py::test_heartbeat"]
    }


def test_cli_json_stdout(tmp_path):
    """CON-8: trace_check emits a single JSON object on stdout; exit 0 iff ok."""
    root = make_fixture(tmp_path, "- ZZ-1: The tool MUST do the thing.\n")
    code, report = check("--root", str(root))  # json.loads proves parseable
    assert isinstance(report, dict)
    assert report["ok"] is False
    assert code != 0


def test_uncovered_must_fails(tmp_path):
    """HAR-9: an uncited MUST requirement makes trace_check exit nonzero
    and list the uncovered ID."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-1" in report["uncovered"]


def test_covered_must_passes(tmp_path):
    """HAR-9: a MUST requirement cited in a test docstring counts as covered."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_heartbeat():\n    """ZZ-1: heartbeats are published."""\n',
    )
    code, report = check("--root", str(root))
    assert code == 0, report
    assert report["ok"] is True
    assert "ZZ-1" in report["covered"]


def test_non_must_id_not_required(tmp_path):
    """HAR-9: only MUST-bearing requirements (RFC 2119 MUST/REQUIRED/SHALL)
    demand test coverage; descriptive IDs do not."""
    root = make_fixture(tmp_path, "- ZZ-2: This bullet is descriptive only.\n")
    code, report = check("--root", str(root))
    assert code == 0
    assert report["ok"] is True


def test_multiline_requirement_detected(tmp_path):
    """HAR-9: requirement text wrapping across indented continuation lines is
    still scanned for MUST keywords (constitution style)."""
    spec = "- ZZ-3: Long requirement that wraps and the keyword\n  MUST appears on the next line.\n"
    root = make_fixture(tmp_path, spec)
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-3" in report["uncovered"]


def test_blank_line_continuation_detected(tmp_path):
    """HAR-9: a blank line inside a multi-paragraph list item does not detach
    the indented continuation from its requirement ID."""
    spec = "- ZZ-4: Intro paragraph.\n\n  The tool MUST reject bad input.\n"
    root = make_fixture(tmp_path, spec)
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-4" in report["uncovered"]


def test_unknown_citation_fails(tmp_path):
    """HAR-9: a test citing an undefined ID with a known spec prefix is an
    error (catches number typos that would silently fake coverage)."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_a():\n    """ZZ-1, ZZ-77: covered."""\n',
    )
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-77" in report["unknown_citations"]


def test_non_test_docstrings_do_not_count_as_coverage(tmp_path):
    """HAR-9: only docstrings of test callables (test_* functions/methods)
    count as citations; module, class, and helper docstrings do not, so a
    stray mention cannot fake coverage."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        '"""Module docstring citing ZZ-1."""\n\n'
        "class TestGroup:\n"
        '    """Class docstring citing ZZ-1."""\n\n'
        "def helper():\n"
        '    """Helper docstring citing ZZ-1."""\n',
    )
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-1" in report["uncovered"]


def test_invalid_specs_range_reported_as_json(tmp_path):
    """CON-8: a malformed, reversed, or empty --specs range yields a
    full-shape JSON error report on stdout (all standard keys present, the
    bad value named in errors) and a nonzero exit, not a Python traceback."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    for bad in ("nope", "10-", "080", "080-000", ""):
        proc = run_tool("trace_check.py", "--root", str(root), "--specs", bad)
        report = json.loads(proc.stdout)
        assert proc.returncode != 0
        assert report["ok"] is False
        assert any(f"{bad!r}" in e for e in report["errors"]), (bad, report["errors"])
        assert "uncovered" in report and "parse_errors" in report  # full shape


def test_specs_range_matching_no_spec_files_fails(tmp_path):
    """HAR-9: a --specs range that selects zero spec files is an error, not a
    vacuously green gate."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    code, report = check("--root", str(root), "--specs", "100-150")
    assert code != 0
    assert report["ok"] is False
    assert any("100-150" in e for e in report["errors"])


def test_non_numeric_spec_filename_fails(tmp_path):
    """HAR-9: a specs/ file without the NNN- numeric prefix is a hard error
    (it could never be scoped by --specs and breaks the naming convention)."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    (root / "specs" / "notes.md").write_text("- YY-1: Stray requirement.\n")
    code, report = check("--root", str(root))
    assert code != 0
    assert any("notes.md" in e for e in report["errors"])


def test_pytest_style_test_names_count_as_coverage(tmp_path):
    """HAR-9: any pytest-collected test callable counts (default collection
    pattern is test*, not only test_*)."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def testHeartbeat():\n    """ZZ-1: heartbeats are published."""\n    assert True\n',
    )
    code, report = check("--root", str(root))
    assert code == 0, report
    assert "ZZ-1" in report["covered"]


def test_ordinary_hyphenated_tokens_are_not_citations(tmp_path):
    """HAR-9: docstring tokens whose prefix matches no spec (SHA-256, UTF-8)
    are ignored, not flagged as citations."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_a():\n    """ZZ-1: hashes with SHA-256 over UTF-8 text."""\n',
    )
    code, report = check("--root", str(root))
    assert code == 0, report
    assert report["unknown_citations"] == []


def test_duplicate_id_across_specs_fails(tmp_path):
    """HAR-9: the same requirement ID defined in two specs is an error, not a
    silent last-wins overwrite that could shadow a MUST."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    (root / "specs" / "998-other.md").write_text("- ZZ-1: Descriptive duplicate.\n")
    code, report = check("--root", str(root))
    assert code != 0
    assert "ZZ-1" in report["duplicate_ids"]


def test_empty_specs_dir_fails(tmp_path):
    """HAR-9: zero requirements found (missing or empty specs/) is an error,
    never a green gate."""
    (tmp_path / "specs").mkdir()
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    code, report = check("--root", str(tmp_path))
    assert code != 0
    assert report["ok"] is False


def test_unparseable_test_file_reported_as_json(tmp_path):
    """CON-8: a test file with a syntax error yields a JSON error report on
    stdout naming the file, not a Python traceback."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        "def broken(:\n",
    )
    code, report = check("--root", str(root))
    assert code != 0
    assert report["ok"] is False
    assert any("test_fixture.py" in e for e in report["parse_errors"])


def test_waiver_suppresses_and_strict_ignores_waivers(tmp_path):
    """HAR-9: waivers in tools/trace_waivers.toml suppress uncovered MUSTs
    pre-M0; --strict ignores waivers (the M0 gate runs strict). See ADR 1."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    (root / "tools").mkdir()
    (root / "tools" / "trace_waivers.toml").write_text('[waivers]\nZZ-1 = "deferred to T99"\n')
    code, report = check("--root", str(root))
    assert code == 0
    assert "ZZ-1" in report["waived"]

    strict_code, strict_report = check("--root", str(root), "--strict")
    assert strict_code != 0
    assert "ZZ-1" in strict_report["uncovered"]


def test_waiver_for_unknown_id_fails(tmp_path):
    """HAR-9: a waiver naming an ID that no spec defines is an error
    (keeps the waiver file honest as specs evolve). See ADR 1."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    (root / "tools").mkdir()
    (root / "tools" / "trace_waivers.toml").write_text('[waivers]\nGONE-1 = "stale"\nZZ-1 = "ok"\n')
    code, report = check("--root", str(root))
    assert code != 0
    assert "GONE-1" in report["unknown_waivers"]


def test_specs_range_scopes_strict_gate(tmp_path):
    """HAR-9: --specs NNN-MMM restricts the MUST universe to spec files in
    that number range, so the M0 gate can run strict on specs 000-080 while
    post-M0 specs stay out of scope. See ADR 1."""
    root = make_fixture(tmp_path, "")
    (root / "specs" / "010-in-scope.md").write_text("- AA-1: The bridge MUST publish state.\n")
    (root / "specs" / "210-post-m0.md").write_text("- BB-1: The base MUST navigate.\n")
    code, report = check("--root", str(root), "--strict", "--specs", "000-080")
    assert code != 0
    assert report["uncovered"] == ["AA-1"]


def add_rust_test(root: Path, text: str) -> None:
    """Add one Rust integration-test file under native/demo/tests/."""
    tests_dir = root / "native" / "demo" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "contract.rs").write_text(text)


def test_rust_test_citing_undefined_id_fails(tmp_path):
    """HAR-9: a Rust test under native/*/tests citing an undefined ID with a
    known spec prefix fails the check, exactly like a Python test would."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_a():\n    """ZZ-1: covered."""\n',
    )
    add_rust_test(root, "/// ZZ-1, ZZ-77: heartbeats.\n#[test]\nfn publishes() {}\n")
    code, report = check("--root", str(root))
    assert code != 0
    assert report["unknown_citations"] == ["ZZ-77"]


def test_rust_citations_are_reported_but_are_not_must_coverage(tmp_path):
    """HAR-9: Rust citations are listed per ID ("relpath::fn") but do not
    cover a MUST; the experimental native surface cannot fake normative
    coverage (ADR-73)."""
    root = make_fixture(tmp_path, "- ZZ-1: Producers MUST publish heartbeats.\n")
    add_rust_test(
        root,
        '/// ZZ-1: heartbeats,\n/// continued.\n#[test]\n#[ignore = "slow"]\nfn publishes() {}\n',
    )
    code, report = check("--root", str(root))
    assert code != 0
    assert report["uncovered"] == ["ZZ-1"]
    assert report["rust_citations"] == {"ZZ-1": ["native/demo/tests/contract.rs::publishes"]}


def test_rust_doc_comments_outside_tests_are_not_citations(tmp_path):
    """HAR-9: only doc comments on #[test] functions are Rust citations;
    helpers and module docs may mention IDs freely."""
    root = make_fixture(
        tmp_path,
        "- ZZ-1: Producers MUST publish heartbeats.\n",
        'def test_a():\n    """ZZ-1: covered."""\n',
    )
    add_rust_test(
        root,
        "//! Module docs mention ZZ-77.\n"
        "/// Helper mentions ZZ-78.\nfn helper() {}\n"
        "/// ZZ-1: real.\n#[test]\nfn publishes() { helper() }\n",
    )
    code, report = check("--root", str(root))
    assert code == 0, report
    assert report["rust_citations"] == {"ZZ-1": ["native/demo/tests/contract.rs::publishes"]}


def test_selfhost_lists_native_nano_citations():
    """HAR-9: the repository's own native Nano Rust tests are scanned, so a
    mistyped requirement ID there fails CI."""
    _, report = check()
    cited = report["rust_citations"]
    assert "native/nano/tests/judge.rs::task_success_rules" in cited["TC-8"]
    assert all(ref.startswith("native/") for refs in cited.values() for ref in refs)
