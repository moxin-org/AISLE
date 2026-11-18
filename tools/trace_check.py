#!/usr/bin/env python3
"""trace_check: enforce spec→test traceability (HAR-9, CON-8).

Scans specs/*.md for requirement IDs (`- <ID>: text` bullets), flags the
MUST-bearing ones (RFC 2119 MUST / REQUIRED / SHALL in the requirement text),
and scans test_* function docstrings under tests/ for citations. Exits nonzero listing
uncovered MUSTs, citations or waivers of undefined IDs, duplicate IDs, and
unparseable test files.

Rust tests under native/*/tests/ are scanned too: doc comments on #[test]
functions are checked for undefined IDs and listed as `rust_citations`, but
they do not count as MUST coverage (experimental native surface, ADR-73).

Pre-M0, not-yet-implemented specs are waived in tools/trace_waivers.toml
(id + reason, reviewable); --strict ignores waivers and --specs NNN-MMM
scopes the MUST universe by spec number (the M0 gate runs
`--strict --specs 000-080`). See docs/decisions/ADR-1.md.
"""

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

ID_PATTERN = r"[A-Z][A-Z0-9]*-[A-Z]?\d+"
REQ_BULLET = re.compile(rf"^- ({ID_PATTERN}):(.*)$")
MUST_KEYWORD = re.compile(r"\b(MUST|REQUIRED|SHALL)\b")
RUST_TEST_ATTRIBUTE = re.compile(r"#\[(\w+::)*test\b")
RUST_FN = re.compile(r"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)")


def extract_requirements(specs_dir: Path) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """Scan all specs. Returns (id → requirement text, spec file number →
    ids defined there, ids defined more than once).

    Requirement text spans the bullet plus indented continuation lines;
    blank lines inside a list item do not end it.
    """
    requirements: dict[str, str] = {}
    by_spec_number: dict[str, set[str]] = {}
    duplicates: set[str] = set()
    for spec in sorted(specs_dir.glob("*.md")):
        number = spec.name.split("-")[0]
        ids_here = by_spec_number.setdefault(number, set())
        current_id = None
        for line in spec.read_text().splitlines():
            match = REQ_BULLET.match(line)
            if match:
                current_id = match.group(1)
                if current_id in requirements:
                    duplicates.add(current_id)
                requirements[current_id] = match.group(2)
                ids_here.add(current_id)
            elif not line.strip():
                continue  # blank line inside a list item keeps the current requirement
            elif current_id and line.startswith(" "):
                requirements[current_id] += " " + line.strip()
            else:
                current_id = None
    return requirements, by_spec_number, duplicates


def extract_citations(tests_dir: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Map requirement ID -> sorted test references ("relpath::name") from
    docstrings of test callables (test* functions/methods only — module,
    class, and helper docstrings do not count as coverage); report
    unparseable files instead of crashing (CON-8)."""
    cited: dict[str, set[str]] = {}
    parse_errors: list[str] = []
    for path in sorted(tests_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(f"{path}: {exc.msg} (line {exc.lineno})")
            continue
        ref_base = path.relative_to(tests_dir.parent).as_posix()
        for node in ast.walk(tree):
            # pytest's default collection pattern is test*, not test_*
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                "test"
            ):
                doc = ast.get_docstring(node)
                for rid in re.findall(ID_PATTERN, doc or ""):
                    cited.setdefault(rid, set()).add(f"{ref_base}::{node.name}")
    return {rid: sorted(refs) for rid, refs in cited.items()}, parse_errors


def extract_rust_citations(root: Path) -> dict[str, list[str]]:
    """Map requirement ID -> sorted Rust test references ("relpath::fn") from
    `///` doc comments directly above #[test] functions in native/*/tests/.
    Module docs and helper functions do not count."""
    cited: dict[str, set[str]] = {}
    for path in sorted((root / "native").glob("*/tests/**/*.rs")):
        ref_base = path.relative_to(root).as_posix()
        doc: list[str] = []
        is_test = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("///"):
                doc.append(stripped)
            elif stripped.startswith("#["):
                is_test = is_test or bool(RUST_TEST_ATTRIBUTE.match(stripped))
            else:
                match = RUST_FN.match(stripped)
                if match and is_test:
                    for rid in re.findall(ID_PATTERN, " ".join(doc)):
                        cited.setdefault(rid, set()).add(f"{ref_base}::{match.group(1)}")
                doc, is_test = [], False
    return {rid: sorted(refs) for rid, refs in cited.items()}


def load_waivers(root: Path) -> dict[str, str]:
    """Flat id → reason table under [waivers]."""
    path = root / "tools" / "trace_waivers.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f).get("waivers", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--strict", action="store_true", help="ignore waivers (the M0 gate runs strict)"
    )
    parser.add_argument(
        "--specs", metavar="NNN-MMM", help="restrict the MUST universe to this spec-number range"
    )
    args = parser.parse_args()

    errors: list[str] = []
    span = None
    if args.specs is not None:
        match = re.fullmatch(r"(\d{1,4})-(\d{1,4})", args.specs)
        span = (int(match.group(1)), int(match.group(2))) if match else None
        if span is None or span[0] > span[1]:
            errors.append(f"invalid --specs range {args.specs!r} (expected NNN-MMM, NNN <= MMM)")
            span = None

    requirements, by_spec_number, duplicates = extract_requirements(args.root / "specs")
    errors += [
        f"spec file without NNN- numeric prefix: {prefix}"
        for prefix in sorted(by_spec_number)
        if not prefix.isdigit()
    ]
    must_ids = {rid for rid, text in requirements.items() if MUST_KEYWORD.search(text)}
    if span:
        in_range = {n for n in by_spec_number if n.isdigit() and span[0] <= int(n) <= span[1]}
        if in_range:
            must_ids &= {rid for n in in_range for rid in by_spec_number[n]}
        else:
            errors.append(f"--specs {args.specs} matches no spec files")
    if args.specs is not None and (span is None or not in_range):
        must_ids = set()  # range unusable: report the error, not a misleading uncovered list
    citation_map, parse_errors = extract_citations(args.root / "tests")
    known_prefixes = {rid.split("-")[0] for rid in requirements}
    citation_map = {c: t for c, t in citation_map.items() if c.split("-")[0] in known_prefixes}
    cited = set(citation_map)
    rust_map = {
        c: t
        for c, t in extract_rust_citations(args.root).items()
        if c.split("-")[0] in known_prefixes
    }
    waivers = {} if args.strict else load_waivers(args.root)

    unknown_citations = sorted((cited | set(rust_map)) - requirements.keys())
    unknown_waivers = sorted(set(waivers) - requirements.keys())
    uncovered = sorted(must_ids - cited - waivers.keys())
    if not requirements:
        errors.append(f"no requirement IDs found under {args.root / 'specs'}")
    ok = not (
        uncovered or unknown_citations or unknown_waivers or duplicates or parse_errors or errors
    )

    report = {
        "ok": ok,
        "strict": args.strict,
        "specs_range": args.specs,
        "requirements": len(requirements),
        "must_ids": sorted(must_ids),
        "covered": sorted(must_ids & cited),
        # deterministic MUST-ID -> citing-test mapping (audit finding 6)
        "coverage_map": {rid: citation_map[rid] for rid in sorted(must_ids & cited)},
        # checked for undefined IDs only; never MUST coverage (ADR-73)
        "rust_citations": {rid: rust_map[rid] for rid in sorted(rust_map)},
        "waived": sorted(set(waivers) & must_ids),
        "uncovered": uncovered,
        "unknown_citations": unknown_citations,
        "unknown_waivers": unknown_waivers,
        "duplicate_ids": sorted(duplicates),
        "parse_errors": parse_errors,
        "errors": errors,
    }
    print(json.dumps(report))
    if not ok:
        for rid in uncovered:
            print(f"uncovered MUST: {rid}: {requirements[rid][:80]}", file=sys.stderr)
        for rid in unknown_citations:
            print(f"test cites undefined ID: {rid}", file=sys.stderr)
        for rid in unknown_waivers:
            print(f"waiver for undefined ID: {rid}", file=sys.stderr)
        for rid in sorted(duplicates):
            print(f"duplicate requirement ID: {rid}", file=sys.stderr)
        for msg in parse_errors + errors:
            print(msg, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
