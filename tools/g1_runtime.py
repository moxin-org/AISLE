#!/usr/bin/env python3
"""Install Unitree SDK2 out-of-lock for the G1 driver and receipt it (ADR-g1-driver).

The G1 hardware path needs `unitree_sdk2py` (which pins `cyclonedds==0.10.2`,
setup.py@814556d1) — hardware-only dependencies must not enter the frozen
uv lock, so this tool installs them into the project environment with
`uv pip` (the ADR-67/ADR-68 out-of-lock pattern) and writes a gitignored
receipt naming the pinned commit. `read_receipt()` is the public provenance
reader `g1_driver` tooling and run manifests can carry; like every AISLE
CLI: JSON on stdout, logs to stderr, exit 0 iff ok (CON-8).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / ".g1-runtime-receipt.json"
#: unitreerobotics/unitree_sdk2_python, pinned (G1-D; the arm-SDK example
#: the driver cites line-by-line is readable at exactly this commit)
SDK2_REPO = "https://github.com/unitreerobotics/unitree_sdk2_python"
SDK2_COMMIT = "814556d15970dd2ecf1c9984e845ca02ab07e206"
#: unitree_sdk2py's own pin (setup.py install_requires@814556d1)
CYCLONEDDS_PIN = "0.10.2"


def receipt_path(root: Path | None = None) -> Path:
    """Where the SDK2 install receipt lives (default: this checkout)."""
    return (ROOT if root is None else Path(root)) / RECEIPT.name


def read_receipt(root: Path | None = None) -> dict:
    """The G1 SDK2 install provenance for run manifests to carry.

    The receipt is gitignored and local to the machine that installed, so
    recording it is the only trace of which SDK2 sources a hardware run
    used. Never raises: an absent or malformed receipt is a reported fact,
    not an exception. Returns {"installed": bool, "path": str,
    "receipt": dict | None, "problem": str | None}."""
    path = receipt_path(root)
    result: dict = {"installed": False, "path": str(path), "receipt": None, "problem": None}
    if not path.is_file():
        result["problem"] = f"no g1 runtime receipt at {path}; run `tools/g1_runtime.py install`"
        return result
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result["problem"] = f"unreadable g1 runtime receipt at {path}: {exc}"
        return result
    if not isinstance(receipt, dict):
        result["problem"] = f"malformed g1 runtime receipt at {path}: expected a JSON object"
        return result
    return {"installed": True, "path": str(path), "receipt": receipt, "problem": None}


def _dist_version(dist: str) -> str:
    return importlib.metadata.version(dist)


def install(python: Path) -> dict:
    """Out-of-lock install of the pinned SDK2 + cyclonedds via `uv pip`."""
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        f"git+{SDK2_REPO}@{SDK2_COMMIT}",
        f"cyclonedds=={CYCLONEDDS_PIN}",
    ]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{proc.stderr.strip()}")
    return verify(python, write=True)


def verify(python: Path, *, write: bool = False) -> dict:
    """Check the environment against the pins; `write` records a fresh
    receipt (the install path), otherwise an existing receipt is required."""
    version = _dist_version("unitree_sdk2py")
    cyclonedds = _dist_version("cyclonedds")
    if cyclonedds != CYCLONEDDS_PIN:
        raise ValueError(
            f"installed cyclonedds {cyclonedds} differs from the pin {CYCLONEDDS_PIN} "
            "(unitree_sdk2py setup.py@814556d1)"
        )
    receipt = {
        "schema_version": 1,
        "sdk2_commit": SDK2_COMMIT,
        "sdk2_repo": SDK2_REPO,
        "version": version,
        "cyclonedds": cyclonedds,
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }
    if write:
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    read = read_receipt()
    if not read["installed"]:
        raise FileNotFoundError(read["problem"])
    recorded = read["receipt"]
    for key in ("sdk2_commit", "version"):
        if recorded.get(key) != receipt[key]:
            raise ValueError(
                f"installed {key} {receipt[key]!r} differs from the receipt's {recorded.get(key)!r}"
            )
    return recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="out-of-lock install of the pinned unitree_sdk2py")
    inst.add_argument("--python", type=Path, default=Path(sys.executable))
    ver = sub.add_parser("verify", help="check the install against the receipt/pins")
    ver.add_argument("--python", type=Path, default=Path(sys.executable))
    sub.add_parser("receipt", help="print the install provenance a run manifest records")
    args = parser.parse_args(argv)
    if args.command == "receipt":
        read = read_receipt()
        print(json.dumps({"ok": read["installed"], **read}))
        if read["problem"]:
            print(read["problem"], file=sys.stderr)
        return 0 if read["installed"] else 1
    try:
        if args.command == "install":
            receipt = install(args.python)
        else:
            receipt = verify(args.python)
    except (
        RuntimeError,
        FileNotFoundError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **receipt}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
