"""Export a versioned, self-contained offline dataset (P1; CON-8).

All outcomes and excluded commands remain available for analysis. Training uses
only eligible successful episodes in the train split. No model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_dataset import export_dataset


def align_to_frames(action_ts: list[int], frame_ts: list[int]) -> list[int]:
    """Legacy arithmetic helper; P1 uses stricter, episode-scoped joins.

    For each action timestep, the index of the LATEST frame at or
    before it (the observation the policy would have acted on). Actions
    before the first frame get index -1 and are dropped by the caller —
    never paired with a future observation."""
    out, j = [], -1
    for t in action_ts:
        while j + 1 < len(frame_ts) and frame_ts[j + 1] <= t:
            j += 1
        out.append(j)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        help="JSON embodiment, ordered joints, units, frame and action semantics; "
        "absent means unknown",
    )
    parser.add_argument(
        "--endpoints", type=Path, help="JSON topic -> producer__topic.arrow for ambiguous streams"
    )
    args = parser.parse_args(argv)
    try:
        manifest = export_dataset(
            args.run,
            args.out,
            contract=json.loads(args.contract.read_text()) if args.contract else None,
            endpoints=json.loads(args.endpoints.read_text()) if args.endpoints else None,
        )
        result = {
            "ok": True,
            "dataset": str(args.out),
            "format_version": manifest["format_version"],
            "coverage": manifest["coverage"],
        }
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
