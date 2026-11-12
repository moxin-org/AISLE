"""finetune_smolvla — the pre-registered MPS LoRA fallback experiment
(analysis/m1/finetune_protocol.md; owner-approved item 1).

Trains LoRA adapters on SmolVLA's action head from the exported
demonstration tuples (guard-clamped actions, causal frame pairing —
tools/demo_export.py) and saves the adapter for the eval graph's
backend to load via AISLE_VLA_ADAPTER. Single config, no sweep (the
protocol's rule: the first contrast is tuned-vs-zero-shot). CON-8:
JSON to stdout, logs to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def build_frame_cache(dataset: Path, frame_paths: set):
    """Read only verified lossless RGB assets; no video seek or size-only cache."""
    import numpy as np
    from demo_dataset import read_dataset

    manifest = read_dataset(dataset)["manifest"]
    out = {}
    for name in sorted(frame_paths):
        if name not in manifest["artifacts_sha256"]:
            raise ValueError(f"unverified frame asset: {name}")
        with np.load(dataset / name, allow_pickle=False) as data:
            out[name] = data["rgb"].copy()
    return out


def load_tuples(dataset: Path, run_dir: Path | None, stride: int, limit: int | None):
    """Versioned SO-101 view. run_dir is retained only for caller compatibility."""
    from demo_dataset import SO101_CONTRACT, read_dataset, training_samples, valid_vector

    if stride < 1 or (limit is not None and limit < 1):
        raise ValueError("stride and limit must be positive")
    data = read_dataset(dataset)
    manifest = data["manifest"]
    if manifest["embodiment"] != "so101":
        raise ValueError("SO-101 trainer refuses other embodiments, including Franka")
    if manifest["contract"] != SO101_CONTRACT:
        raise ValueError("SO-101 training requires the supported explicit action contract")
    selected = training_samples(data["episodes"])[::stride]
    if selected:
        if "stats.json" not in manifest["artifacts_sha256"]:
            raise ValueError("training statistics missing from artifact hashes")
        stats = json.loads((dataset / "stats.json").read_text())
        for key in ("state_mean", "state_std", "action_mean", "action_std"):
            values = stats.get(key) if isinstance(stats, dict) else None
            if (
                not isinstance(values, list)
                or len(values) != 6
                or not all(
                    isinstance(x, (float, int))
                    and math.isfinite(x)
                    and (not key.endswith("std") or x > 0)
                    for x in values
                )
            ):
                raise ValueError(f"invalid training statistics: {key}")
    out = []
    for sample in selected[:limit]:
        if not all(
            valid_vector(sample[name], width)
            for name, width in (("forwarded", 6), ("state", 6), ("gripper", 1))
        ):
            raise ValueError("invalid SO-101 vector dimensions or values")
        out.append(
            {
                "frame_path": sample["pixels"]["path"],
                "state6": sample["state"]["data"],
                "action": sample["forwarded"]["data"][:5] + sample["gripper"]["data"],
                "task": sample["instruction"]["text"],
                "source_command": sample["forwarded"],
                "goal_revision": sample["instruction"]["revision"],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="deprecated: datasets now bundle their sources")
    parser.add_argument("--out", type=Path, required=True, help="adapter output dir")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument(
        "--checkpoint-at",
        default="",
        help="comma-separated step numbers to dump checkpoints at (ADR-45 dose ladder)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint.pt to resume from")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Reject legacy/incompatible/incomplete datasets BEFORE importing a model.
    try:
        tuples = load_tuples(args.dataset, args.run, args.stride, args.limit)
        if not tuples:
            raise ValueError("no eligible training tuples")
        stats = json.loads((args.dataset / "stats.json").read_text())
        frame_cache = build_frame_cache(args.dataset, {row["frame_path"] for row in tuples})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aisle.nodes.vla_backend import MODEL_ID, PINNED_REVISION, load_smolvla

    device = "cpu"  # MPS crashes at load (backend fence); CPU fallback per protocol
    print(f"[finetune] device={device}", file=sys.stderr)
    policy = load_smolvla()
    mods = dict(policy.named_modules())
    with torch.no_grad():
        si = mods["normalize_inputs.buffer_observation_state"]
        si.mean.copy_(torch.tensor(stats["state_mean"]))
        si.std.copy_(torch.tensor(stats["state_std"]))
        for name in ("normalize_targets.buffer_action", "unnormalize_outputs.buffer_action"):
            am = mods[name]
            am.mean.copy_(torch.tensor(stats["action_mean"]))
            am.std.copy_(torch.tensor(stats["action_std"]))
    print("[finetune] stats injected", file=sys.stderr)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            n for n, m in policy.named_modules() if type(m).__name__ == "Linear" and "expert" in n
        ]
        or "all-linear",
        lora_dropout=0.05,
    )
    model = get_peft_model(policy, lora)
    model.to(device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[finetune] trainable params: {trainable:,}", file=sys.stderr)

    print(f"[finetune] tuples: {len(tuples)}", file=sys.stderr)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    rng = np.random.default_rng(0)
    losses = []
    start_step = 0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        losses = ck["losses"]
        start_step = ck["step"]
        print(f"[finetune] resumed at step {start_step}", file=sys.stderr)
    dumps = sorted(int(s) for s in args.checkpoint_at.split(",") if s.strip())

    def dump_checkpoint(step: int) -> None:
        ckdir = args.out / f"checkpoint_{step}"
        ckdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckdir)
        import shutil as _sh

        _sh.copy(args.dataset / "stats.json", ckdir / "stats.json")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "rng": rng.bit_generator.state,
                "losses": losses,
                "step": step,
            },
            args.out / "checkpoint.pt",
        )
        print(f"[finetune] checkpoint at step {step} -> {ckdir}", file=sys.stderr)

    for step in range(start_step, args.steps):
        row = tuples[int(rng.integers(len(tuples)))]
        frame = frame_cache[row["frame_path"]]
        import torch.nn.functional as F

        img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
        width = len(row["action"])
        chunk = torch.tensor([row["action"]], device=device).unsqueeze(1)
        chunk = chunk.expand(1, getattr(model.config, "chunk_size", 50), width)
        batch = {
            "observation.state": torch.tensor([row["state6"]], device=device),
            "observation.images.camera1": img.to(device),
            "observation.images.camera2": torch.zeros(1, 3, 256, 256, device=device),
            "observation.images.camera3": torch.zeros(1, 3, 256, 256, device=device),
            "action": chunk.contiguous(),
            "task": [row["task"]],
        }
        out = model.forward(batch)
        loss = out.get("loss") if isinstance(out, dict) else out[0]
        if not torch.is_tensor(loss):
            print(json.dumps({"ok": False, "error": f"no loss in forward output {type(out)}"}))
            return 1
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 50 == 0:
            print(f"[finetune] step {step}: loss {float(loss):.4f}", file=sys.stderr)
        if (step + 1) in dumps:
            dump_checkpoint(step + 1)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    (args.out / "training_record.json").write_text(
        json.dumps(
            {
                "base": {"id": MODEL_ID, "revision": PINNED_REVISION},
                "steps": args.steps,
                "tuples": len(tuples),
                "loss_first": losses[0] if losses else None,
                "loss_last": sum(losses[-20:]) / max(1, len(losses[-20:])),
                "device": device,
                "lr": args.lr,
                "target_semantics": "single_forwarded_target_repeated_across_chunk",
                "dataset_format_version": 1,
            }
        )
    )
    print(
        json.dumps(
            {
                "ok": True,
                "adapter": str(args.out),
                "loss_first": losses[0],
                "loss_last_20_mean": sum(losses[-20:]) / max(1, len(losses[-20:])),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
