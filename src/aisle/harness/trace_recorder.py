"""Trace recorder node (SPEC 070 HAR-4): Arrow IPC traces of every wired
topic plus the overhead video.

EVERY wired topic becomes $AISLE_TRACE_DIR/<topic>.arrow in Arrow IPC
STREAM format — readable batch-by-batch even if the recorder dies before
a clean close (the FILE format needs a footer, and a SIGKILL'd recorder
left unreadable truncated files in the T09 smoke). Numeric payloads fill
the data column, JSON payloads the text column, image topics record
payload-free rows with pixels in overhead.mp4 (10 fps; ADR-11) — except
seg_overhead rows, whose text carries mask PROVENANCE (#131). SIGTERM
is handled so teardown flushes writers. Measurement only: this node runs in
the rollout runner's INSTRUMENTED copy of the graph, which is never the
graph that the HAR-2 validation gate checks — VAL-6's oracle isolation
governs the composed graph, not the harness's own recording (ADR-11).

$AISLE_FRAME_CAPTURE_PERIOD_S (default 0 = off) additionally persists RAW
pixels as `frames/<camera>/<sim_time_ns>.npz` at VER-9's judged-frame
cadence — ADR-11 clause 14. `CaptureSchedule` reproduces
`checkpoint_stamps`' choice online: last frame at or before each
boundary, counted from each episode's goal receipt. The last complete
RGB/depth pair is retained separately so an intervening 30 Hz RGB-only tick
cannot evict it before a 15 Hz pair crosses the boundary (issue #136). The mp4
is lossy, 10 fps and carries no depth at all, so a run recorded without
this captures NOTHING the realistic verifier can replay (VER-5/VER-6):
its containment and upright stages need the overhead depth, and byte
equality is what makes a replay a replay. Capture is opt-in because it is
expensive: 140 KB per instant measured over a live two-episode run
(480x640 RGB + depth, 240x320 wrist), so a 600 s episode at the VER-9
5 s cadence costs ~17 MB.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

TRACE_SCHEMA = pa.schema(
    [
        ("sim_time_ns", pa.int64()),
        ("env_id", pa.int32()),
        ("seq", pa.int64()),
        # at most one of data/text is non-null per row: numeric payloads
        # in data, JSON payloads in text; image rows carry no payload
        # (pixels live in the mp4, ADR-11) except seg_overhead, whose text
        # is the #131 provenance JSON — never the mask itself
        ("data", pa.list_(pa.float64())),
        ("text", pa.string()),
    ]
)
# rows buffered per topic before a batch is written: one batch per message
# costs ~300 bytes of IPC framing per ~40-byte row
BATCH_ROWS = 100
# ALL buffers also flush on this wall cadence: low-rate endpoints (an
# episode_result has ~2 rows per run) never reach BATCH_ROWS, and the
# SIGTERM exit-flush cannot be relied on inside dora's blocking event
# read (PR #11 round 2) — periodic flushing bounds any loss to this many
# seconds of tail
FLUSH_EVERY_S = 5.0
# endpoints whose rows are METADATA-ONLY. Anything not listed here goes down the
# generic numeric path, which calls .tolist() on the payload: a 640x480
# seg_overhead mask is 307,200 float64 list values per frame, ~2.5 MB, ~37 MB/s
# at 15 Hz, buffered BATCH_ROWS deep. (Earlier revisions of this comment cited
# "ADR-11's ~17 MB budget" — that figure is ADR-11's MEASUREMENT of the opt-in
# npz capture set for a 600 s episode, as this module's own docstring states,
# not a trace budget. The arithmetic above is what justifies the routing.)
IMAGE_ENDPOINTS = ("__rgb_overhead", "__rgb_wrist", "__depth_overhead", "__seg_overhead")

# Low-rate identity evidence. Preserve the existing Arrow schema for historical
# readers; the sidecar binds these metadata to the exact captured row instead.
IDENTITY_TOPICS = {
    "reset_done",
    "episode_goal",
    "episode_result",
    "target_request",
    "read_move",
    "move_done",
    "plan_done",
}


def trace_row_digest(row: dict) -> str:
    """HAR-4: hash the five stored Arrow fields, independent of metadata extras."""
    captured = {name: row[name] for name in TRACE_SCHEMA.names}
    return hashlib.sha256(
        json.dumps(captured, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def trace_identity(endpoint: str, index: int, row: dict, metadata: dict) -> dict:
    """TC-6/7: retain observed correlation IDs; never synthesize a missing ID."""
    return {
        "format": "aisle_trace_identity",
        "version": 1,
        "endpoint": endpoint,
        "row": index,
        "trace_row_sha256": trace_row_digest(row),
        "metadata": {key: metadata[key] for key in ("goal_id", "request_id") if key in metadata},
    }


def is_image_endpoint(topic: str) -> bool:
    """Whether this endpoint's row is PAYLOAD-FREE (ADR-11, TC-9; seg rows
    additionally carry #131 provenance text, never the mask itself).

    A named function, not an inline `topic.endswith(IMAGE_ENDPOINTS)`, so the
    routing decision itself is what a test binds: asserting only on the
    constant left `IMAGE_ENDPOINTS[:-1]` at the call site green while masks went
    back down the numeric path."""
    return topic.endswith(IMAGE_ENDPOINTS)


def decode_frame(metadata: dict, value) -> np.ndarray | None:
    """The RAW pixels of an image payload, or None when the metadata does
    not describe a frame. `enc` is the bridge's own declaration (BRG-2),
    never inferred from the topic name: a depth stream decoded as uint8
    would be silently wrong rather than absent."""
    h, w = int(metadata.get("h", 0)), int(metadata.get("w", 0))
    if not (h and w):
        return None
    enc = metadata.get("enc")
    if enc == "rgb8":
        return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.uint8).reshape(h, w, 3)
    if enc == "depth32f":
        return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(h, w)
    if enc == "seg_i32":
        # issue #131, amending #129's decline: at L1 the mask IS the pose's
        # determining input, so it decodes — into the VER-9 capture set and
        # the per-row provenance hash. The mp4 stays rgb-only via the
        # stream-name guard in main().
        return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.int32).reshape(h, w)
    return None


def seg_provenance(frame: np.ndarray) -> str:
    """JSON for a mask's trace row (issue #131 option 2): a sha256 and the
    labeled-pixel count — enough to prove WHICH mask produced an L1
    target_pose (or refusal) at ~100 bytes, where the payload itself is
    ~1.2 MB per frame and stays out of the trace (ADR-11 budget).

    `nonzero_px` counts every labeled pixel in the SCENE — it is not the
    estimator's `mask_pixels`, which counts the target's mask alone; the
    sha256 is the cross-artifact anchor, the count a coarse sanity bound."""
    return json.dumps(
        {
            "mask_sha256": hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
            "nonzero_px": int(np.count_nonzero(frame)),
        }
    )


class CaptureSchedule:
    """VER-9's judged-frame selector, run online.

    `checkpoint_stamps()` snaps each wanted stamp to the nearest rendered
    frame AT OR BEFORE it, and counts checkpoints from GOAL RECEIPT. A
    recorder cannot see future frames, so it reproduces the same choice by
    retaining the newest capture-ready frame set and writing it once a later
    frame proves the boundary has passed. Both details are what make the
    persisted set the judged set rather than merely a periodic sample
    (PR #105 review, issue #136):

    * at-or-BEFORE: with renders at 4.967 s and 5.033 s around a 5.000 s
      checkpoint, the verifier judges 4.967 s — writing the current frame
      when the boundary is crossed would persist 5.033 s and no byte
      comparison (VER-7) or replay (VER-6) could hold;
    * goal-RELATIVE: a process-global schedule phase-shifts every episode
      that does not start on a boundary, so `start()` re-bases it on each
      `episode_goal` stamp.
    """

    def __init__(self, period_ns: int):
        self.period_ns = int(period_ns)
        self.next_boundary_ns: int | None = None

    @property
    def enabled(self) -> bool:
        """Capture is off entirely at period 0 (HAR-4 records the mp4
        either way)."""
        return bool(self.period_ns)

    def start(self, goal_sim_time_ns: int) -> None:
        """Re-base the schedule on an episode's goal receipt (VER-9)."""
        if self.enabled:
            self.next_boundary_ns = int(goal_sim_time_ns)

    def crossed(self, sim_time_ns: int) -> bool:
        """True when `sim_time_ns` proves the pending boundary has passed,
        so the RETAINED frame is the last one at or before it. A frame
        landing exactly ON the boundary is itself the right frame, so it is
        retained rather than triggering — the next frame writes it."""
        return (
            self.enabled
            and self.next_boundary_ns is not None
            and sim_time_ns > self.next_boundary_ns
        )

    def advance(self, sim_time_ns: int) -> None:
        """Step past every boundary this frame has overtaken. Skipped
        boundaries are gaps in the render stream, not recoverable frames."""
        if not self.enabled or self.next_boundary_ns is None:
            return
        while sim_time_ns > self.next_boundary_ns:
            self.next_boundary_ns += self.period_ns


def capture_frames(frames_dir: Path, retained: dict[str, tuple[int, np.ndarray]]) -> int | None:
    """Persist the retained payloads as one npz per camera, keyed by the
    frame's own sim stamp, and return the overhead stamp written.

    Returns None — capturing NOTHING — unless the overhead rgb and depth
    share a stamp: the verifier's geometry back-projects the two together
    (VER-10/VER-11), so a pair from different ticks is worse than a gap.
    BRG-2 renders both in one pass whenever both are due, so a mismatch
    means the frames did not come from one render."""
    rgb, depth = retained.get("rgb_overhead"), retained.get("depth_overhead")
    if rgb is None or depth is None or rgb[0] != depth[0]:
        return None
    arrays = {"rgb": rgb[1], "depth": depth[1]}
    seg = retained.get("seg_overhead")
    # issue #131: at L1 the mask rides with the judged pair — but only from
    # the SAME render pass (the TC-9 stamp rule); a mask from another tick
    # would attest a scene this pair never saw. L0 runs have no seg stream
    # and the npz is unchanged.
    if seg is not None and seg[0] == rgb[0]:
        arrays["seg"] = seg[1]
    write_camera_frame(frames_dir, "overhead", rgb[0], arrays)
    wrist = retained.get("rgb_wrist")
    if wrist is not None:
        write_camera_frame(frames_dir, "wrist", wrist[0], {"rgb": wrist[1]})
    return rgb[0]


def retain_capture_frame(
    latest: dict[str, tuple[int, np.ndarray]],
    ready: dict[str, tuple[int, np.ndarray]],
    stream: str,
    sim_time_ns: int,
    frame: np.ndarray,
) -> None:
    """Retain one stream and preserve the newest CAPTURE-READY frame set.

    RGB runs at 30 Hz while overhead depth runs at 15 Hz. A faster RGB-only
    tick therefore makes ``latest`` temporarily mismatched; issue #136 showed
    that using it directly destroys the complete pair needed by the next
    at-or-before checkpoint. ``ready`` advances only when RGB and depth share
    a stamp. Wrist RGB remains independent, and segmentation joins only the
    overhead pair from its own render pass."""
    retained = (int(sim_time_ns), frame)
    latest[stream] = retained
    if stream == "rgb_wrist":
        ready[stream] = retained

    rgb, depth = latest.get("rgb_overhead"), latest.get("depth_overhead")
    if rgb is None or depth is None or rgb[0] != depth[0]:
        return
    ready["rgb_overhead"] = rgb
    ready["depth_overhead"] = depth
    seg = latest.get("seg_overhead")
    if seg is not None and seg[0] == rgb[0]:
        ready["seg_overhead"] = seg
    else:
        ready.pop("seg_overhead", None)


def record_image_frame(
    stream: str,
    sim_time_ns: int,
    frame: np.ndarray,
    schedule: CaptureSchedule,
    latest: dict,
    ready: dict,
    frames_dir: Path,
) -> str | None:
    """Retention-order and boundary-capture logic for ONE image frame;
    returns the provenance text for seg rows (None otherwise). Extracted
    from the event loop so the ORDER is what a test drives with the real
    30/15 Hz interleave (PR #134 review P1/P2).

    The order is load-bearing and asymmetric:

    * rgb/depth retain AFTER the boundary check — capture uses the separate
      last capture-ready pair, which a faster RGB-only tick cannot destroy
      (issue #136; VER-9's at-or-before rule);
    * seg retains BEFORE it. If no pre-boundary pair exists yet, a seg event
      can be the event that makes the first post-boundary capture complete;
      retaining first includes that pair's same-render mask. When a prior
      capture-ready pair exists, the earlier rgb event already captured it
      and advanced the schedule before any post-boundary payload replaced it."""
    row_text = None
    if stream == "seg_overhead":
        retain_capture_frame(latest, ready, stream, sim_time_ns, frame)
        row_text = seg_provenance(frame)
    if schedule.crossed(sim_time_ns):
        if capture_frames(frames_dir, ready) is not None:
            schedule.advance(sim_time_ns)
    if stream != "seg_overhead":
        retain_capture_frame(latest, ready, stream, sim_time_ns, frame)
    return row_text


def write_camera_frame(
    frames_dir: Path, camera: str, sim_time_ns: int, arrays: dict[str, np.ndarray]
) -> Path:
    out = frames_dir / camera
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sim_time_ns:020d}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_frames(trace_dir: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """`frames[camera][sim_time_ns] -> {"rgb", "depth"[, "seg"]}` — the
    mapping `verifier.realistic.judge_frames` consumes, so an offline
    replay (VER-6) reads the same arrays the live judge saw. Empty when
    the run was recorded without frame capture."""
    frames: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for camera_dir in sorted((Path(trace_dir) / "frames").glob("*")):
        if not camera_dir.is_dir():
            continue
        per_stamp = {}
        for path in sorted(camera_dir.glob("*.npz")):
            with np.load(path) as data:
                per_stamp[int(path.stem)] = {name: data[name] for name in data.files}
        if per_stamp:
            frames[camera_dir.name] = per_stamp
    return frames


def main() -> None:
    import imageio.v2 as imageio
    from dora import Node

    trace_dir = Path(os.environ["AISLE_TRACE_DIR"])
    trace_dir.mkdir(parents=True, exist_ok=True)

    schema = TRACE_SCHEMA
    writers: dict = {}
    buffers: dict[str, list] = {}
    row_counts: dict[str, int] = {}

    def flush(topic: str) -> None:
        rows = buffers.get(topic)
        if not rows:
            return
        if topic not in writers:
            writers[topic] = pa.ipc.new_stream(trace_dir / f"{topic}.arrow", schema)
        writers[topic].write_batch(
            pa.record_batch(
                [
                    pa.array([r[0] for r in rows], pa.int64()),
                    pa.array([r[1] for r in rows], pa.int32()),
                    pa.array([r[2] for r in rows], pa.int64()),
                    pa.array([r[3] for r in rows], pa.list_(pa.float64())),
                    pa.array([r[4] for r in rows], pa.string()),
                ],
                schema=schema,
            )
        )
        buffers[topic] = []

    def buffer_row(topic, metadata, data, text) -> None:
        row = (
            int(metadata.get("sim_time_ns", 0)),
            int(metadata.get("env_id", 0)),
            int(metadata.get("seq", 0)),
            data,
            text,
        )
        index = row_counts.get(topic, 0)
        row_counts[topic] = index + 1
        if topic.rsplit("__", 1)[-1] in IDENTITY_TOPICS:
            evidence = trace_identity(
                topic, index, dict(zip(schema.names, row, strict=True)), metadata
            )
            identities.write(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
            identities.flush()
        buffers.setdefault(topic, []).append(row)
        if len(buffers[topic]) >= BATCH_ROWS:
            flush(topic)

    video = None
    frame_shape: tuple[int, int] | None = None
    last_flush = time.monotonic()

    schedule = CaptureSchedule(
        int(float(os.environ.get("AISLE_FRAME_CAPTURE_PERIOD_S", "0")) * 1e9)
    )
    frames_dir = trace_dir / "frames"
    latest: dict[str, tuple[int, np.ndarray]] = {}
    ready: dict[str, tuple[int, np.ndarray]] = {}

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # run finally
    node = Node()
    identities = (trace_dir / "identities.jsonl").open("w")
    try:
        for event in node:
            now = time.monotonic()
            if now - last_flush > FLUSH_EVERY_S:
                last_flush = now
                for pending in list(buffers):
                    flush(pending)
            if event["type"] != "INPUT":
                continue
            topic = event["id"]  # <producer>__<topic> endpoint key
            metadata = event.get("metadata") or {}
            if is_image_endpoint(topic):
                # image endpoints: payload-free rows; overhead pixels go to
                # the mp4 and, when capture is on, raw arrays (ADR-11).
                # seg_overhead (TC-9, L1) is routed here or nowhere: the
                # generic numeric path below would call .tolist() on 640x480 =
                # 307,200 float64s per frame, ~2.5 MB, ~37 MB/s at 15 Hz.
                # Issue #131: its row carries PROVENANCE (hash + labeled px)
                # in text, and the decoded mask joins the capture set — the
                # mp4 stays rgb-only via the stream guard below.
                stream = topic.rsplit("__", 1)[-1]
                frame = decode_frame(metadata, event["value"])
                row_text = None
                if frame is not None:
                    sim_time_ns = int(metadata.get("sim_time_ns", 0))
                    # retention order + boundary capture live in
                    # record_image_frame — the order is load-bearing and
                    # test-driven (PR #134 review)
                    row_text = record_image_frame(
                        stream, sim_time_ns, frame, schedule, latest, ready, frames_dir
                    )
                    if stream == "rgb_overhead":
                        h, w = frame.shape[0], frame.shape[1]
                        if video is None:
                            video = imageio.get_writer(
                                trace_dir / "overhead.mp4", fps=10, macro_block_size=1
                            )
                            frame_shape = (h, w)
                        if (h, w) == frame_shape:
                            video.append_data(frame)
                buffer_row(topic, metadata, None, row_text)
                continue
            value = event["value"]
            if pa.types.is_string(value.type) or pa.types.is_large_string(value.type):
                # JSON payloads fill the text column of the same trace
                buffer_row(topic, metadata, None, value[0].as_py())
                if schedule.enabled and topic.endswith("__episode_goal"):
                    # VER-9 counts checkpoints from GOAL RECEIPT, so the
                    # schedule is re-based per episode rather than global
                    schedule.start(int(metadata.get("sim_time_ns", 0)))
                if schedule.enabled and topic.endswith("__episode_result"):
                    # the TERMINAL frame is always judged (VER-9), and it is
                    # the one a mid-period episode end would otherwise drop
                    capture_frames(frames_dir, ready)
                continue
            values = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1)
            buffer_row(topic, metadata, values.tolist(), None)
    finally:
        identities.close()
        for topic in list(buffers):
            flush(topic)
        for writer in writers.values():
            writer.close()
        if video is not None:
            video.close()


if __name__ == "__main__":
    main()
