"""G1-C5: the in-process render warmup runs BEFORE the dora Node() is
created (source-level ordering pin).

The outer-loop pinned smoke (outer-smoke-g1, dora 48e96b5) proved the
failure mode this pins out: with the warmup AFTER Node(), the 11.2 s
compile blocked the event loop, >1100 10 ms tick events piled into the
zenoh channel, and the bridge DROPPED the first reset ("event channel
full") — episode 0 sat at "holding at sim step 0 until the first reset"
until the wall budget died. Pre-Node there is no subscription yet, so
nothing can pile up.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SOURCE = Path(__file__).resolve().parents[2] / "src" / "aisle" / "nodes" / "dora_genesis.py"


def test_render_warmup_happens_before_node_creation():
    """The warmup call must precede `node = Node()` in the bridge source:
    Node() opens the subscription; any blocking work after it starves the
    event loop and can drop the first reset (outer-smoke-g1 evidence)."""
    text = SOURCE.read_text()
    warmup_at = text.find("warmup_times = warmup_render(")
    node_at = text.find("node = Node()")
    assert warmup_at != -1 and node_at != -1, (warmup_at, node_at)
    assert warmup_at < node_at, (
        "render warmup must run BEFORE the dora Node() is created: after it, "
        "the compile blocks the event loop, tick events pile into the zenoh "
        "channel, and the first reset is dropped (G1-C5, outer-smoke-g1)"
    )


def test_warmup_channels_derive_without_the_node():
    """The hoisted block derives its channel set from env + cfg alone
    (rung_topic_rates + AISLE_TURN_OUTPUTS filter) — nothing between the
    scene build and the warmup call may need the node. Pinned by asserting
    the pre-Node region contains the local derivation, so a future move
    cannot silently reintroduce a node dependency."""
    text = SOURCE.read_text()
    node_at = text.find("node = Node()")
    pre_node = text[:node_at]
    assert "rung_topic_rates(" in pre_node, "channel derivation must live pre-Node"
    assert pre_node.count("_warmup_channels") >= 3, "the warmup block uses its own local channels"
    # and the node-dependent section never re-runs the warmup
    post_node = text[node_at:]
    assert "warmup_times = warmup_render(" not in post_node, "exactly one warmup call, pre-Node"
