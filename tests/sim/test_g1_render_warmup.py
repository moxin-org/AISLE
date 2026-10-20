"""G1-C3 sim tests: in-process render warmup pays the compile, changes nothing.

- first-window parity: after warmup_render, the first REAL render pass
  costs the same order as the second one — the ~10.9 s/frame compile
  (the r3/r5 ep0 wall_clamp evidence) is paid at startup, inside the
  bridge process, not inside episode 0's wall budget. Robust to an
  already-warm pytest process: if the compile cache is warm the warmup
  itself is fast and the parity assertion holds trivially.
- determinism (CON-5): the warmup frames are DISCARDED — two same-seed
  scenes, one with warmup before the first observation, produce
  bit-identical rgb/joint_state/oracle_state.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None, reason="sim extra not installed"
    ),
]


def test_first_real_render_after_warmup_costs_the_same_as_the_second():
    from aisle.harness.simulator_work import work_context
    from aisle.nodes.dora_genesis import warmup_render
    from aisle.scenes.pharmacy import build_scene

    handle = build_scene(seed=11, embodiment="g1_right", headless=True)
    with work_context({}, dt_ns=10_000_000, n_envs=1) as work:  # _DirectCalls: no AISLE_SIM_WORK_* binding
        channels = ["rgb_overhead", "depth_overhead", "seg_overhead", "rgb_wrist"]
        times = warmup_render(handle.cams, channels, work)
        assert set(times) == {"overhead_s", "wrist_s"} and all(v >= 0.0 for v in times.values())

        import time

        t0 = time.perf_counter()  # episode 0's first real window, post-warmup
        work.call("render", handle.cams["overhead"].render, rgb=True, depth=True, segmentation=True)
        work.call("render", handle.cams["wrist"].render)
        first = time.perf_counter() - t0

        t0 = time.perf_counter()  # the next window
        work.call("render", handle.cams["overhead"].render, rgb=True, depth=True, segmentation=True)
        work.call("render", handle.cams["wrist"].render)
        second = time.perf_counter() - t0

    # Threshold basis: the compile-in-window form measured ~10.9 s/frame
    # (r3/r5 ep0 wall_clamp); a WARM frame is 0.02-0.4 s (sim_warmup
    # evidence). 1.5 s is ~4x the observed warm worst case and ~7x below
    # the cold first frame — comfortably separates the two regimes.
    assert first < 1.5, (first, second, times)
    assert second < 1.5, (first, second, times)
    assert first <= max(1.0, 3.0 * second), (first, second)  # same order, no compile cliff


def test_warmup_leaves_same_seed_observations_bit_identical():
    from aisle.harness.simulator_work import work_context
    from aisle.nodes.dora_genesis import warmup_render
    from aisle.scenes.pharmacy import build_scene, oracle_state, to_numpy

    def observe(warm: bool):
        handle = build_scene(seed=23, embodiment="g1_right", headless=True)
        if warm:
            with work_context({}, dt_ns=10_000_000, n_envs=1) as work:
                warmup_render(handle.cams, ["rgb_overhead", "rgb_wrist"], work)
        robot = handle.robot
        joints = to_numpy(robot.get_qpos()).reshape(-1).astype(np.float32).tobytes()
        oracle = np.asarray(oracle_state(handle), dtype=np.float32).tobytes()
        with work_context({}, dt_ns=10_000_000, n_envs=1) as work:
            rgb = np.asarray(
                work.call("render", handle.cams["overhead"].render, rgb=True)[0], dtype=np.uint8
            )
        return joints, oracle, rgb.tobytes()

    cold = observe(warm=False)
    warm = observe(warm=True)
    assert cold[0] == warm[0]  # joint_state bitwise (CON-5)
    assert cold[1] == warm[1]  # oracle_state bitwise (CON-5)
    assert cold[2] == warm[2]  # first rgb frame bitwise — the warmup frames were discarded
