**V4 physics acceptance: PASS for navigation and integrated left-arm navigation+pickup. Experiment completion: PASS for both.**

Each frozen-binary experiment ran once. Both independent verdicts and launchers report success; both Dora processes exited normally, with no interrupt recorded. The requested Rust evidence acceptance test passed once against each run: **2 tests passed, 0 failed**. Other unit/Genesis suites were not repeated. All previous runs and reports were preserved.

| Measured result | Navigation, seed 7 | Integrated left pickup, seed 0 |
|---|---:|---:|
| Base goal | (2.1, 3.7) m | (1.6, −1.2) m |
| Judge samples / raw frames | 530 / 530 | 373 / 373 |
| Guarded commands | 529 | 372 |
| Sampled base path | 4.987089 m | 2.164020 m |
| Final base XY | (2.089183, 3.668126) m | (1.573168, −1.194764) m |
| Endpoint error | 0.033659 m | 0.027338 m |
| Furniture-contact frames / pair records | 0 / 0 | 0 / 0 |
| Wall duration | 243.017 s | 146.406 s |
| Delivery retries | 2, including injected omission | 0 |

Motion is verified from matching raw/judge positions, not controller completion. The integrated base passed within 59.2 mm of the pre-dock waypoint (0.9, −1.2), then moved +0.669 m in X and −0.054 m in Y to its endpoint. Terminal stopped-arrival criteria held for 11.0 s and 28.4 s respectively. All wheel commands were zero for the final 3.0 s / 26.8 s; final base displacement per 0.2 s was 0.715 µm / 2.744 µm, with orientation increments below 4.5×10⁻⁵ rad. Wheel targets independently reconstruct from measured angles and the float32 integration timestep, with residuals below 2.85×10⁻¹³ rad.

Pickup meets the stronger physical criteria. Raw pairs reconstruct both fingers contacting the cube in the **same physics step**; maximum lift was **0.097491 m**, with supported lift above 0.07 m lasting **5.4 s**. Final cube position was **(1.917974, −0.971156, 0.199999) m**, inside the destination tolerance around (1.94, −0.97, 0.20) m. Placement remained stable for the final **7.0 s with zero contact from ANY robot link**, and held at termination. Opening commands preceded contact loss at the destination; no failed slip/drop was observed. These measurements establish this integrated left-arm case, unlike the prior local-origin pickup.

Dataset checks passed, with the numeric metadata exception below retained. All **903 NPZ hashes** match their audits. Each of the three cameras contains 530 navigation or 373 integrated RGB/depth frames, with uint8 RGB 640×480×3 and finite float32 depth 640×480. All **six videos decode without errors**, at 5 Hz: 530 frames/106.0 s or 373 frames/74.6 s. All latest camera images were inspected and exactly match raw frames 520 and 370. Pickup views show the red cube resting on the table after release.

Observation IDs, source timestamps and judge samples are contiguous. Source clocks advance exactly 0.2 s; replay clocks retain their documented +0.2 s offset. There are exactly **10,580 / 7,440 distinct physics ticks**, 20 per advancing interval, without duplication. The omitted navigation depth delivery retains its raw frame and recovers through observation retries. Every applied command matches the guard; all motion passes through it. Policy inputs exclude object truth and segmentation. ANY-robot contact flags are independently reconstructed from raw pairs and match the stronger judge.

Run-local executable, URDF, **137 referenced mesh**, asset-manifest, scene, graph, Dora and recorded build-lock hashes verify. Provenance settings exactly match simulator scene-info, including actual URDF actuator effort limits. Capture metadata correctly uses `success: null`, `capture_complete: true`; physical results are separate. The copied integrated maximum-lift field differs from the source by **one ULP (1.39×10⁻¹⁷ m)**; all other verdict fields match exactly, and raw recomputation supports the same result. Initial audit diagnostics are retained, including the corrected float32-timestep calculation.

Limits: two experimental simulation cases, not frozen benchmark or hardware certification. Navigation uses a known map and simulator localization; robot self-collision is disabled. Poses are sampled at 5 Hz and contacts at 100 Hz; sub-frame object height, motor torque readback, GPU contact-buffer bounds and individual delivery/receipt packets are not retained.

Full JSON, video and Rust-test audit artifacts remain local. No implementation edits, commits, publishing, hardware control or automatic experiment retries occurred.
