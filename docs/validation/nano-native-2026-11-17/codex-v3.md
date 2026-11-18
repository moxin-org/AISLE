**V3 NAVIGATION: PASS. Local left-arm PICKUP: PASS. Dataset integrity: PASS for both.**

Each requested frozen-binary experiment ran once. Both independent results contain `ok: true` and `success: true`; both CLI exits were 0. Navigation took 101.545 wall seconds; pickup took 83.059 seconds. V1/V2 reports remain byte-for-byte unchanged.

| Measured result | Navigation, seed 7 | Local pickup, seed 0 |
|---|---:|---:|
| Judge samples / raw frames | 258 / 258 | 125 / 125 |
| Guarded motion commands | 257 | 124 |
| Sampled base path | 4.994548 m | 0.000079 m |
| Final base XY | (2.087959, 3.685367) m | (−0.000069, −0.000005) m |
| Base endpoint error | 0.018950 m | 0.000070 m |
| Furniture-contact frames / pair records | 0 / 0 | 0 / 0 |
| Delivery retries | 1, injected observation 5 | 0 |

Navigation first entered the existing 0.12 m goal radius at observation 234, source time 46.8 s. These measurements come from judge/raw base positions, which agree with the control recording. Path lengths are 5 Hz sampled lower bounds.

Pickup independently satisfies the physical criteria. Reconstructed raw contacts show **both left fingers touching the cube in the same physics step**, for 786 consecutive steps spanning 7.85 s. Maximum lift was **0.093034 m**; lift above 0.07 m with both-finger support persisted **2.6 s**. The final cube position was **(0.345338, 0.227920, 0.199999) m**, against destination (0.34, 0.23, 0.20) m. It remained stable inside the destination with **neither finger touching it for the final 4.4 s**, exceeding the required 1 s. No failed slip/drop was observed: gripper-opening velocity preceded contact loss at the destination. The subsequent return from peak height to table height is supported lowering and release, not an observed dropped-cube failure.

Dataset checks passed separately from those task outcomes. All **383 NPZ captures** match their stored SHA-256 values. Each camera retains 258 navigation and 125 pickup RGB/depth frames, with RGB uint8 640×480×3 and finite float32 depth 640×480. All **six videos decode without errors**, at 640×480 and 5 Hz: 258 frames/51.6 s or 125 frames/25.0 s. All three latest images from each run were visually inspected and exactly match raw frames 250 and 120 respectively. Pickup head/wrist views show the red cube on the table after release; navigation views show the room and robot foreground, with substantial self-occlusion.

Recorded observations and judge samples are contiguous, with source-clock increments of exactly 0.2 s and the documented +0.2 s replay-clock offset. The injected depth frame is retained; repeated observation-5 receipts recover its delivery. There are exactly **5,140 navigation and 2,480 pickup physics ticks**, 20 per advancing interval, without duplicates. All applied velocities match guard outputs; limits and base/arm exclusion pass. The sole motion route passes through the guard. Policy inputs exclude object truth and segmentation; simulator finished notifications carry observation IDs. Independent judge acknowledgements are wired into advancement.

Run-local policy/simulator binaries, input URDF, **all 137 referenced meshes**, asset manifest, scene, graph, Dora, and recorded policy Cargo.lock hashes match provenance in both runs. Each asset manifest covers 138 files including the URDF. Dataset copies of the independent result and provenance match their originals.

Limits: this is an experimental native simulation, not frozen benchmark or hardware acceptance. V3 changes both the contact auditor and navigation fixture: the station anchor is (1.6, −1.2), with general obstacle inflation 0.55 m and a pickup-table docking exception of 0.27 m. It is not a strict controller-only comparison with V1/V2. Pickup starts at goal (0,0), so **these runs do not establish integrated navigation-plus-pickup success**. Robot self-collision is disabled. Cube poses are recorded at 5 Hz; raw pairs support 100 Hz contact reconstruction, but GPU buffer bounds/separation values, individual delivery/acknowledgement packets, and applied motor torques are not independently recorded. URDF effort limits and 0.1 rad target-windup bounds remain simulation settings.

Full raw/video audit artifacts remain local with the retained run directories. No implementation edits, automatic experiment retries, commits, pushes, hardware control, or publication were performed.
