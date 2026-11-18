**READ-ONLY AUDIT: PASS under the current stricter judge. Recording integrity and original experiment completion also pass.**

Audited the retained right-arm, seed-0 episode at navigation goal (2.3, 3.9) m. No simulation was executed, and neither motion nor dataset was modified. All earlier reports remain unchanged.

- **Actual navigation:** 756 judge samples and raw frames; **5.933047745 m** sampled base travel. Final base XY **(2.269594, 3.899353) m**, endpoint error **0.030413 m**. Furniture-contact frames and raw furniture-contact pair records: **0**. Terminal stopped arrival held for 27.4 s; wheel commands were zero for the final 27.0 s. Final displacement per 0.2 s was **1.53 µm**, with angular change **7.92×10⁻⁶ rad**.
- **Physical pickup:** independently reconstructed same-step contact from both right fingers in 63 observations. Maximum cube lift: **0.093743876 m**; supported lift above 0.07 m lasted **5.2 s**, exceeding 0.4 s.
- **Stricter placement-entry check:** first entry into the placement zone after qualified pickup was observation **711**, source time **142.2 s**, with both-finger support. Supported zone observations continued through 718. Thus this episode satisfies the added protection against an unsupported drop into the destination.
- **Terminal release:** final cube XYZ **(2.634841, 3.670989, 0.199999) m**, inside the destination tolerance around (2.64, 3.67, 0.20) m. It remained stable with **no contact from ANY robot link** over observations **734–755**, the final **4.2 s**. Earlier loss of two-finger support did not count as release while another robot contact remained. No failed slip/drop was observed.

All **756 NPZ hashes** match their audits. Head and both wrists each retain 756 uint8 RGB arrays at 640×480×3 and finite float32 depth arrays at 640×480. All three videos decode without errors: **756 frames each, 640×480, 5 Hz, 151.2 s**. All latest images were inspected and exactly match raw observation 750; head/right-wrist views show the red cube on the table beneath the raised/open gripper.

Source timestamps and observation IDs are contiguous, with exact 0.2 s increments and the documented +0.2 s replay offset. All **15,100 distinct physics ticks** are covered, 20 per advancing interval, without duplicated capture intervals. All **755 applied commands** match guard outputs; motion passes through the guard and policy inputs exclude object truth/segmentation. Delivery retries: **0**. Run-local binaries, URDF, all **137 referenced meshes**, asset manifest, scene, graph and Dora hashes verify. Simulator settings match provenance. Capture completion is separate from task success; logs support normal Dora exit, with a retained wall duration of 295.916 s.

The frozen v4 episode predates the new supported-placement-entry gate. Re-evaluating its unchanged traces with the **current** judge passes; this does not mean the new judge authored the original verdict. The requested evidence test from `native/nano/Cargo.toml` passed: **1 passed, 0 failed**. Verdict booleans/counts agree exactly; numeric serialization comparisons use only the specified **1×10⁻¹² m** tolerance.

Limits: one experimental static-map simulation, not benchmark or hardware certification. Pose evidence is 5 Hz and contact pairs are 100 Hz; sub-frame cube height, motor torque readback and original GPU contact-buffer bounds are not retained. Robot self-collision is disabled. No source edits, dataset edits, broad gate reruns or publishing occurred.

Full JSON, video and Rust-test audit artifacts remain local with the retained episode.
