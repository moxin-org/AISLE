# ADR-p3-return-orientation — Point the return gripper toward the tray

Status: implementation interpretation of CAP-5, TC-1/2 and ADR-32;
no spec, frozen scene, graph, guard or scoring-rule change.

The return planner emits a standard executor grasp pose. The executor treats
the pose's local z axis as the direction from wrist to fingertips and computes
the pregrasp by moving backward along that axis. An identity quaternion points
that axis upward, placing the pregrasp below a tray object. The measured-pose
and tray-centre fallback paths now use the existing top-down orientation helper,
including its Franka hand-mount compensation, so the pregrasp is above the object.
The node passes its selected embodiment explicitly. SO-101 retains its existing
orientation path; this change and its planning evidence are scoped to Franka.
Target positions, return slots, approach distance and release-height metadata
retain their existing values.

Tests check the tool-axis convention, above-object approach and feasibility
through the actual staged IK planner. Existing assertions that pinned the
identity quaternion are replaced with these geometric properties. IK feasibility
does not establish collision-free physical execution, a successful return, task
eligibility or a history benefit. New physical qualification requires its own
declared source pin and bounded protocol; this repair does not authorize a
replacement of a closed batch or complete P3.
