# macOS confinement capability audit

This directory contains a **synthetic, unscored capability audit** for SPEC 420
TRT-3/TRT-5/TRT-6. It tests whether a controller-owned macOS `sandbox-exec`
profile can preserve declared visible/output operations while denying synthetic
hidden absolute, parent-traversal, symlink, subprocess, and write paths.

It is not a confirmatory agent session and does not establish Linux support,
vendor-network or credential compatibility, Claude/Codex parity, git-object or
alternate-worktree denial, or complete treatment integrity. The report keeps
`confirmatory_ready` false and preserves these limitations.

Regenerate a fresh timestamped audit:

```bash
uv run python -m aisle.harness.treatment_confinement audit-macos \
  --output analysis/treatment-integrity/macos-confinement/capability-audit-new.json
```

The command refuses to overwrite an existing record. Compare the new report's
case outcomes, denial-detection rate, false-alarm rate, platform identity,
adapter hash, imported `system.sb` hash, compiled-profile hash, and policy id.
Temporary synthetic hidden bytes and the compiled profile are removed after the
audit; only their hashes, exposure booleans, and role-scoped policy metadata are
retained.

Since the session-bound attestation (2026-09-14), `audit-macos --policy
<policy.json>` runs the same matrix under a session's exact policy
(`MacOSPolicy.canonical_dict` JSON) so the retained record's profile hash and
policy id are the ones `wrap_verified_command` demands; `policy_source` in the
report says whether the synthetic or a session policy was audited, and the
report's hidden roots are digests, not paths. `deny-external` allows no
network. `loopback` ([ADR-69](../../../docs/decisions/ADR-69.md)) allows TCP to
one pinned relay port; it is exclusive only while the controller holds that
port on every host address, so the audit holds it the same way and adds the
relay-reach, foreign-port, external, UDP and intruder-bind cases. Loopback
attestations from before ADR-69 lack the UDP and intruder cases, so the launch
wrapper's exact case-set check refuses them. `--widen-probes` attests an
interpreter-only policy under itself plus the audit's own probes; the record
then carries `audit_profile_sha256`/`audit_policy_id` beside the session's
`compiled_profile_sha256`/`policy_id` and a `probe_widening` declaration.
`attest-many` audits a directory of policies concurrently. See
`docs/monolithic/matched-session.md`.
