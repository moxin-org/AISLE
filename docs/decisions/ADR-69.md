# ADR-69 — An exclusive relay transport: reserve the relay port on every host address

Status: PROPOSED — owner review required (confinement class, TRT-5). Revisits
the `loopback` refusal recorded in #589. Answers the TRT gate blocker named
in the `cse-causal-study-pilot-v13` registration and on #347.

## Context

A confined Codex session must reach exactly one network endpoint: the
campaign-owned provider relay. Every other destination is a stop condition
("relay bypass", "unattested environment"). Two facts block that today.

1. **Codex cannot use a Unix socket for its provider.** Checked against the
   source at `rust-v0.153.4` (the pinned agent) and `rust-v0.167.0` (latest):
   the provider client in `codex-rs/http-client` connects over TCP only, direct
   or through a TCP proxy. The 0.167 named-pipe proxy route is
   `#[cfg(windows)]` only. Upgrading Codex does not change this.
2. **The only TCP grant macOS SBPL offers is not exclusive.** #589 rejected
   `(allow network-outbound (remote ip "localhost:P"))` after reproducing an
   IPv6 listener on the same port answering while the relay held IPv4.

A probe on 2026-11-09 (macOS 27.2, synthetic listeners and canaries, no repo
code) measured the rule precisely:

| Destination on port P | Sandbox verdict |
|---|---|
| `127.0.0.1`, `::1`, `fe80::1%lo0`, `::ffff:127.0.0.1` | admitted |
| the host's LAN address (en0) | **admitted** |
| `127.0.0.2` (not assigned to the host) | denied (EPERM) |
| `127.0.0.1` on any other port | denied (EPERM) |

`localhost` in SBPL means *every address assigned to this host*, so the defect
is wider than #589 recorded. The probe then tried three reservations against
unsandboxed intruders binding P under all four `SO_REUSEADDR`/`SO_REUSEPORT`
combinations:

| Controller holds | Intruders that bound | Who answered the sandbox |
|---|---|---|
| dual-stack wildcard `::` only | 4 (specific addresses, `SO_REUSEADDR`) | intruder, on every loopback address |
| the three lo0 addresses only | 3 (wildcards and the LAN address) | intruder, on the LAN address |
| **wildcard `::` plus all 11 assigned addresses** | **0 of 52** | **controller, everywhere it connected** |

The intruders in that matrix were same-user, unprivileged processes. Review
then found the rule is wider still: `remote ip` covers UDP as well as TCP. A
follow-up probe (same host) sent a sandboxed UDP datagram to an unsandboxed
listener on port P:

| Sandbox rule | UDP to `127.0.0.1:P` | UDP to the LAN address | TCP to the relay |
|---|---|---|---|
| `(remote ip "localhost:P")` | **delivered** | **delivered** | connected |
| `(remote tcp "localhost:P")` | denied (EPERM) | denied (EPERM) | connected |

The rule #589 proposed would therefore have leaked over UDP even with a
perfect TCP reservation.

## Decision

1. **The grant is TCP only.** The profile MUST use
   `(allow network-outbound (remote tcp "localhost:P"))`, never `remote ip`,
   and the capability audit carries a UDP negative case.
2. **Port reservation.** Before a confined session starts, the controller
   binds the dual-stack wildcard `::` on an ephemeral port P, then binds P on
   every address currently assigned to the host. No reservation socket sets
   `SO_REUSEPORT`; the specific binds set `SO_REUSEADDR` only because BSD
   requires it to bind a specific address beneath a held wildcard. Any bind
   failure refuses the session; there is no partial reservation.
3. **Only the relay listens.** The relay serves on the `127.0.0.1` reservation
   socket. Every other reservation socket stays bound and never listens, so a
   connection to it is never answered. Measured on macOS 27.2, the kernel
   drops the SYN to a bound, non-listening socket instead of resetting it
   (`net.inet.tcp.blackhole` is 0), so a stray connect hangs until the
   client's timeout rather than failing fast. Listener threads that accept
   and close were rejected as extra moving parts for no confinement gain.
4. **Address drift is a stop condition.** The reservation retains the address
   set it covered and re-enumerates once after binding, refusing on any
   change. During the session the controller re-enumerates on a fixed cadence
   (set in PR B) and at teardown; any added or removed address, or a failed
   enumeration, ends the session and marks it *unattested environment* (an
   existing stop condition). Detection is after the fact: an intruder that
   binds a new address can receive traffic until the next check, so sessions
   also require a quiescent network as an attested precondition (VPN and
   container bridges down, no interface churn), and PR B evaluates watching
   `PF_ROUTE` for immediate detection.
5. **Retained evidence names no addresses.** The record carries the port, the
   counts per family and a SHA-256 over the sorted address set, not the LAN
   topology; refusal messages name an address's family and position only.
   The digest is unsalted, so records from one host are linkable to each
   other; that is accepted, since a session record already binds its host.
6. **`loopback` returns only with a reservation.** The policy compiles to the
   port-pinned TCP rule, but profile compilation, the capability audit,
   provider admission and launch verification all require a live reservation
   for the same port. The capability audit gains the probe's intruder cases as
   negative cases under the session's own profile.

## Rollout

- **PR A (this ADR):** `src/aisle/harness/relay_reservation.py` — address
  enumeration, the reservation, the drift check, and their tests on real
  sockets. New file outside every hashed source set; no registration change.
- **PR B:** wire the reservation into `treatment_confinement.py`
  (re-admit `loopback` with a reservation), `provider_relay.py` (serve on the
  reservation socket), the matched-session runner (drift checks), the audit's
  intruder and UDP cases, a per-session relay token (so other local
  processes cannot spend the provider credential through the relay), an
  explicit hand-off of only the relay socket (no reservation descriptor may
  reach the confined process), and successor registrations for the changed
  hashed sources.

## Consequences and residual risk

- **Mid-session address changes** (VPN connect, DHCP renewal, IPv6
  temporary-address rotation, a new `utun`) open a window between the address
  appearing and the next drift check. The check excludes the session, but
  traffic sent in that window has already left; the quiescent-network
  precondition is what keeps the window rare. Temporary-address rotation on
  an otherwise idle host will exclude sessions by design.
- **Addresses not yet usable** (`tentative`, `duplicated`, `detached`) refuse
  the reservation instead of being skipped, because they may become bindable
  later; expect an occasional refusal shortly after wake or a network join.
- **Privileged processes** are out of scope: the threat is an ordinary local
  service on the same port, not root on the controller host. An operator
  account with passwordless `sudo` is outside that boundary.
- **Undocumented semantics.** What SBPL `localhost` admits is not a public
  contract; it was measured on macOS 27.2 build 26B123. The audit's intruder
  and UDP cases re-measure it on every attested host and build.
- **Enumeration source.** Addresses come from `ifconfig` text, run with a
  pinned environment and checked to include loopback. `getifaddrs(3)` would
  be a stronger source; any address it lists that `ifconfig` misses would be
  an uncovered address.

## Alternatives rejected

- **Unix-socket provider transport:** unavailable in Codex (above).
- **An in-sandbox TCP-to-Unix-socket shim:** the agent still reaches the shim
  through the same `localhost:P` grant, so nothing changes.
- **Linux network namespaces:** exclusive by construction, but the confinement,
  audit and hidden-access sensors are macOS-built; a port is a larger project.
- **Patching or injecting into Codex:** changes the agent under study.
