# Pinned JIT compatibility dependencies

The Apache 2.0 terms are included in `LICENSE-APACHE`. The Zenoh license
and upstream notices are in `ZENOH-LICENSE` and `ZENOH-NOTICE.md`.

`static_init/` is the crates.io `static_init` 1.0.4 source (upstream
`https://gitlab.com/okannen/static_init`, declared license MIT OR Apache-2.0).
All distributed upstream source, tests and manifest metadata are retained.
Cargo's local cache marker and VCS cache metadata are omitted.

The only source changes are `build.rs` declaring `cfg(relax)` and
`src/exit_sequentializer.rs` registering the original global exit cleanup
through `atexit` when running under Relax. Native builds keep the original
destructor attribute. Registration happens in a constructor before `main`,
so `atexit` callbacks registered by `main` still run before static cleanup.
Lazy resources are still registered in the original
synchronized list, and cleanup executes that same list in the same order.
No Zenoh cleanup routine is removed or stubbed. Explicit user destructor
attributes and thread-local destructor features are outside this patch.

The exit fixture verifies normal return, explicit exit, failure exit, exact
drop count/order, concurrent first use, statics initialized during destruction,
and absence of drops for untouched lazy statics.

Upstream commit: `a15b9fc6eeed143f9bf91cb26a8c7bd5f6f82914`.
Published crate SHA-256:
`8bae1df58c5fea7502e8e352ec26b5579f6178e1fdb311e088580c980dee25ed`.

## PSM stack probing

`psm/` contains upstream 0.1.32 (MIT OR Apache-2.0), commit
`cdf19cf36cd0343a97c174ee6f7f8241f88b35f0`, crate SHA-256
`4dcd034599e63b970727f70d79e02d62390a4a84f7c6b827c27c46d5ac3fa622`.
Only `build.rs` and `src/lib.rs` differ. On Relax/aarch64, `stack_pointer`
uses `mov <register>, sp`, the same instruction as upstream's assembly helper.
This works in Relax's procedural-macro compiler before the normal native
dependency bundle is loaded. Stack probing stays enabled; other builds retain
the upstream external helper. No process-wide library preload is needed.

## Zenoh routing coercion

`zenoh/` contains upstream 1.9.0 (EPL-2.0 OR Apache-2.0), commit
`81c6c933b6e41d72a05f04c4442ef57717ddc72b`, crate SHA-256
`85e22d7002ac149ef17fe400bb40a267ebbba40a83413bab03da7762256fa94e`.
`build.rs`, `src/lib.rs` and `src/net/routing/dispatcher/resource.rs` differ. Relax
does not perform the `&Child` to `&Arc<Resource>` dereference coercion inside
a tuple assignment. Its branch names that same Arc field explicitly and then
assigns the suffix. Native builds retain the original tuple assignment.
Discovery, routing, transport and resource ownership are unchanged.
The JIT-only version label uses the vendored upstream commit plus `aisle-jit`.
The upstream `git_version!` macro embeds the enclosing checkout's mutable
Git index; lazy compilation could otherwise fail when a developer stages
files during a session. Native version-label generation is unchanged.

`zenoh-protocol/` is the same upstream 1.9.0 commit and license, crate SHA-256
`eeab45020bbecc077f14f06ee8f5aee65ce760af72481e663cac58b5dbfa66dd`.
Its manifest declares `cfg(relax)`; `src/core/wire_expr.rs` uses
`Cow::to_mut().push_str()` instead of the missing `Cow<str>::AddAssign<&str>`
under Relax. Empty suffixes retain their original ownership state. The wire
expression tests cover borrowed/owned strings, empty strings, Unicode,
scope and mapping. Native builds retain the upstream operator.

## Matching Relax std snapshot

`relax-std.patch` is the already merged Makepad
[`as_mut_ptr_range` fix](https://github.com/makepad/makepad/commit/3ec45e06b698d4c1eaaf034b7fb4a7129c4967e8).
Zenoh's allocator needs this missing slice method. The launcher copies the
installed runner's matching std into a content-keyed private cache, applies
this patch only if the method is missing, and sets `RELAX_STD` for its own
process. It never edits the installed runner/std or changes another session.
The upstream regression is retained as
`../jit-fixtures/src/slice_as_mut_ptr_range.rs`.

These compatibility changes are validated on macOS Apple Silicon with the
runner pinned in `../../jit-runtime.json`; they do not establish support for
every platform or arbitrary destructor/linker sections. Exit cleanup covers
normal return, explicit exit and the tested panic path. OS `_exit`, abort and
SIGKILL do not run Rust/C exit cleanup.
