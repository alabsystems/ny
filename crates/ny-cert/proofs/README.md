<!-- Copyright 2026 Andrew Yates -->
<!-- Author: Andrew Yates <andrewyates.name@gmail.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Machine-checked soundness proofs for `ny-cert`

These are **real, machine-checked proofs** of the soundness lemmas that make the
CROWN certificate construction sound — not statistical tests. The Kani models
are NY-owned; the Lean project is an NY-owned overlay over a pinned Clean Lake
dependency.

The lemmas proven are exactly the per-premise validity obligations the certifier
relies on (see `../src/crown.rs` and `../SPEC.md`):

- **ReLU lower envelope** — for `α ∈ [0,1]` and all `z`: `α·z ≤ ReLU(z)`.
- **ReLU upper envelope (unstable chord)** — for `l < 0 < u`, `z ∈ [l,u]`,
  `s = u/(u−l)`: `ReLU(z) ≤ s·(z−l)`.
- **Farkas combination** — a non-negative combination of valid `≤` inequalities
  is a valid `≤` inequality (single-step and general `n`-row induction).

Clean's external verifier checks the *linear-program* half of each certificate
(that the multipliers derive the bound); these proofs discharge the *other* half
— that the premises the certifier emits are genuinely sound relaxations of ReLU.
Internal development uses the private Clean repository pin recorded in
`lean/lakefile.toml`; publication rewrites that dependency through the release
mapping.

## `kani/` — symbolic model checking (all inputs, exact)

Kani lowers each lemma to a SAT/SMT query over the **entire** bounded-integer
input lattice (rationals scaled to integers, widened so the arithmetic is exact
with no overflow) — this is exhaustive over the modeled domain, not sampling.

```sh
cd kani && cargo kani
# => Complete - 4 successfully verified harnesses, 0 failures, 4 total.
```

Verified this repo with **Kani 0.67.0 (CBMC 6.8.0 + CaDiCaL 2.0.0)**:
`VERIFICATION:- SUCCESSFUL` for the upper envelope, lower envelope, and Farkas
combination harnesses.

## `lean/` — Lean 4 + mathlib (deductive proof over ℚ)

Full deductive proofs over the rationals `ℚ` (an ordered field), no `sorry`.

```sh
cd lean
lake update
lake exe cache get
lake build NyProof
# => Build completed successfully.
# Every theorem's `#print axioms` lists ONLY [propext, Classical.choice, Quot.sound]
# and never `sorryAx` — the proofs are complete, not stubbed.
```

Verified with **Lean 4.30.0 + mathlib v4.30.0**: `relu_lower`, `relu_upper`,
`farkas_pair`, and `farkas_comb` (general `n`-row list induction) all typecheck
with the standard mathlib axiom base only.

### Clean dependency and NY overlay

Clean's `Crownproof` library supplies the shared soundness and certificate
checker modules at exact commit
`a119ed0cfdafcb3eca4904253fdc51283e2ff0f8`. NY does not copy those sources.
The 67 NY-origin modules live under `lean/NyProof/` and import the Clean base as
`Crownproof.*`. See `lean/PROVENANCE.md` for the complete retained/removed
classification and authorship record.

## What is still an axiom (honest gap)

The end-to-end statement *"Clean accepting NY's certificate implies the certified
bound is a true lower bound of the network"* is **not yet** a single
kernel-checked theorem inside Clean: Clean's own `nn_verify` library currently
takes `farkas_to_interval` as an **axiom** (`clean-kernel` builds and its 1371
`nn_verify` tests pass, but that bridge lemma has no constructive proof term).
Closing it — constructively proving `farkas_to_interval` in Clean's kernel and
making Clean's cert parser combine NY's exact-rational multipliers — is the next
milestone. The Kani and Lean proofs here establish the mathematical content of
that bridge in two independent systems.
