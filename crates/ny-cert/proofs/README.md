<!-- Copyright 2026 Andrew Yates -->
<!-- Author: Andrew Yates <andrewyates.name@gmail.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Machine-checked soundness proofs for `ny-cert`

These are machine-checked proofs, not statistical tests, of the lemmas that make
the CROWN certificate construction sound. The Lean project is an NY-owned overlay
over a pinned Clean Lake dependency.

The lemmas are the per-premise validity obligations the certifier relies on
(see `../src/crown.rs` and `../SPEC.md`):

- **ReLU lower envelope** — for `α ∈ [0,1]` and all `z`: `α·z ≤ ReLU(z)`.
- **ReLU upper envelope (unstable chord)** — for `l < 0 < u`, `z ∈ [l,u]`,
  `s = u/(u−l)`: `ReLU(z) ≤ s·(z−l)`.
- **Farkas combination** — a non-negative combination of valid `≤` inequalities
  is a valid `≤` inequality (single-step and general `n`-row induction).

Clean's external verifier checks the linear-program half of each certificate
(that the multipliers derive the bound); these proofs discharge the other half
(that the premises the certifier emits are sound relaxations of ReLU).
Internal development uses the private Clean repository pin recorded in
`lean/lakefile.toml`; publication rewrites that dependency through the release
mapping.

The certifier's code carries these lemmas by citation.
`crown::Relu1Problem::certify` cites `crownproof::crown_bridge`
(`#[trust::cite]`, checked by `cite_check` against the pinned Clean revision).
Its proof discharges every emitted premise through `premiseFun_sound`, which
applies `relu_lower` and `relu_upper`, and combines the premises with
`farkas_premise_combination`. The α optimizer's clamp cites `relu_lower`
directly (`crates/ny-propagate/src/network/graph_alpha/propagate_dag/alpha_update.rs`).

## `lean/` — Lean 4 + mathlib (deductive proof over ℚ)

Full deductive proofs over the rationals `ℚ`, no `sorry`.

```sh
cd lean
lake update
lake exe cache get
lake build NyProof
lake build NyProof.AxiomAudit
lake env lean NyProof/AxiomAudit.lean
# => Build completed successfully.
# Every theorem's `#print axioms` lists ONLY [propext, Classical.choice, Quot.sound]
# and never `sorryAx` — the proofs are complete, not stubbed.
```

Toolchain: Lean 4.30.0 (`lean/lean-toolchain`) with Mathlib v4.30.0.
`relu_lower`, `relu_upper`, `farkas_pair`, and `farkas_comb` (general `n`-row
list induction) typecheck with the standard axiom base only.

### Clean dependency and NY overlay

Clean's `Crownproof` library supplies the shared soundness and certificate
checker modules at exact commit
`bc05e5f8b2fb4336dc20112503dfe0d90f80c69d`. NY does not copy those sources.
The 70 NY-origin modules live under `lean/NyProof/` and import the Clean base as
`Crownproof.*`. See `lean/PROVENANCE.md` for the retained/removed
classification and authorship record.

## What is still an axiom (honest gap)

The end-to-end statement *"Clean accepting NY's certificate implies the certified
bound is a true lower bound of the network"* is **not yet** a single
kernel-checked theorem inside Clean: Clean's `nn_verify` library takes
`farkas_to_interval` as an **axiom** (no constructive proof term). Closing it
means proving `farkas_to_interval` constructively in Clean's kernel and making
Clean's cert parser combine NY's exact-rational multipliers. The Lean proofs
here establish the mathematical content of that bridge.
