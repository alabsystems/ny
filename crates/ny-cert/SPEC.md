# ny-cert: the proof-carrying soundness spec (Trust = Clean fusion)

ny-cert emits exact-rational Farkas/entailment certificates that Clean's kernel
re-checks. The Rust checker carries Trust contracts whose soundness link is a
Clean theorem. A float bug in certificate production therefore surfaces as a
rejected certificate, never a trusted false `unsat`. VNN-COMP scoring does not
measure this (see "VNN-COMP positioning").

## The one soundness obligation (the fixed point)

For premises `gᵢ : S → ℚ`, non-negative multipliers `μᵢ ≥ 0`, output `out`, margin
`c`, and a validity predicate over the input box:

> if each `gᵢ` is a sound `≤ 0` relaxation on valid states **and** `Σ μᵢ·gᵢ = −out − c`,
> then `∀ s. valid s → out s ≥ −c`.

This is **`farkas_premise_combination`** in the `Crownproof.Bridge` module:
Lean-proven, sorry-free (checked by `cite_check`), and resting only on
`[propext, Classical.choice, Quot.sound]` (`proofs/lean/AXIOM_AUDIT.txt`).
ReLU / depth-k / softmax / LayerNorm / β-CROWN are instantiations.

Clean is a pinned Lake dependency, not vendored source. `proofs/lean/lakefile.toml`
uses private Clean commit `fae0c94829740c9ffe527eeeb4ea472f9c08001b` with
`subDir = "crown-proofs/lean"`; the dependency supplies `Crownproof`, including
the shared soundness-method proofs (Bridge, Deep, DeepK, Sbar, Gelu, LayerNorm,
McCormick, Rsqrt, Hull/ConvHull, Complete*, cert checkers, BaB/branch). NY-origin
modules live in the separate `NyProof` library under `proofs/lean/NyProof/`.
`proofs/lean/PROVENANCE.md` classifies every module.

## Trust-verified status of the Rust checker (the moat, realized)

Contract verification needs the external Trust toolchain. The 2026-06-27 L0
survey (L0 = panic-freedom + arithmetic safety + bounded allocation) reported:

| Function (`src/selfcheck.rs`) | Status | Clean grounding |
|---|---|---|
| `check_farkas`     | **VERIFIED** | `farkas_premise_combination` |
| `check_entailment` | **VERIFIED** | `farkas_premise_combination` |
| `check_chain`      | **VERIFIED** | `farkas_premise_combination` per step, composed by the cut rule (list form: `cert_list_sound`) |

Since the 2026-08-19 toolchain, the strict lane (`targo trust check`) refuses
the ny-cert unit before compiling it: `ny-contracts` (lib name `trust`, which
supplies `#[ensures]`/`#[trust::cite]`) is a proc-macro crate, and evidence-grade
verification enforces a no-proc-macro TCB boundary. Re-run the lane before
quoting either result as current.

## L1 functional contracts (the soundness postconditions)

The checker carries its L1 soundness contracts as `#[ensures]` postconditions:

- `check_farkas`: `Ok(c) ⇒ ¬c.is_positive()` (an accepted cert is a Farkas
  contradiction).
- `check_entailment`, `check_chain`: `Ok((derived, claimed)) ⇒ derived ≤ claimed`.

Mechanism: `lib.rs` sets `#![cfg_attr(trust_verify, feature(contracts))]`.
Under `--cfg trust_verify`, `selfcheck.rs` imports `core::contracts::ensures`,
the builtin that emits a static postcondition VC. Otherwise it imports the
no-op `ny-contracts` macro. Each checker also carries
`#[trust::cite(crownproof::farkas_premise_combination)]`.

`cite_check` resolves every cited theorem against the pinned dependency
checkout (`proofs/lean/.lake/packages/crownproof/crown-proofs/lean/Crownproof`).
It validates the lakefile requirement, the `lake-manifest.json` entry, and the
checkout's `HEAD` against `CLEAN_GIT_REV`, and fails closed when the checkout is
missing (run `lake update`). It never reads a copied local corpus.

## Remaining proof work — the cite-discharge (the literal fusion)

In the 2026-06-27 `--contracts` survey, the L1 postconditions of the three
checkers were captured as real VCs (3 assertion obligations each). The full
verifier reported them `unknown` (`unsupported MIR
FullVerification::Postcondition`): ay does not yet discharge the modular proof
over the opaque bignum `Rat` (tracking `combined.constant` through `combine()`
and the control-flow guard). This captured-but-unknown state is the premise of
`CertifiedModuloCite`.

The discharge path is the cite-discharge: `trust_verify` discharges the
postcondition modulo the cited Clean theorem, reusing the `cite_check`
resolver, and reports `CertifiedModuloCite`. This is a change to `trust_verify`
(the rustc fork) plus a toolchain rebuild. A green `--contracts` build of
ny-cert then becomes a kernel-checked theorem that accepting a certificate
implies the bound holds, on an explicit trust base (ay + the cited Clean
theorems).

`src/proof_carrying.rs` already composes the same status outside the verifier
(`compose_from_surveys`, `classify`) from the two survey JSONs and the cite
check. No code in the workspace calls it.

## Float adequacy (`R_float ⊑ R_real`): closed by construction for emitted certs

The certificate path is exact. `f32_to_rat` (`src/bin/certify_onnx.rs`, via
`Rat::from_f32_exact`) lifts every f32 weight losslessly to an `n/2^k` rational
(tested by `from_f32_exact_is_lossless_and_fails_closed` in `src/rational.rs`).
The CROWN backward pass and relaxation slopes then run in exact `Rat`
(`crown.rs`, `crown_deep::{preact_bounds_crown, crown_bound_z_exact}`). For
emitted certificates `R_float = R_real` exactly, so `farkas_premise_combination`'s
"every premise ≤ 0 on valid states" holds over ℚ unchanged.

The former f64 `preact_bounds_crown_snapped` fast path has been removed.
`preact_bounds_crown` is exact by default; its opt-in `interm_round` config
outward-rounds each layer's bounds to 53-significant-bit dyadics
(`dyadic_round_out`), which only widens them. The residual float TCB is the
non-certificate ny-propagate verifier, which uses directed outward rounding
(`next_up`/`next_down`) and emits no certificate. `NyProof/FloatAdequacy.lean`
proves the per-op directed-rounding lemmas (`interval_outward_contains`,
`affine_lower_adequate`/`affine_upper_adequate`, `box_adequate`,
`relu_chord_upper_adequate`, `float_bound_implies_real`). They de-TCB the
non-certificate paths and are not needed for emitted-certificate soundness.

**Directed rounding is necessary but not sufficient for transcendental ops.**
The `dn/up` model assumes correctly outward-rounded operations and cannot see
transcendental underflow, which is correct IEEE-754 yet can make an algorithm
under-approximate. ny-propagate's softmax IBP once shifted every `exp` by a
shared row max: `exp(score − M)` underflowed to `0.0`, a `+SOFTMAX_EPSILON`
denominator term swamped the survivors, and a reachable key's `p_hi` collapsed
to `~0`, so the certified interval excluded a reachable true softmax of `1.0`
(a false proof). The fix shifts each ratio by its own dominant term
(`exp(0) = 1`, no underflow, no epsilon); see
`crates/ny-propagate/src/layers/softmax/ibp/mod.rs`.

`NyProof/SoftmaxFloatRange.lean` makes this a theorem. It models
finite-precision `exp` as an under-estimating operator `FExp` whose structure
fields (`nonneg`, `le_exp : f x ≤ exp x`, `at_zero : f 0 = 1`) name the float
TCB as hypotheses, not axioms. `softmax_phi_upper_sound` proves the per-ratio
form a sound over-approximation (general `n`); `softmax_phi_shared_unsound`
constructively refutes the shared-shift form (flush-to-zero `FExp` + a
`>745`-gap witness → computed `0 <` true `1`). The module prints axioms for
both; the expected result is `[propext, Classical.choice, Quot.sound]`.

## TCB manifest (the named trust base — machine-tracked)

The trust base is recorded in the machine-readable manifest
**`proofs/tcb.json`**: each row's assumption, status, residual TCB, and the
discharge path that would retire it. This table mirrors it:

| Row | Status | Retired by |
|---|---|---|
| `float_adequacy` (`R_float ⊑ R_real`) | closed-by-construction for emitted certs | `FloatAdequacy.lean` (directed-rounding) + `SoftmaxFloatRange.lean` (transcendental-underflow, real-IEEE) — residual fast/non-cert paths only |
| `clean_kernel` (cited theorems valid) | kernel-checked | irreducible (the 3-axiom trust root) |
| `ay_discharge` (L1 captured) | modulo-cite | teach ay modular `Rat` reasoning → `CertifiedToAxioms` |

`tcb_check`'s tests fail when this table and `tcb.json` disagree on row ids or
statuses, or when a `.lean` file named in a row's `discharge_path` is missing
from `proofs/lean/NyProof/` or from that row's "Retired by" cell. Certificate
emission does not yet attest the manifest.

## VNN-COMP positioning

NY has not been an official VNN-COMP entrant, so it has no official score or
rank. Its local, reproducible measurements and their limits are described in
[`reports/measured/README.md`](../../reports/measured/README.md). The
proof-carrying machinery above is a separate contribution; VNN-COMP scoring
does not measure it.
