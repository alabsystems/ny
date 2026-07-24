# ny-cert: the proof-carrying soundness spec (Trust = Clean fusion)

This crate is NY's **proof-carrying** core: it emits exact-rational Farkas/entailment
certificates that an independent, tiny-TCB Clean kernel re-checks, and whose Rust
*checker* is itself **machine-proven by the Trust verifying compiler**. This is the
one axis on which NY is first-of-kind versus the entire VNN-COMP field (α,β-CROWN
included): a verdict is a *kernel-checkable proof*, not a float claim — even a future
float bug surfaces as a **rejected certificate**, never a trusted false `unsat`.

> Honest scope: VNN-COMP scores coverage + speed only and gives zero credit for this.
> On raw coverage NY does **not** out-cover the SOTA champion α,β-CROWN (official
> VNN-COMP 2025: α,β-CROWN 1st of 8). This document records the *soundness moat*,
> which is real, novel, and orthogonal to the competition's scoring. See "VNN-COMP
> positioning" below.

## The one soundness obligation (the fixed point)

For premises `gᵢ : S → ℚ`, non-negative multipliers `μᵢ ≥ 0`, output `out`, margin
`c`, and a validity predicate over the input box:

> if each `gᵢ` is a sound `≤ 0` relaxation on valid states **and** `Σ μᵢ·gᵢ = −out − c`,
> then `∀ s. valid s → out s ≥ −c`.

This is **`farkas_premise_combination`** in the `Crownproof.Bridge` module —
sorry-free, Lean-proven, and independently re-typechecked by Clean's 3-axiom
kernel (`KERNEL_IMPORT.md`). ReLU / depth-k / softmax / LayerNorm / β-CROWN are
instantiations.

Clean is a pinned Lake dependency, not vendored source. `proofs/lean/lakefile.toml`
uses private Clean commit `a119ed0cfdafcb3eca4904253fdc51283e2ff0f8` with
`subDir = "crown-proofs/lean"`; the dependency supplies `Crownproof`. NY's 67
NY-origin modules live in the separate `NyProof` library under
`proofs/lean/NyProof/`. The dependency contains the shared soundness-method
proofs (Bridge, Deep, DeepK, Sbar, Gelu, LayerNorm, McCormick, Rsqrt, the
Hull/ConvHull proofs, the Complete* proofs, cert checkers, and BaB/branch
proofs), while NY retains only its overlays, generated instances, and audit.
See `proofs/lean/PROVENANCE.md` for the exact ownership classification.

## Trust-verified status of the Rust checker (the moat, realized)

`targo trust check -p ny-cert` (contract verification uses the external Trust
toolchain — optional, not included in this repository), verified end-to-end at L0 (panic-freedom + arithmetic-safety + bounded-allocation):

| Function (`src/selfcheck.rs`) | Status | Clean grounding |
|---|---|---|
| `check_farkas`     | **VERIFIED** | `farkas_premise_combination` |
| `check_entailment` | **VERIFIED** | `farkas_premise_combination` |
| `check_chain`      | **VERIFIED** | `cert_list_sound` / `farkas_combine_list` |

`check_farkas`/`check_entailment` required a *sound Trust-toolchain improvement* (not a
spec weakening): the `UnboundedAllocation` gate now recognizes that collecting from an
already-materialized std collection (`coeffs.keys().cloned().collect()`,
`a.keys().chain(b.keys())…`) is bounded by the source's own already-gated allocation
(trust commit `cab286f009`).

Overall `ny-cert` at L0 (live trustc survey, 2026-06-27): **163 / 238 functions
verified**, 171 / 501 obligations proved; the three checker entry points above each
verify clean (1 obligation, proved, verdict `Verified`).

## L1 functional contracts (the soundness postconditions)

The checker carries its **L1 soundness contracts** as *captured* obligations:
`#[ensures]` on `check_farkas`/`check_entailment`/`check_chain` (e.g.
`Ok(c) ⇒ ¬c.is_positive()` — accepting a cert yields a Farkas contradiction). Mechanism
(verified working via `targo trust survey --contracts`): bare `#[ensures]` +
`#![cfg_attr(trust_verify, feature(contracts))]` + `core::contracts::ensures` under
tRustc / NY's `ny-contracts` compatibility macros under stable → a **static
postcondition VC**.

The Clean groundings are kernel-checked through the pinned `Crownproof`
dependency and NY's `NyProof.AxiomAudit` module. The Rust `cite_check` resolver
must consume that dependency boundary rather than a copied local corpus; until
that resolver migration is complete, it must fail closed instead of treating a
missing in-tree mirror as successful citation evidence.

## Remaining proof work — the cite-discharge (the literal fusion)

**Empirical status (live `--contracts` survey, 2026-06-27, trustc `df0b82fe5`)**: the
P1.2-A/B front-end lowering (the `matches!`/`is_positive` contract lowering in
`trust_contract_query.rs` + the `{base}_sign` LIA theory in `spec_parse.rs`) is landed,
and **the L1 postconditions are now CAPTURED as real VCs**: the survey of
`check_farkas` / `check_entailment` / `check_chain` shows **3 assertion obligations each**
(previously the `matches!(.., is_positive())` predicate was *rejected* by the contract
front-end — not a VC at all). The full verifier reports them `unknown`
(`unsupported MIR FullVerification::Postcondition`): ay does **not** discharge the modular
functional proof over the opaque bignum `Rat` type (tracking `combined.constant` through
`combine()` + the control-flow guard) — the deeper item #2, still open (needs
full-verifier postcondition-MIR support **and** ay's `Rat` modular reasoning). **This
captured-but-unknown state is exactly the cite-discharge premise → `CertifiedModuloCite`.**
(Datatype reasoning is *not* the blocker — ay already has `Sort::Datatype`; the earlier
"M-SORT cliff" framing is stale.)

The discharge path is the **cite-discharge** — exactly the Trust = Clean fusion: since
ay alone cannot prove the soundness link, ground the postcondition in
`farkas_premise_combination` (which *does* prove it in Clean, kernel-checked,
`cite_check`-verified sorry-free). Mechanism: a cite-map → `trust_verify` discharges the
postcondition *modulo* the cited theorem → honest `Certified-modulo-cited-theorem` tag
(reusing the `cite_check` resolver). This is a `trust_verify` (rustc-fork) change +
toolchain rebuild — a focused, soundness-critical effort. A green `--contracts` build of
`ny-cert` then becomes a kernel-checked theorem that *accepting a certificate implies the
bound holds*, on an explicit, minimal trust base (ay + the cited Clean theorems).

## Float adequacy (`R_float ⊑ R_real`): closed by construction for emitted certs

The float-soundness TCB once flagged as the program's open go/no-go is **not in the
proof-carrying path**. The authoritative ny-cert certificate path is **exact**:
`f32_to_rat` (`certify_onnx.rs:225`) lifts every f32 weight *losslessly* to an `n/2^k`
bignum rational (verified: 0 round-trip failures in 100k random f32s), and the entire
CROWN backward pass / relaxation-slope math then runs in exact `Rat` (`crown.rs`,
`crown_deep::{preact_bounds_crown, crown_bound_z_exact}`). So **for emitted certificates
`R_float = R_real` exactly** — there is no float in the cert math, and
`farkas_premise_combination`'s "every premise ≤ 0 on valid states" holds over ℚ
unchanged. `R_float ⊑ R_real` is therefore *closed by construction* on this path.

The residual float TCB is narrow and already-contained: (a) the fast
`preact_bounds_crown_snapped` f64 path (`crown_deep.rs:551`) is, by its own code
(`:790-799`), **cut-DISCOVERY-only** and falls back to the exact-rational path for the
verdict; (b) the broader ny-propagate f32 verifier does *directed outward
rounding* (`round_to_precision_outward`/`next_up`/`next_down`) but is not
certificate-emitting. The per-op adequacy lemma set (affine / box / ReLU-chord
`R_float ⊑ R_real`) is tractable as Clean lemmas composed through the Farkas core
(`affine_premise_adequacy` prototyped); it would de-TCB the fast/non-cert paths but is
**not needed for the soundness of emitted certificates**.

**Correction (2026-06-29) — directed rounding is necessary but NOT sufficient for
transcendental ops.** The `dn/up` model (`FloatAdequacy.lean`) assumes correctly
outward-rounded operations; it is structurally blind to transcendental *underflow*,
which is correct IEEE-754 yet can make an algorithm under-approximate. A reachable
**false proof** was found and fixed in ny-propagate's softmax IBP: a shared row-max
shift made `exp(score − M)` underflow to exactly `0.0` and a `+SOFTMAX_EPSILON`
denominator term swamped the survivors, collapsing a *reachable* key's `p_hi` to `~0`
so the certified interval **excluded a reachable true softmax of `1.0`**. The fix is a
per-ratio shift (the dominant term becomes `exp(0)=1`, no underflow, no epsilon).
`SoftmaxFloatRange.lean` makes this a theorem: it models finite-precision `exp` as an
**under-estimating** operator `FExp` (named float TCB: `nonneg`, `le_exp : f x ≤ exp x`,
`at_zero : f 0 = 1` — structure fields, not axioms) and proves
`softmax_phi_upper_sound` (the per-ratio form is a sound over-approximation, general `n`)
while `softmax_phi_shared_unsound` **constructively refutes** the shared-shift form
(flush-to-zero `FExp` + a `>745`-gap witness → computed `0 <` true `1`). `lake build`
clean; `#print axioms` on both `= [propext, Classical.choice, Quot.sound]`. This is the
first real-IEEE (vs abstract-rounding) adequacy lemma; it de-TCBs the transcendental
residual the `dn/up` lemmas could not reach.

## TCB manifest (the named trust base — machine-tracked)

The adequacy binding and the rest of the trust base are integrated as a structured,
machine-readable manifest — **`proofs/tcb.json`** — not prose alone: each emitted
certificate's named assumptions, their status, the residual TCB they imply, and the
discharge path that would retire each. Rows:

| Row | Status | Retired by |
|---|---|---|
| `float_adequacy` (`R_float ⊑ R_real`) | closed-by-construction for emitted certs | `FloatAdequacy.lean` (directed-rounding) + `SoftmaxFloatRange.lean` (transcendental-underflow, real-IEEE) — residual fast/non-cert paths only |
| `clean_kernel` (cited theorems valid) | kernel-checked | irreducible (the 3-axiom trust root) |
| `ay_discharge` (L1 captured) | modulo-cite | teach ay modular `Rat` reasoning → `CertifiedToAxioms` |

Next (build-gated): wire cert emission to attest the manifest **by default**
(batteries-included — on, not opt-in) and add a `tcb_check` drift-guard mirroring
`cite_check`.

## VNN-COMP positioning (honest, measured)

Official VNN-COMP 2025 (the most recent edition): 8 teams; ranking
**α,β-CROWN (1st) · NeuralSAT · PyRAT · CORA · NNV · nnenum · SobolBox**. α,β-CROWN ran
all benchmarks and scores ~100 on most; scoring = Σ per-benchmark-normalized (top=100;
+10 correct / −150 wrong / 0 timeout).

NY's measured coverage is strong on many benchmarks (collins-rul 100%, dist-shift 100%,
safenlp ~92%, acasxu ~89%) but has genuine algorithmic gaps where the SOTA scores ~100
(nn4sys 41/194; lsnc_relu / cora / cifar100 / tinyimagenet timeout). **NY would place
mid-field, not 1st** — it does not retroactively win on coverage. Closing that gap is a
multi-year coverage-research program (graph+input-split α-CROWN convergence, the
deep-resnet keystone BaB). The **proof-carrying moat above is NY's genuine win** on an
axis the competition does not measure.
