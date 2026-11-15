# `#u4` DOWNSTREAM-GUARD AUDIT — where the taint word must be consulted

> Question: **which verdict-facing consumers of the in-band sentinel must
> consult the out-of-band `u32` taint word for rung 5 to arm honestly, and
> where is that word plumbed from?**
>
> The defect: `ops/sentinel_taint_selfcheck.rs` lanes 2 and 5 launder the
> in-band `±FALLBACK_BOUND` / `1e30` magnitude sentinel by downscaling. The
> fix is the transport twins (`GEMM_F32_TAINT_SHADER`,
> `GEMM_F32_SMALL_K_TAINT_SHADER`, `CROWN_AW_ERROR_COMBINE_TAINT_SHADER`,
> `CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`, the Conv reshape/col2im twins)
> plus the C2 consumer `CROWN_EFT_MIN_COMBINE_TAINT_SHADER`.
>
> Status: U4 is armed for admitted routes. AUTO/default resident walks
> (Linear, Activation, Conv) dispatch the twins, carry words on-device, and
> supply per-row words to C1; ResNet composition ORs rows across seams; the
> admitted Linear/Activation host driver runs a per-step G13 sweep. C2 is
> selected whenever EFT min-tightening is active.
> `PRODUCTION_GUARDS_CONSULT_TAINT_WORD = true`, so absent, malformed, or
> tainted C1 words refuse. Host Conv (its fused GEMM-to-col2im interior has
> no word boundary), the device-resident segment stream, and legacy
> `NY_CONV_ERR_ROWMAX=1` typed-refuse. `NY_GPU_TAINT_WORDS=0` yields absent
> words, which C1 refuses; it is not an authority bypass.
> `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED = true`; a device still needs an
> explicit typed request and a passing live qualification
> (`ComputeDevice::new_for_proof`), after which only the CROWN backward seam
> opens. The GB10/Vulkan + DenormPreserve ladder measured 5/5.

Source symbols and quoted predicates are the anchors; line numbers are hints
and drift. `shaders*.rs` and `contract_tests.rs` live in `wgpu_device/`;
other ny-gpu files cited without a directory live in `wgpu_device/ops/`.

The propagation rule (canon; the twins implement it, this file only cites it):

```text
taint_out = OR over inputs of
            (taint_in AND (partner_value != 0 OR partner_taint != 0))
         OR (this op itself saturated/degraded)
```

Clean exact-zero partners annihilate (`R*0 == 0` for every finite real the
sentinel stands for); a tainted stored zero cannot authenticate exact
annihilation. Saturating to `±inf` instead is REFUTED — `inf*0 = NaN` degrades
every dead-ReLU row (see the `GEMM_F32_TAINT_SHADER` doc block).

---

## 1. The chain being guarded (data-flow, sound path)

```text
GEMM_F32_SHADER / _SMALL_K (nan_safe_clamp: Inf -> ±1e10, NaN preserved)
  V = A@W          S = |A|@|W| (s_prod)      P = E@|W| (prop)
        \                 \                    /
         \             CROWN_AW_ERROR_COMBINE_SHADER  (s_prod|prop >= 1e10 => e = 1e30)
          \                        |
           \       [NY_EFT_ERR: CROWN_EFT_MIN_COMBINE_SHADER
            \       may lower e only when its magnitude guards accept S and P]
             \                      |
   CROWN_ACTIVATION_RESIDENT_SHADER (A'', E'')  [+ bias/intercept folds]
                     |
   Conv: reshape twin -> scheduled GEMM twin -> col2im twin (A, S, and P words)
                     |
   host readback  -> ResidentCoeff { lower_a, upper_a, lower_err, upper_err,
                                     lower_b(+err), upper_b(+err) }
                     |
   concretize_sound_gpu_batched  == host f64 preflight + CROWN_CONCRETIZE_SOUND_SHADER
                     |
   (lower, upper) bounds -> GpuCrownResult -> ny-propagate gpu_suffix ingestion
                     |
   LinearBounds / verdict
```

The diagram shows the value path. The word path runs beside it by default
whenever the device can build the twins: it selects the applicable twins (and
C2 when EFT min-tightening is active), carries word buffers on-device, applies
bias/intercept/row folds, and ORs rows across ResNet seams. Conv's col2im twin
observes every running add, so an internal sentinel followed by cancellation
stays worded.

Every sound-path CROWN verdict bound funnels through ONE function,
`WgpuDevice::concretize_sound_gpu_batched`. Production callers:

* `concretize_resident_coeff_batched`: called directly by the ResNet path, and
  through `concretize_resident_coeff` by the resident, seeded sound, residual,
  and intermediate-sweep paths;
* `crown_backward_sound_host`, through the single-domain
  `concretize_sound_gpu` wrapper.

The certified-coefficient egress (`certified_coeffs_from_resident`, shared by
`crown_backward_sound_resident_certified_coeffs` and
`crown_backward_gpu_resnet_sound_certified_coeffs`) publishes coefficients
instead of bounds and runs the same C1 consult first.

That funnel is what makes a small consult set possible (§4).

---

## 2. Guard-by-guard table

Legend. **Channel**: what the guard inspects. **Launderable?**: can a
downscaled (laundered) sentinel pass it — `MAGNITUDE-only` = yes, that is the
measured defect; `safe` = no, with the reason. **NaN?**: does the guard catch
non-finite results (the `nan_safe_clamp` contract *preserves* NaN so
concretize can catch it — `GEMM_F32_SHADER` doc, shaders.rs:771-781 and
:804-817). **Tier**: VERDICT (feeds a bound a verdict can consume), DIAG
(diagnostics/benchmark tier, no verdict authority), or MID (mid-chain
transport detector, not itself an endpoint).

### 2a. ny-gpu — shaders

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G1 | `CROWN_AW_ERROR_COMBINE_SHADER`, shaders.rs:2306 | `if (s_prod[i] >= FALLBACK_BOUND \|\| prop[i] >= FALLBACK_BOUND) { e = 1e30; }` | err combine input | **MAGNITUDE-only.** Lane 2 measured: `s_prod = 1e-10` after one `1e-20` weight → the degrade never fires. | Yes, but only via :2299 `if (is_nonfinite(e) \|\| e < 0.0) { e = 1e30; }` — NaN in `s_prod`/`prop` propagates into `e` (computed at :2298) and is caught there; :2306 alone would MISS NaN (WGSL `NaN >= x` is false). | MID |
| G2 | `CROWN_EFT_MIN_COMBINE_SHADER`, shaders.rs:2461 / :2474 | `if (pr >= FALLBACK_BOUND) { return; }` … `if (s_prod[i] >= FALLBACK_BOUND) { return; }` | EFT tightening gate | **The base shader is magnitude-only; the worded route selects C2.** A laundered `s_prod`/`prop` could let `min(err_out, e_eft)` lower a deliberately-degraded charge, so the AUTO/default route binds `CROWN_EFT_MIN_COMBINE_TAINT_SHADER` whenever this optional Linear or Conv tightening is active. C2 refuses on either word and is device- and through-walk-tested. | Partly: the non-finite refusal (:2460) covers `v`, `value`, `r`, and `prop` and runs before the magnitude arms; `s_prod` has no non-finite test, so a NaN `s_prod` reaches the NaN-blind `>=`. In C2 a NaN `s_prod` sets `taint_s` (G7 and the col2im twin seed on NaN) and refuses. | MID (optional EFT arm; `NY_EFT_ERR=1` gated) |
| G3 | `CROWN_CONCRETIZE_SHADER`, shaders.rs:2084 / :2094 | `if (a_l != a_l \|\| abs(a_l) >= FALLBACK_BOUND) { lb_degraded = true; }` (same for `a_u`) | fast-path per-coeff | **MAGNITUDE-only** (a laundered `1e-10` coefficient passes as legitimate). | Yes: `a_l != a_l` catches NaN per coefficient; final assembly re-catches via `nan_safe_lower/upper` (:2035-2048, `is_non_finite` bit test) + inversion guard :2131-2138. Degraded threads emit ±Inf (:2109-2110), which survives the reduction. | DIAG (fast unsound tier; see §3) |
| G4 | `CROWN_CONCRETIZE_SOUND_SHADER`, shaders.rs:3139 | `if (a_l != a_l \|\| abs(a_l) >= FALLBACK_BOUND \|\| a_u != a_u \|\| abs(a_u) >= FALLBACK_BOUND) { degraded = true; }` | sound per-coeff | **MAGNITUDE-only** — the terminal in-shader guard the lanes launder past. | Yes: `x != x` per coefficient; degraded rows emit ±Inf (:3215-3217); final `if (is_non_finite(lb)) { lb = -FALLBACK_BOUND; }` (:3289-3290) + inversion repair :3291-3293. `a_err` NaN is NOT tested in-shader — the host preflight (G5) refuses it before dispatch. | VERDICT |
| G5 | `concretize_sound_gpu_batched` host preflight (crown_concretize_sound.rs) | `if lower_radius >= fallback \|\| upper_radius >= fallback { return Err(…"outward affine radius … is not enclosed by FALLBACK_BOUND"…) }` — the f64 proof `Σ_j (\|a\|+err)·xmax + \|bias\| < 1e10` per spec row | sound whole-row | **C1 is the armed out-of-band verdict guard.** Magnitudes alone cannot see laundering, so the AUTO/default resident route (Linear/Activation/Conv), ResNet composition, and admitted Linear/Activation host G13 sweep supply complete `Some(rows)`. `consult_spec_row_taint` then refuses absent, mis-sized, or nonzero words. Host Conv and other unsupported/unworded configurations refuse before this funnel. | Yes, robustly: host bit tests refuse non-finite `a`/bias/input boxes and non-finite or negative `a_err`; non-finite radius arithmetic also refuses. C1 independently refuses malformed or tainted row words. | VERDICT — **the funnel** |
| G6 | `GEMM_F32_SHADER` `nan_safe_clamp`, shaders.rs:804-817 (tiled twin :944-949, small-K :1060-1066) | `if (x != x) { return x; } return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);` — applied to the FINAL value only (:868-870; the running sum is never clamped, :854-860, so a clamped partial cannot cancel back into range) | producer | Not a guard — the sentinel PRODUCER. Listed because its "final-write-only" rule is what makes the single-GEMM channel (G10) safe and the multi-op chain unsafe. | NaN preserved BY CONTRACT for downstream catch (#2708; `test_nan_safety_functions_present_in_gemm_shader` in contract_tests.rs pins the `x != x` pattern). | — |
| G7 | scheduled GEMM taint twins (`GEMM_F32_TAINT_SHADER` shaders.rs:1013, `GEMM_F32_SMALL_K_TAINT_SHADER` shaders_taint.rs:89) | `if (guarded != guarded \|\| abs(guarded) >= FALLBACK_BOUND) { taint = 1u; }` | taint seed | This IS the fix: magnitude/NaN is tested only AT THE OP THAT SATURATES, where it is exact; later non-annihilating ops OR-carry the word. Supported AUTO/default resident walks dispatch the twin matching the base schedule. | Yes — NaN seeds the taint (`guarded != guarded`). | MID (armed worded resident route) |

### 2b. ny-gpu — host paths

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G8 | `resnet_bound_exploded` (crown_backward_sound_resident.rs) | `lo.iter().chain(hi.iter()).any(\|v\| !v.is_finite() \|\| v.abs() >= crate::FALLBACK_BOUND)` | post-concretize bounds | **MAGNITUDE-only, but verdict-safe behind C1.** The armed preflight receives the composed ResNet row words and refuses a tainted row before this tightness-recovery trigger sees bounds. | Yes (`!is_finite`). | VERDICT-adjacent (fallback trigger; see §4 note) |
| G9 | `sanitize_readback`, ops/mod.rs (called from the fast path `crown_backward_gpu_seeded_inner`, crown_backward.rs) | `if !lower[i].is_finite() \|\| !upper[i].is_finite() { lower[i] = -FALLBACK_BOUND; upper[i] = FALLBACK_BOUND; }` + inversion `WidenToFallback` | fast-path readback | **MAGNITUDE-blind entirely** — repairs non-finite only; the finite sentinel and every laundered value pass through. Defense-in-depth for shader bugs (#2785), not a sentinel guard. | Yes (NaN and Inf). | DIAG |
| G10 | `fl_value_gemm` refusal, `crates/ny-gpu/src/fl_value_gemm.rs:291-300` | `if out.iter().any(\|v\| !v.is_finite() \|\| v.abs() >= ny_core::FALLBACK_BOUND) { return Err(NyError::NumericalInstability(…)) }` | value-GEMM output | **SAFE as-is.** Single-kernel channel: `nan_safe_clamp` runs only on the final write (G6), and Inf reaches the final write intact (`±Inf+finite = ±Inf`), so the sentinel appears at EXACTLY `±1e10` or not at all — no second op can launder it before this guard. Typed refusal → the CPU tiers recompute. | Yes (`!is_finite`). | **VERDICT** when the opt-in forward-linear f32 seam is armed (§3); the consumer re-checks all-finite in ny-propagate `forward_linear/image.rs` `certified_f32_gemm_deadline_gpu` |

### 2c. ny-propagate — GPU-result ingestion

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G11 | gpu_suffix ingestion (`network/graph_alpha/{bounds,backward}/gpu_suffix.rs`) via `gpu_crown_result_is_publishable` (sound_gpu_gate.rs) | row count matches and every row has `lower.is_finite() && upper.is_finite() && lower <= upper`; otherwise `Ok(None)` → CPU fallback | ingested bounds | **Cannot defend by construction**: a laundered bound is small and plausible. Protection is upstream: armed C1 receives complete row words and its typed refusal reaches the existing CPU fallback. | Yes: NaN and Inf both refuse. | VERDICT (qualified WGPU seam) |
| G12 | `add_concrete_bounds` (`network/graph_alpha/bounds/gpu_suffix.rs`) | NaN sums repaired via `RepairStrategy::Conservative` (NaN → ±inf; "the repair must widen, never clamp") | merged bounds | Same as G11 — finite launder invisible; correctly refuses to clamp to a finite substitute. | Yes (NaN → ±inf). | VERDICT (same condition) |
| G13 | seed-side firewall: gpu_suffix ingestion plus `taint_seed_word` (crown_backward_sound_resident.rs) | Non-finite seeds skip GPU at ingestion; under the worded resident route, finite `\|a\| >= CROWN_COEFF_MAX` coefficients and composed seed-error/bias markers enter pre-tainted. | seeds INTO the GPU | **Complete for admitted routes.** AUTO/default resident seeds coefficient/error/bias channels, Conv twins transport exact-op births, ResNet seam OR preserves rows, and the admitted Linear/Activation host driver sweeps every host-visible step boundary. Host Conv and routes without complete word transport typed-refuse. Out of taint scope, coefficient-error planes (`#gpu-suffix-coeff-err`): `bounds/gpu_suffix.rs` charges them into the seed bias over the node-output box and refuses when that charge is not finite or has no box; the `backward/` seam refuses via `gpu_suffix_seed_is_err_free`. | Yes (`!is_finite` skips; the word seeder also marks NaN/Inf). | VERDICT input |
| G14 | CPU reference stickiness, `bounds/batched/compose.rs:356-366` | `if [a2_l,a2_u,a1_l,a1_u].into_iter().all(is_crown_coeff_safe) { interval_mul… } else { (f32::NEG_INFINITY, f32::INFINITY) }` — comment: "Its taint must survive even multiplication by an exact zero." | CPU compose | **SAFE (strictly sticky)** — the CPU promotes the finite sentinel to ±inf at composition, refusing even exact-zero annihilation (MORE conservative than the taint rule; the GPU twins deliberately keep annihilation exact — divergence noted under `Expect::AnnihilateExactly` in sentinel_taint_selfcheck.rs). The reference the GPU must not be weaker than. | Yes (`is_crown_coeff_safe` = finite ∧ `< 1e10`, ny-core gemm.rs). | CPU reference |

### 2d. Out-of-scope-but-graded: the interval (IBP) channel

The other `FALLBACK_BOUND` uses in shaders.rs are non-finite → ±FALLBACK_BOUND
repairs and final-value clamps, not magnitude *tests*. They are the fast-path
CROWN backward clamp helpers (`clamp_coeff`/`clamp_bias_*` in
`CROWN_ACTIVATION_BACKWARD_SHADER`,
`CROWN_ACTIVATION_RELU_DUAL_ALPHA_BACKWARD_SHADER`,
`CROWN_BIAS_ACCUMULATE_SHADER`, `CROWN_MAXPOOL2D_BACKWARD_SHADER`), which
produce the sentinel on the fast path (DIAG tier, like G3), and the interval
(IBP) channel. The fast-path IBP shaders also clamp finite endpoints (e.g.
`ADD_IBP_SHADER`, documented as an unsound tightening and not verdict-legal);
the sound bodies (`IBP_SOUND_PRELUDE` + `*_IBP_SOUND_BODY`) repair non-finite
endpoints only. They would inherit the u4 obligation if a fused IBP chain ever
multiplied a repaired endpoint onward, but they are not on the CROWN verdict
chain and are not in the minimal set below.

---

## 3. Ordering: which guards gate VERDICTS vs diagnostics

1. **G10** (`fl_value_gemm`) is independent of WGPU CROWN qualification. It is
   live whenever the opt-in forward-linear f32 value seam is armed
   (`model.forward_linear_f32_seam` or `NY_FORWARD_LINEAR_F32=1`). Its gamma
   certificate covers rounding but NOT clamped overflow; the refusal is what
   keeps that true.
2. **Verdict guards on the qualified WGPU CROWN seam**: **C2/G2** protects the
   optional error-lowering step and **C1/G5 → G4** protects the terminal host
   preflight and sound concretize. Results are ingested through **G11/G12**,
   whose CPU fallback on a typed `Err` or `Ok(None)` is the fail-closed edge.
   The routing seam is `gpu_crown_backward_route_with_deadline` (ny-propagate
   sound_gpu_gate.rs), filtering on `provides_sound_gpu_crown()`
   (crown_backward.rs: `sound_gpu_authority_cached() ||
   charged_flush_authority_cached().is_some()`).
3. **Diagnostics-tier only: G3, G9** (the fast unsound path, reachable only by
   opting OUT of the sound gate) and **G8** (tightness-recovery trigger). They
   carry no authority and must never GAIN it without the same consult (rung 5
   enforces this).

---

## 4. Minimal consult set and implementation status

Three choke points form the minimal consult design; status is in the header.
Low-level raw operations carry no proof authority.

### C1 — the host preflight of `concretize_sound_gpu_batched` (extends G5)

*Where*: `consult_spec_row_taint` (crown_concretize_sound.rs), called right
after the per-row affine-radius proof in `concretize_sound_gpu_batched`, and
first in `certified_coeffs_from_resident`.
*Built and armed*: one optional pre-OR'd `u32` word per spec row. With
`PRODUCTION_GUARDS_CONSULT_TAINT_WORD = true`, absent words, a wrong row count,
or any nonzero word return the same `NyError::InvalidSpec` shape as the radius
proof. The AUTO/default resident route, ResNet composition
(`resnet_seeded_compose_coeff`), and the admitted Linear/Activation host G13
sweep supply `Some(rows)`; host Conv, explicit opt-out, and other unsupported
paths cannot pass C1.
*Plumbed from* (resident route): the `taint_rows` of the final
`ResidentCoeff`, read back beside `lower_a`/`upper_a`. The preflight takes host
slices, so no shader binding is added. The rows OR together:

* value taint from the last taint-twin kernel: `taint_out` of
  `GEMM_F32_TAINT_SHADER` / `GEMM_F32_SMALL_K_TAINT_SHADER`, the Conv
  reshape/col2im twins, or `taint_a_out` of
  `CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`, after any fold/reshape/merge
  transport;
* error taint from `CROWN_AW_ERROR_COMBINE_TAINT_SHADER` (ORs the words of the
  `|A|@|W|` and `E@|W|` GEMMs; its G1 magnitude arms seed taint) or
  `taint_e_out` of the activation twin.

Transport semantics:

* `TAINT_ROW_OR_SHADER` runs the walk's row-OR, bias-fold, and intercept-fold
  transports on-device. Its partner modes mirror the CPU references
  `bias_fold_taint` (keep a word only where `bias[k] != 0.0`) and
  `intercept_fold_taint` (keep it unless BOTH sign-routed intercepts are
  exactly zero; a tainted coefficient's sign is untrusted). Those host
  functions and `concretize_error_taint_into_bias` are `#[allow(dead_code)]`
  test references, not production calls.
* ResNet composition carries a conservative per-row OR, so moving error mass
  into bias within a row cannot clear its word.
* `CONV_RESHAPE_TAINT_SHADER` permutes values and words identically; the GEMM
  twins transport coefficient, S, propagated-error, and row-L1 words;
  `CONV_COL2IM_TAINT_SHADER` OR-carries gathered words and seeds on every
  sentinel/non-finite running sum, including a later-cancelled partial.
* The legacy row-max Conv diagnostic (`NY_CONV_ERR_ROWMAX=1`) refuses under
  the word gate: it has no elementwise word output.

*Why this is the anchor*: every sound-path CROWN verdict bound funnels through
this function (§1). Complete row words plus this consult close lanes 2 and 5
end to end without new concretize-shader bindings. The preflight refuses before
dispatch, and ny-propagate's existing `Err` handling falls back to the CPU
sound backward, so **no ny-propagate edit is required**.

### C2 — the EFT min-combine tightening gate (guards magnitude-only G2)

*Where*: the two magnitude refusal arms in `CROWN_EFT_MIN_COMBINE_SHADER`.
*Built and selected*: `CROWN_EFT_MIN_COMBINE_TAINT_SHADER` (shaders_taint.rs)
adds read-only `taint_s` and `taint_p` bindings and refuses the tightening when
either word is set. The resident walk dispatches it on both the Linear and Conv
EFT arms whenever that tightening runs under the word gate. The device probe
(`eft_min_combine_taint_probe.rs`) pins clean-word identity and both
laundering refusals. An unworded route cannot pass armed C1. The C1 constant
does not select C2; the resident word gate does.
*Plumbed from*: `taint_out` of the scheduled tiled/small-K GEMM twins
(Linear) or the post-col2im `ws`/`wprop` streams (Conv), beside
`s_prod = fl(|A|@|W|)` and `prop = fl(E@|W|)`.
*Why it cannot be deferred to C1*: `min(err_out, e_eft)` is the ONLY operation
in the chain that can LOWER an error charge, and it happens mid-chain, per
element, never read back — a laundered `s_prod` here erases a degrade that C1
would otherwise have seen. C1 catches what survives; C2 prevents the one op
that can un-survive it.

### C3 — `fl_value_gemm`: NO change required (G10 is already safe)

The single-kernel structure (final-write-only clamp, G6) makes the magnitude
test exact: no op sits between the saturation and the test, so the sentinel
cannot be laundered. Keep the guard byte-identical. (Optional hardening if the
taint twin becomes this tier's GEMM: read the word instead of the magnitude,
which also covers a future kernel that clamps earlier. This is uniformity, not
a hole.)

### Explicitly NOT in the minimal set, and why

* **G1 (Higham combine)** — a TRANSPORT node: its taint twin ORs input words
  and keeps the magnitude arms as its saturation seed. The AUTO/default
  resident route dispatches it; ResNet seams OR the resulting per-row state.
  The terminal consumer is C1.
* **G3/G9 (fast path)** — diagnostics tier with no authority; leaving their
  magnitude/NaN tests unchanged is correct. Rung 5 and
  `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED` keep this tier from ever
  carrying a verdict without the consult.
* **G8 (`resnet_bound_exploded`)** — armed C1 refuses a tainted row with a
  typed `Err` before G8 sees bounds. Optionally OR a per-row taint into the
  trigger for tightness (forcing the concretized-merge re-run instead of a
  refusal) — a quality improvement, not a soundness requirement.
* **G11/G12/G13 (ny-propagate ingestion)** — cannot see laundered values by
  construction (§2c). Their protection is armed C1 supplied with complete row
  words. G13 pre-tainting covers admitted resident, ResNet-composed, and
  Linear/Activation host-swept routes; host Conv and other configurations
  without a complete channel refuse.
* **G14 (CPU compose)** — already strictly sticky; it is the floor the GPU
  rule must meet (and, on exact-zero annihilation, deliberately exceeds in
  tightness with the `!= 0` conjuncts).

### Rung-5 arming note

`verify_sentinel_taint_sticky` (rung 5) probes the shipped value sources for
the modeled GEMM → AW-combine → activation segment plus a parallel diagnostic
word path. Its magnitude-only predicate is a lane proxy, not an execution of
C1 or C2. `taint_chain.rs` runs the three modeled transport twins end to end
and `eft_min_combine_taint_probe.rs` covers C2; neither grants authority. The
resident and ResNet integrations have their own through-walk tests and C1/C2
have focused refusal tests; the selfcheck does not substitute for them. With
the consult armed, the rung reads the with-word verdict (measured PASS on
GB10/Vulkan + DenormPreserve), while its magnitude-only fields still expose
lanes 2 and 5 as laundering controls. A stuck word on an exact-zero
annihilation lane, missing C1 input, or any transport regression refuses.

---

## 5. NaN cross-check summary (the `nan_safe_clamp` contract)

The contract (`GEMM_F32_SHADER`, shaders.rs:804-817): NaN is PRESERVED
through the value kernels — "NaN propagates through subsequent backward steps
and is caught at concretize time". Summary of the table above:

* Caught in-shader via `x != x` / `is_non_finite` bit tests: G1 (:2299 — the
  magnitude arm alone is NaN-blind), G2 (:2460 for `v`/`value`/`r`/`prop`
  only; `s_prod` is covered in C2 by its word), G3 (:2084 + :2035-2048), G4
  (:3139 + :3289-3290), G7 (shaders.rs:1013, shaders_taint.rs:89).
* Caught host-side with compiler-immune tests (the robust layer — WGSL NaN
  comparison semantics are implementation-defined, which is why GEMM's
  in-shader `x != x` is also text-pinned in contract_tests.rs): G5
  (crown_concretize_sound.rs:423-424, :441, :452 —
  `bits & 0x7f80_0000 == 0x7f80_0000` catches NaN AND Inf), G8, G9, G10, G11,
  and G13 (`!is_finite`).
* Caught with widen-only repairs, by design: G12 (NaN → ±inf, never a finite
  clamp).
* Repaired to the finite ±FALLBACK_BOUND, like G9: the CPU-parallel
  sanitization in `accelerated/crown_parallel.rs`
  (`crown_per_position_{sequential,parallel}_with_engine`, NaN/Inf →
  ±FALLBACK_BOUND).

No guard in the table relies on a NaN surviving a WGSL `max`/`min`: the #2708
fix moved all such detection BEFORE the pos/neg splits (G3's comment,
shaders.rs:2061-2065).
