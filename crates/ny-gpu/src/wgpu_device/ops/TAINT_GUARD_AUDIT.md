# `#u4` DOWNSTREAM-GUARD AUDIT — where the taint word must be consulted

> Audit begun 2026-08-10 against `main` at `e7593fc1`, then updated through the
> 2026-08-11 UTC U4 arming review and the Conv transport closure. Companion to
> the transport twins
> (`GEMM_F32_TAINT_SHADER`, `CROWN_AW_ERROR_COMBINE_TAINT_SHADER`, and
> `CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`), the separate
> `CROWN_EFT_MIN_COMBINE_TAINT_SHADER`, their device probes, and the measured defect in
> `ops/sentinel_taint_selfcheck.rs` (lanes 2 and 5 launder the in-band
> `±FALLBACK_BOUND` / `1e30` magnitude sentinel via downscaling). The C1
> preflight, C2 twin, host folds, on-device row OR, and ResNet seam transport are
> built, tested, and armed. Conv reshape, both tiled and small-K GEMM schedules,
> and col2im now have exact-value twins. AUTO is the default: supported resident
> walks select the applicable transport twins (plus C2 when optional EFT min-tightening
> is active), carry words, and supply per-row words to C1. The admitted
> Linear/Activation host driver uses a per-step G13 sweep; host Conv refuses
> because its fused GEMM-to-col2im interior has no word boundary.
> `PRODUCTION_GUARDS_CONSULT_TAINT_WORD = true`; absent,
> malformed, or tainted C1 input refuses. Unsupported configurations, including
> the device-resident segment stream and legacy `NY_CONV_ERR_ROWMAX=1`,
> typed-refuse. The measured
> GB10/Vulkan + DenormPreserve ladder is 5/5. U5/U6 and B0 are now discharged
> and the raw-device CROWN source gate is open. The separate public
> `ComputeDevice`/CLI proof-integration seam remains closed.
>
> Question answered here: **every verdict-facing consumer of the in-band
> sentinel — which of them must consult the out-of-band `u32` taint word for
> rung 5 to honestly arm, and where does that word have to be plumbed from?**

Source symbols and quoted predicates are the durable anchors in this audit.
Any retained `file:line` coordinate records the original `e7593fc1` checkpoint
and may drift as nearby code changes; it is informational rather than an
authority claim about the current line number.

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

This diagram shows the value path; the admitted production-shaped word path now
runs beside it by default whenever the device can build the twins. It selects
the applicable twins and, when Linear EFT min-tightening is active, C2; carries
word buffers on-device; applies bias/intercept/row folds; and OR-composes rows
across ResNet seams. Conv's col2im twin observes every running add, so an
internal sentinel followed by cancellation remains worded. Other unsupported
shapes and the device-resident segment stream still refuse. Explicit
`NY_GPU_TAINT_WORDS=0` yields absent C1 words, which the armed preflight refuses;
it is not an authority bypass.

Every sound-path CROWN verdict bound funnels through exactly ONE function:
`WgpuDevice::concretize_sound_gpu_batched`. Production call sites,
exhaustively:

* `concretize_resident_coeff_batched` (used by the resident, resnet, and seeded
  sound paths);
* `crown_backward_sound_host`, through the single-domain
  `concretize_sound_gpu` wrapper.

That funnel is what makes a small minimal consult set possible (§4).

---

## 2. Guard-by-guard table

Legend. **Channel**: what the guard inspects. **Launderable?**: can a
downscaled (laundered) sentinel pass it — `MAGNITUDE-only` = yes, that is the
measured defect; `safe` = no, with the reason. **NaN?**: does the guard catch
non-finite results (the `nan_safe_clamp` contract *preserves* NaN specifically
so concretize can catch it — GEMM_F32_SHADER doc, shaders.rs:761-771 and
:794-806). **Tier**: VERDICT (feeds a bound a verdict can consume when GPU
authority opens; `fl_value_gemm` is live TODAY), DIAG (diagnostics/benchmark
tier, no verdict authority), or MID (mid-chain transport detector, not itself
an endpoint).

### 2a. ny-gpu — shaders

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G1 | `CROWN_AW_ERROR_COMBINE_SHADER`, shaders.rs:2293 | `if (s_prod[i] >= FALLBACK_BOUND \|\| prop[i] >= FALLBACK_BOUND) { e = 1e30; }` | err combine input | **MAGNITUDE-only.** Lane 2 measured: `s_prod = 1e-10` after one `1e-20` weight → the degrade never fires. | Yes, but only via line 2286: `if (is_nonfinite(e) \|\| e < 0.0) { e = 1e30; }` — NaN in `s_prod`/`prop` propagates into `e` and is caught there; line 2293 alone would MISS NaN (WGSL `NaN >= x` is false). Order in shader: 2285 compute, 2286 nonfinite, 2293 magnitude. | MID |
| G2 | `CROWN_EFT_MIN_COMBINE_SHADER` | `if (pr >= FALLBACK_BOUND) { return; }` … `if (s_prod[i] >= FALLBACK_BOUND) { return; }` | EFT tightening gate | **The base shader is magnitude-only; the admitted worded route selects C2.** A laundered `s_prod`/`prop` could let `min(err_out, e_eft)` lower a deliberately-degraded charge, so the AUTO/default route binds `CROWN_EFT_MIN_COMBINE_TAINT_SHADER` whenever this optional Linear or Conv tightening is active. C2 refuses on either post-transform word and is device- and through-walk-tested. | Yes: the base non-finite refusal runs BEFORE the magnitude tests, so NaN can never reach the (NaN-blind) `>=` comparisons. The C2 twin preserves that order. | MID (optional EFT arm; `NY_EFT_ERR=1` gated) |
| G3 | `CROWN_CONCRETIZE_SHADER`, shaders.rs:2072 / :2082 | `if (a_l != a_l \|\| abs(a_l) >= FALLBACK_BOUND) { lb_degraded = true; }` (same for `a_u`) | fast-path per-coeff | **MAGNITUDE-only** (a laundered `1e-10` coefficient passes as legitimate). | Yes: `a_l != a_l` catches NaN per-coefficient; final assembly re-catches via `nan_safe_lower/upper` (shaders.rs:2028-2035, `is_non_finite` bit test) + inversion guard :2120-2126. Degraded threads emit ±Inf (:2097-2098) which WGSL guarantees survives the reduction. | DIAG (fast unsound tier; see §3) |
| G4 | `CROWN_CONCRETIZE_SOUND_SHADER`, shaders.rs:3086 | `if (a_l != a_l \|\| abs(a_l) >= FALLBACK_BOUND \|\| a_u != a_u \|\| abs(a_u) >= FALLBACK_BOUND) { degraded = true; }` | sound per-coeff | **MAGNITUDE-only** — the terminal in-shader guard the lanes launder past. | Yes: `x != x` per coefficient; degraded rows emit ±Inf (:3161-3162); final `if (is_non_finite(lb)) { lb = -FALLBACK_BOUND; }` (:3234-3235) + inversion repair :3236-3238. `a_err` NaN is NOT tested in-shader — it is refused by the host preflight (G5) before dispatch. | VERDICT |
| G5 | `concretize_sound_gpu_batched` host preflight | `if lower_radius >= fallback \|\| upper_radius >= fallback { return Err(…"outward affine radius … is not enclosed by FALLBACK_BOUND"…) }` — the f64 proof `Σ_j (\|a\|+err)·xmax + \|bias\| < 1e10` per spec row | sound whole-row | **C1 is the armed out-of-band verdict guard.** Magnitudes alone cannot see laundering, so the AUTO/default resident route (Linear/Activation/Conv), ResNet composition, and admitted Linear/Activation host G13 sweep supply complete `Some(rows)`. `consult_spec_row_taint` then refuses absent, mis-sized, or nonzero words. Host Conv and other unsupported/unworded configurations refuse before this funnel. | Yes, robustly: host bit tests refuse non-finite `a`/bias/input boxes and non-finite or negative `a_err`; non-finite radius arithmetic also refuses. C1 independently refuses malformed or tainted row words. | VERDICT — **the funnel** |
| G6 | `GEMM_F32_SHADER` `nan_safe_clamp`, shaders.rs:794-806 (twin: :933-938, small-K: :1045-1053) | `if (x != x) { return x; } return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);` — applied to the FINAL value only (:844-850: partial sums are never clamped, so a clamped partial cannot cancel back into range) | producer | Not a guard — the sentinel PRODUCER. Listed because its "final-write-only" rule is the property that makes the single-GEMM channel (G10) safe and the multi-op chain unsafe. | NaN preserved BY CONTRACT for downstream catch (#2708, contract_tests.rs:570-585 pins the `x != x` pattern). | — |
| G7 | scheduled GEMM taint twins (`GEMM_F32_TAINT_SHADER`, `GEMM_F32_SMALL_K_TAINT_SHADER`) | `if (guarded != guarded \|\| abs(guarded) >= FALLBACK_BOUND) { taint = 1u; }` | taint seed | This IS the fix: magnitude/NaN is tested only AT THE OP THAT SATURATES, where it is exact; subsequent non-annihilating ops OR-carry the word. Supported AUTO/default resident walks dispatch the twin matching the base schedule. | Yes — NaN seeds the taint (`guarded != guarded`). | MID (armed worded resident route) |

### 2b. ny-gpu — host paths

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G8 | `resnet_bound_exploded` | `lo.iter().chain(hi.iter()).any(\|v\| !v.is_finite() \|\| v.abs() >= crate::FALLBACK_BOUND)` | post-concretize bounds | **MAGNITUDE-only, but verdict-safe behind C1.** The armed preflight receives the composed ResNet row words and refuses a tainted row before this tightness-recovery trigger. | Yes (`!is_finite`). | VERDICT-adjacent (fallback trigger; see §4 note) |
| G9 | `sanitize_readback`, ops/mod.rs:78-92 (called from the fast path, crown_backward.rs:2413) | `if !lower[i].is_finite() \|\| !upper[i].is_finite() { lower[i] = -FALLBACK_BOUND; upper[i] = FALLBACK_BOUND; }` + inversion `WidenToFallback` | fast-path readback | **MAGNITUDE-blind entirely** — repairs non-finite only; the finite sentinel and every laundered value pass through. Defense-in-depth for shader bugs (#2785), not a sentinel guard. | Yes (NaN and Inf). | DIAG |
| G10 | `fl_value_gemm` refusal, fl_value_gemm.rs:285-294 | `if out.iter().any(\|v\| !v.is_finite() \|\| v.abs() >= ny_core::FALLBACK_BOUND) { return Err(NyError::NumericalInstability(…)) }` | value-GEMM output | **SAFE as-is.** Single-kernel channel: `nan_safe_clamp` runs only on the final write (G6), Inf reaches the final write intact (`±Inf+finite = ±Inf`), so the sentinel appears at EXACTLY `±1e10` or not at all — there is no second op to launder it before this guard. The doc block at :274-284 states this argument. Typed refusal → the CPU tiers recompute. | Yes (`!is_finite`). | **VERDICT, live TODAY** (the FL value tier runs under the sound gate; consumer re-checks all-finite at ny-propagate image.rs:508) |

### 2c. ny-propagate — GPU-result ingestion

| # | Guard (source anchor) | Quoted predicate | Channel | Launderable? | NaN? | Tier |
|---|---|---|---|---|---|---|
| G11 | gpu_suffix ingestion | NaN results return `Ok(None)`; `BoundedTensor::new_repaired(…, RepairStrategy::Widen)` handles conservative infinities | ingested bounds | **Cannot defend by construction**: a laundered bound is small and plausible. Protection is upstream and now live in the production-shaped GPU call: armed C1 receives complete row words and typed refusal reaches the existing CPU fallback. The raw authority source gate is open but remains request- and probe-qualified. | NaN yes; Inf deliberately reaches the conservative widen repair. | VERDICT (qualified raw seam; public integration pending) |
| G12 | `add_concrete_bounds`, gpu_suffix.rs:375-409 | NaN sums repaired via `RepairStrategy::Conservative` (NaN → ±inf, "the repair must widen, never clamp" :390-398) | merged bounds | Same as G11 — finite launder invisible; correctly refuses to clamp to a finite substitute. | Yes (NaN → ±inf). | VERDICT (same condition) |
| G13 | seed-side firewall, gpu_suffix ingestion plus `taint_seed_word` | Non-finite seeds skip GPU at ingestion; under the worded resident route, finite `\|a\| >= CROWN_COEFF_MAX` coefficients and composed seed-error/bias markers enter pre-tainted. | seeds INTO the GPU | **Complete for admitted routes.** AUTO/default resident seeds coefficient/error/bias channels, Conv twins transport exact-op births, ResNet seam OR preserves rows, and the admitted Linear/Activation host driver sweeps every host-visible step boundary. Host Conv and routes without complete word transport typed-refuse. | Yes (`!is_finite` skips; the word seeder also marks NaN/Inf). | VERDICT input (armed word handling; raw source gate open) |
| G14 | CPU reference stickiness, bounds/batched/compose.rs:357-367 | `if [a2_l,a2_u,a1_l,a1_u].into_iter().all(is_crown_coeff_safe) { interval_mul… } else { (f32::NEG_INFINITY, f32::INFINITY) }` — comment: "Its taint must survive even multiplication by an exact zero." | CPU compose | **SAFE (strictly sticky)** — the CPU promotes the finite sentinel to ±inf at composition, refusing even exact-zero annihilation (MORE conservative than the taint rule; the GPU twins deliberately keep annihilation exact — divergence documented in sentinel_taint_selfcheck.rs:73-80). Listed as the reference the GPU must not be weaker than. | Yes (`is_crown_coeff_safe` = finite ∧ `< 1e10`, ny-core gemm.rs:66). | CPU reference |

### 2d. Out-of-scope-but-graded: the interval (IBP) channel

The remaining `FALLBACK_BOUND` hits in shaders.rs (IBP prelude :115-117, :139,
:192, :246-248, :390-392, :422-424, :461-465, :498, ADD_IBP :3723-3731,
AVGPOOL :3821-3829, conv-IBP :3592-3593 and :3642-3643, MATMUL_IBP
:1187-1188, softmax :1244/:1289-1302, scale :1501-1502, backward/bias/maxpool
clamp helpers :1543-1567, :1680-1699, :1803-1818, :1895-1915) are all
**non-finite → ±FALLBACK_BOUND repairs or final-value clamps on the interval
channel**, not magnitude *tests* — several explicitly refuse to clamp finite
values because that would be an unsound tightening (ADD_IBP doc :3689-3691).
Their soundness precondition ("the true magnitude stays below FALLBACK_BOUND",
:3207-3208) is proved by the corresponding host preflights, the same shape as
G5. They inherit the same u4 obligation *if* fused IBP chains ever multiply a
repaired endpoint onward, but they are not on the CROWN verdict chain this
rung guards and are not part of the minimal set below.

---

## 3. Ordering: which guards gate VERDICTS vs diagnostics

1. **Live verdict guard today (the only one): G10** (`fl_value_gemm`). The FL
   value tier feeds forward-linear image bounds in production, under the sound
   gate, with a gamma certificate that covers rounding but NOT clamped
   overflow — the refusal at fl_value_gemm.rs:285 is what keeps that true.
2. **Verdict guards on the qualified raw WGPU seam** (the B0 review opened
   `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`; an exact request and all five
   live rungs are still required):
   **C2/G2** protects the optional
   error-lowering step and **C1/G5 → G4** protects the terminal host-preflight
   and sound-concretize step. Results are ingested through **G11/G12**, whose
   fallback-on-typed-`Err` is the fail-closed edge. The
   routing seam is `gpu_crown_backward_route_with_deadline`
   (ny-propagate sound_gpu_gate.rs:573-592) filtering on
   `provides_sound_gpu_crown()` (crown_backward.rs:785-787 →
   `sound_gpu_authority_cached()`).
3. **Diagnostics-tier only: G3, G9** (the fast unsound path, reachable only by
   opting OUT of the sound gate) and **G8** (tightness-recovery trigger). They
   never carry authority; they must simply never GAIN it without the same
   consult (the ladder's rung 5 enforces exactly this).

---

## 4. Minimal consult set and implementation status

**Three choke points form the minimal consult design. U4 is armed for admitted
routes: AUTO/default resident execution transports and folds words, ResNet
composition ORs them at every seam, the admitted Linear/Activation host driver
performs per-step G13 sweeps, C2 protects optional Linear EFT min-tightening,
and C1 consumes the final row words. Host Conv and other unsupported
configurations typed-refuse. The raw authority source gate is open, and the
public `ComputeDevice`/CLI proof router exposes only CROWN after an explicit
typed five-rung qualification; low-level raw operations carry no proof
authority.**

### C1 — the host preflight of `concretize_sound_gpu_batched` (extends G5)

*Where*: `crown_concretize_sound.rs`, immediately after the per-row
affine-radius proof in `concretize_sound_gpu_batched`.
*Built and armed*: `concretize_sound_gpu_batched` accepts one optional pre-OR'd
`u32` word per spec row. Because `PRODUCTION_GUARDS_CONSULT_TAINT_WORD` is true,
absent words, a wrong row count, or any nonzero word produce the same typed
`NyError::InvalidSpec` refusal shape as the radius proof. The AUTO/default
resident route, ResNet composition, and admitted Linear/Activation host G13
sweep supply `Some(rows)`; host Conv, explicit opt-out, or another unsupported
path cannot pass absent words through C1.
*Plumbed on the resident route from*: taint buffers on the final
`ResidentCoeff`, read back beside `lower_a`/`upper_a` (the preflight takes host
slices, so no extra shader binding is needed at this point):
value-taint from the last taint-twin kernel in the chain
(`GEMM_F32_TAINT_SHADER` / `GEMM_F32_SMALL_K_TAINT_SHADER`'s `taint_out`, the
Conv reshape/col2im twins, or
`CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`'s `taint_a_out`, after any intervening
fold/reshape/merge transport); err-taint from the authored
`CROWN_AW_ERROR_COMBINE_TAINT_SHADER` (ORs the words of the `|A|@|W|` and
`E@|W|` GEMMs per the rule and keeps the existing magnitude arms at G1 as its
saturation-seeding term) or `taint_e_out` of the activation twin.
The AUTO/default supported main walk dispatches and threads these buffers. It
invokes the bias/intercept companions and performs conservative row transports.
Conv-containing resident walks use the complete internal channel. Fine-split
and ResNet seams carry their conservative per-row OR state. Relevant helper
semantics are:

* `concretize_error_taint_into_bias` pins the finer per-coefficient fold
  semantics. The current ResNet composition carries a conservative per-row OR,
  so moving error mass into bias within the same row cannot clear its word;
* `bias_fold_taint` is invoked by the main worded walk and ORs
  coefficient/error words only where `bias[k] != 0.0`;
* `intercept_fold_taint` is invoked by that walk and keeps the word unless BOTH possible sign-routed
  intercepts are exactly zero (a tainted coefficient's sign is untrusted);
* `CONV_RESHAPE_TAINT_SHADER` permutes values and words identically;
  `GEMM_F32_TAINT_SHADER` / `_SMALL_K_TAINT_SHADER` transport coefficient,
  S, propagated-error, and row-L1 words; `CONV_COL2IM_TAINT_SHADER` OR-carries
  gathered words and seeds on every sentinel/non-finite running sum, including
  a later-cancelled partial. The legacy row-max diagnostic remains refused
  under the word gate because it has no elementwise word output.

*Why this is the anchor*: §1 shows every sound-path CROWN verdict bound funnels
through this function, directly from the resident path or through the host
wrapper; resnet reaches the same resident funnel. The complete production row
words and this consult close lane 2 and lane 5 end-to-end
without adding concretize-shader bindings. The preflight refuses before
dispatch, and ny-propagate's existing `Err` handling already falls back to the
CPU sound backward — **no ny-propagate edit is required**.

### C2 — the EFT min-combine tightening gate (guards magnitude-only G2)

*Where*: the two magnitude refusal arms in `CROWN_EFT_MIN_COMBINE_SHADER`.
*Built and selected on the admitted worded EFT route*:
`CROWN_EFT_MIN_COMBINE_TAINT_SHADER` adds read-only `taint_s`
and `taint_p` bindings and refuses tightening when either word is set. The
device probe pins clean-word identity and both laundering refusals. AUTO/default
resident execution selects/binds the twin whenever this tightening is active.
An explicit unworded route cannot pass the armed C1 funnel. C1's source constant
does not select C2; the resident word gate does.
*Plumbed under the word gate from*: the post-col2im `ws`/`wprop` streams for
Conv, or `taint_out` of the scheduled tiled/small-K GEMM twins for Linear; these
produce `s_prod = fl(|A|@|W|)` and `prop = fl(E@|W|)`.
*Why it cannot be deferred to C1*: `min(err_out, e_eft)` is the ONLY operation
in the whole chain that can LOWER an error charge, and it happens mid-chain,
per element, never read back — a laundered `s_prod` here erases a degrade that
C1 would otherwise have seen. C1 catches what survives; C2 prevents the one op
that can un-survive it.

### C3 — `fl_value_gemm`: NO change required (G10 is already safe)

The single-kernel structure (final-write-only clamp, G6) makes the magnitude
test exact: the sentinel cannot be laundered before the guard because there is
no op between the saturation and the test. Keep the guard byte-identical.
(Optional hardening once the taint twin becomes the production GEMM: read the
word instead of the magnitude — it also covers any future kernel change that
clamps earlier — but this is uniformity, not a hole.)

### Explicitly NOT in the minimal set, and why

* **G1 (Higham combine)** — is a TRANSPORT node: its authored taint twin ORs
   input words and keeps the magnitude arms as its saturation seed. The
  AUTO/default resident route dispatches and threads it; ResNet seams OR the
  resulting per-row state. The terminal consumer is C1.
* **G3/G9 (fast path)** — diagnostics tier with no authority; leaving their
  magnitude/NaN tests unchanged is correct. The ladder (rung 5 +
  `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`) is what prevents this tier from
  ever carrying a verdict without the consult.
* **G8 (`resnet_bound_exploded`)** — C1 is armed and supplied with complete
  production row words, so a tainted row never reaches it because the
  typed `Err` propagates first. Optionally OR a per-row taint into the trigger
  for tightness (forcing the concretized-merge re-run instead of a refusal) — a
  quality improvement, not a soundness requirement.
* **G11/G12/G13 (ny-propagate ingestion)** — cannot see laundered values by
  construction (§2c). Their protection is the armed C1 typed refusal supplied
  with complete production row words. G13 pre-tainting covers admitted resident,
  ResNet-composed, and Linear/Activation host-swept routes; host Conv and other
  configurations without a complete channel refuse.
* **G14 (CPU compose)** — already strictly sticky; it is the floor the GPU
  rule must meet (and, on exact-zero annihilation, deliberately exceeds in
  tightness with the `!= 0` conjuncts).

### Rung-5 arming note

`verify_sentinel_taint_sticky` probes the shipped value sources for the modeled
GEMM → AW-combine → activation segment plus a parallel diagnostic word path;
its magnitude-only predicate is the historical end-of-segment lane proxy, not
a literal execution of C1 or optional C2. `ops/taint_chain.rs` executes all
three modeled transport twins end to end, and
`ops/eft_min_combine_taint_probe.rs` covers C2. Neither diagnostic path grants
authority. The main resident and ResNet integrations have separate through-walk
tests, and C1/C2 have focused refusal tests; the selfcheck is not a substitute
for them. The 2026-08-11 UTC review armed the source gate after those production
channels landed. The rung therefore selects the with-word verdict and measured
PASS on GB10/Vulkan + DenormPreserve, while its magnitude-only fields continue
to expose lanes 2 and 5 as laundering controls. A stuck word on an exact-zero
annihilation lane, missing C1 input, or any transport regression still refuses.

---

## 5. NaN cross-check summary (the `nan_safe_clamp` contract)

The contract (GEMM_F32_SHADER :794-806): NaN is PRESERVED through the value
kernels — "NaN propagates through subsequent backward steps and is caught at
concretize time". Verified consumer by consumer above; summary:

* Caught in-shader via `x != x` / `is_non_finite` bit tests: G1 (:2286 — note
  the ordering, the magnitude arm alone is NaN-blind), G2 (:2439, before the
  NaN-blind `>=`), G3 (:2072 + :2028-2035), G4 (:3086 + :3234-3235), G7
  (:1001).
* Caught host-side with compiler-immune bit patterns (the robust layer — WGSL
  NaN comparison semantics are implementation-defined, which is why the
  in-shader `x != x` patterns are additionally text-pinned by
  contract_tests.rs:570-597): G5 (:313, :331, :341 —
  `bits & 0x7f80_0000 == 0x7f80_0000` catches NaN AND Inf), G8
  (`!is_finite`), G9 (`!is_finite`), G10 (`!is_finite`), G13 (`!is_finite`).
* Caught with NaN-only or widen-only repairs, by design: G11 (`is_nan`, Inf
  legitimately passes to `Widen`), G12 (NaN → ±inf, never a finite clamp),
  plus the accelerated CPU-parallel sanitization
  (accelerated/crown_parallel.rs:175-185, NaN/Inf → ±FALLBACK_BOUND).

No guard in the table relies on a NaN surviving a WGSL `max`/`min` (the #2708
fix moved all such detection BEFORE the pos/neg splits — G3's doc comment,
shaders.rs:2049-2053).
