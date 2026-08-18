// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u1` SETTLING TEST — does the PRODUCTION `GEMM_F32_EFT_TWIN_SHADER` compile
//! to the sequence its residual channel claims to measure?
//!
//! # The obligation
//!
//! Every EFT probe already in the tree (`ops/eft_selfcheck.rs`,
//! `ops/subnormal_selfcheck.rs`, `ops/double_single_probe.rs`) is a
//! `workgroup_size(1)` straight-line kernel over a handful of scalar operands.
//! The kernel that actually publishes a certified radius is a `16×16`-tiled GEMM
//! with `var<workgroup>` tiles, `workgroupBarrier()`s, zero-padded out-of-range
//! taps and a per-thread loop over `ceil(k/16)·16` products. A passing scalar
//! probe does NOT prove that kernel compiled the same way: register allocation,
//! FMA contraction, or a reassociation of the barriered TwoSum inside a hot
//! tiled loop can differ from the straight-line form.
//!
//! The failure mode is SILENT UNDER-CHARGING. `CROWN_EFT_MIN_COMBINE_SHADER`
//! publishes `min(err_higham, err_eft)`, i.e. it PREFERS the smaller arm, so a
//! twin whose `R` comes back too small does not make the verifier slower — it
//! makes it WRONG. A twin whose `R` comes back too large is merely loose.
//!
//! # What this module does
//!
//! It runs the PRODUCTION pipeline object
//! (`resident_backward_pipelines().eft_twin`, compiled from the shipped shader
//! constant) through the PRODUCTION dispatch helper (`pass_simple_2d`) with the
//! PRODUCTION grid (`n.div_ceil(16) × m.div_ceil(16)`) and the production
//! `GemmParams` uniform, at CROWN-shaped `(m, k, n) = (num_specs, of, if_)`,
//! then bit-compares every `(V, R)` element against a CPU twin
//! ([`cpu_twin`]) that executes the identical tile-ordered sequence with
//! `f32::mul_add` for every `fma(...)`.
//!
//! Three properties make the comparison meaningful rather than decorative:
//!
//! 1. **Sentinel-filled outputs.** `v_out`/`r_out` are pre-filled with a NaN
//!    payload, so an element the grid never wrote is a mismatch, not a zero
//!    that happens to agree.
//! 2. **Negative controls.** The same GPU bytes are compared against
//!    DELIBERATELY DEGRADED CPU twins — one that drops the TwoSum residual, one
//!    that drops the TwoProduct residual, one that reverses the contraction
//!    order. A test that cannot tell those apart from the faithful twin proves
//!    nothing; the counts are reported per case.
//! 3. **Normal-range certification.** For the `Normal` family the CPU twin
//!    asserts that no intermediate (`prod`, `ep`, `es`, `acc`, `rsum`) is
//!    subnormal, so agreement/disagreement on those cases cannot be explained
//!    by this adapter's measured FTZ/DAZ (`ops/subnormal_selfcheck.rs`) — the
//!    composition question is isolated from the flush question.
//!
//! # MEASURED VERDICT — Apple M5 Max / Metal, 2026-08-06
//!
//! 10 CROWN-shaped cases, `57 527` output elements, `56.5M` taps:
//!
//! ```text
//!   V (value)    bit-exact  57527/57527   (100.0%)
//!   R (residual) bit-exact  44224/57527   ( 76.9%)
//!   R BELOW the CPU twin     6966/57527   ( 12.1%)   <-- the unsound direction
//!   worst |R| drift          2 ULP        (every case, k = 1 … 4096)
//!   run-to-run stable        yes          (every case)
//! ```
//!
//! So the settling test's headline answer is **NO — the production kernel does
//! not execute the modeled residual sequence bit-for-bit**, and the deviation
//! lands in the under-charging direction on 12% of elements. What it is NOT is a
//! wholesale re-association: the four whole-reduction hypotheses
//! (`RsumRightAssoc`, `RsumSplitChannels`, `RsumFourWay`, `RsumPerTile`) all
//! drift by 8–128 ULP, an order of magnitude more than the device does.
//!
//! [`tests::u1_prefix_scan_localizes_the_residual_drift`] pins down what the
//! device actually did, WITHOUT editing the shipped shader: for `k ≤ 16·t` every
//! tap past `k` is a zero-padded `0·0` contributing exact zeros, so dispatching
//! the PRODUCTION kernel at `k = 1, 2, …, 64` over prefixes of one operand set
//! reads out its own `rsum` after every tap. Over 65 536 observed steps
//! (32×32 elements = four FULLY OCCUPIED 16×16 workgroups, 4 tile boundaries):
//!
//! ```text
//!   (rsum + |ep|) + |es|   [the shipped source order]  63191   96.4%
//!   (rsum + |es|) + |ep|   [addends commuted]           1504    2.3%
//!   rsum + (|ep| + |es|)   [right-associated]            841    1.3%
//!   RN(rsum + |ep| + |es|) [one rounding / wider acc]      0
//!   UNEXPLAINED                                            0
//!   value chain V exact at every prefix length   65536/65536
//! ```
//!
//! Those three rules ENUMERATE every binary association of the three addends, so
//! `UNEXPLAINED = 0` is a complete characterization: **at every tap the device
//! adds the same two non-negative residual terms to the running sum, in one of
//! the orders a re-associating compiler is free to pick.** No term is dropped,
//! altered or duplicated. The value chain `acc = acc + prod` has no such freedom
//! (its `acc` is consumed by the barriered TwoSum), which is exactly why `V` is
//! bit-exact while `R` is not.
//!
//! # Why that is SOUND anyway — and what it rests on
//!
//! For `N` non-negative terms, ANY binary summation tree satisfies Higham's
//! order-independent `|fl − exact| ≤ γ_{N−1}·exact`. The twin sums `N = 2k` such
//! terms and the host applies `eft_r_slack(k) ≥ 1/(1−γ_{2k+2})`, so the recovery
//! never depended on the ORDER — only on the multiset, which the scan above
//! shows is preserved. Both facts are asserted per case, not assumed:
//! [`higham_envelope`] (multiset preserved) and [`enclosure`] (`R·r_slack`
//! encloses the exact residual sum). Measured headroom:
//!
//! ```text
//!   k=15    deviation 9.9e-8  vs gamma 1.7e-6   5.7% of budget
//!   k=256   deviation 9.4e-7  vs gamma 3.0e-5   3.1% of budget
//!   k=1152  deviation 3.4e-6  vs gamma 1.4e-4   2.5% of budget
//!   k=4096  deviation 9.7e-6  vs gamma 4.9e-4   2.0% of budget
//!   enclosure violations: 0 / 57527 on every case
//! ```
//!
//! The budget share SHRINKS as `k` grows, so the margin is not a small-`k`
//! accident. (U3 asks whether `eft_r_slack`'s `(1+u)^6` headroom is enough for
//! the extra op-count of the tree kernels; this measurement says the twin GEMM's
//! own accumulation consumes only 2–6% of the `γ_{2k+2}` term, leaving the rest
//! of that budget for U3 to spend.)
//!
//! # What this does NOT settle
//!
//! * The `min(higham, eft)` combine's own arithmetic, and the `flush` floor
//!   inside it (`#daz-flush-cover-v2`, `flush_charge_oracle.rs`).
//! * `CONV_COL2IM_EFT_TWIN_SHADER` and the `eft_mode=1` arm of
//!   `CROWN_BIAS_ERR_ACCUMULATE_SHADER` (a 256-thread strided loop + 8-level
//!   `var<workgroup>` tree). Both accumulate residuals with the same
//!   `rsum + term + |es|` shape, so the same re-association is expected there —
//!   but expected is not measured, and the tree kernel's term COUNT is `U3`'s
//!   open question, not this file's.
//! * Anything about a non-Metal adapter. The characterization is per-adapter and
//!   must be re-run on CUDA before the compensated channel is trusted there.
//!
//! Wired into NO verdict path: the module compiles only under
//! `cfg(all(test, feature = "gpu-tests"))`, exactly like `double_single_probe`.
//! It authorizes nothing; it is evidence.

use ny_core::Result;

use super::super::params::GemmParams;
use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;

/// 2^-126, the smallest f32 NORMAL — the shader's flush-safe residual charge.
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
/// 2^-101 — below this product magnitude the TwoProdFMA residual may itself
/// round, and the shader substitutes [`F32_MIN_NORMAL`].
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31;
/// Tile edge of the production twin. MUST match `TILE` in the shipped WGSL.
const TILE: usize = 16;
/// NaN payload pre-written into `v_out`/`r_out`: any element the production
/// grid fails to write shows up as a mismatch instead of a plausible zero.
const UNWRITTEN_SENTINEL: u32 = 0x7FC0_1234;

/// Which sequence the CPU twin executes. `Faithful` is the shipped one; the
/// rest are NEGATIVE CONTROLS — the bit-compare must REJECT them, otherwise it
/// is not discriminating and its agreement means nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum TwinMode {
    /// Line-by-line transcription of `GEMM_F32_EFT_TWIN_SHADER`.
    Faithful,
    /// Drops the fma-barrier TwoSum residual `|es|` from `rsum` — precisely
    /// what a compiler that folded the algebraically-zero compensation term
    /// would produce, and the headline under-charging risk.
    DropTwoSumResidual,
    /// Drops the TwoProduct residual `|ep|` from `rsum` — what a
    /// non-fused-multiply-add lowering would produce.
    DropTwoProdResidual,
    /// Same taps, reversed contraction order: detects whether the bit-compare
    /// is sensitive to the ORDER the device actually executed.
    ReverseK,
    /// Skips the zero-padded out-of-range taps entirely (loops `j < k` instead
    /// of `j < ceil(k/16)·16`). Pins the shader comment's claim that padded
    /// taps "contribute exact zeros and cost no residual".
    NoPadTaps,
    /// HYPOTHESIS: the residual accumulation is re-associated to
    /// `rsum + (eterm + |es|)`. Value chain unchanged.
    RsumRightAssoc,
    /// HYPOTHESIS: the two residual channels are accumulated into SEPARATE
    /// registers and added once at the end (`Σ|ep| + Σ|es|`).
    RsumSplitChannels,
    /// HYPOTHESIS: the accumulation is 4-way interleaved over the term stream.
    RsumFourWay,
    /// HYPOTHESIS: partial residual per 16-tap tile, folded into `rsum` once
    /// per tile (the tile loop is the natural unrolling boundary).
    RsumPerTile,
}

/// What the faithful CPU twin observed while executing the sequence — the
/// evidence that a given case can DISCRIMINATE, and the U3-adjacent
/// measurement of how much the f32 `rsum` accumulation itself lost.
#[derive(Clone, Debug, Default)]
pub(crate) struct TwinStats {
    /// Taps whose TwoProduct residual was nonzero.
    pub(crate) taps_nonzero_ep: u64,
    /// Taps whose fma-barrier TwoSum residual was nonzero.
    pub(crate) taps_nonzero_es: u64,
    /// Taps that took the `< 2^-101` exactness-floor branch.
    pub(crate) taps_floor_charged: u64,
    /// Total taps executed (including zero-padded ones).
    pub(crate) taps_total: u64,
    /// Intermediates (`prod`, `ep`, `es`, `acc`, `rsum`) that were subnormal.
    /// MUST be 0 for the `Normal` family, else the flush question contaminates
    /// the composition question.
    pub(crate) subnormal_intermediates: u64,
    /// Smallest nonzero magnitude seen among those intermediates.
    pub(crate) min_nonzero_intermediate: f32,
    /// `max over elements of (Σ|ep|+|es| in f64) / (f32-accumulated R)` — the
    /// deficit the host's `eft_r_slack` must cover (U3 diagnostic).
    pub(crate) max_r_accum_deficit_ratio: f64,
    /// Per-element EXACT residual sum (Neumaier-compensated f64 over the exact
    /// f32 term stream). This is the quantity the published radius must
    /// enclose, independent of any accumulation order.
    pub(crate) r_exact: Vec<f64>,
}

/// Per-element agreement between the production twin and the CPU twin.
#[derive(Clone, Debug)]
pub(crate) struct TwinAgreement {
    pub(crate) elems: usize,
    /// Elements whose value word `V` matched BIT-EXACTLY.
    pub(crate) v_bit_exact: usize,
    /// Elements whose residual word `R` matched BIT-EXACTLY.
    pub(crate) r_bit_exact: usize,
    /// Elements where the device published a SMALLER residual than the CPU
    /// twin — the wrong-verdict direction. Non-finite device values count here.
    pub(crate) r_under: usize,
    /// Elements where the device published a LARGER residual (merely loose).
    pub(crate) r_over: usize,
    /// Worst `R_cpu / R_gpu` over elements where the device under-charged.
    pub(crate) worst_under_ratio: f64,
    /// Elements whose CPU residual is nonzero (discriminating power: an
    /// all-zero-residual case would agree vacuously).
    pub(crate) r_nonzero: usize,
    /// Largest `|R_gpu − R_cpu|` measured in f32 ULPs (both finite, both
    /// non-negative ⇒ the bit-pattern difference IS the ULP distance).
    pub(crate) max_abs_ulp_delta: i64,
    /// Largest ULP distance in the UNDER-charging direction.
    pub(crate) max_under_ulp: i64,
}

/// One CROWN-shaped case.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TwinCase {
    pub(crate) name: &'static str,
    /// `m = num_specs`, `k = of`, `n = if_` — the production mapping in
    /// `crown_backward_sound_resident.rs`.
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) seed: u64,
    pub(crate) family: Family,
}

/// Magnitude regime of the generated operands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    /// Every intermediate normal-range: isolates COMPOSITION from FLUSH.
    Normal,
    /// Alternating-sign rows engineered for catastrophic cancellation, still
    /// normal-range: maximizes nonzero TwoSum residuals.
    Cancelling,
    /// Deliberately near-underflow: the twin's residuals land in the subnormal
    /// band, so on a flushing adapter this case is EXPECTED to disagree. Kept
    /// separate so it can never be confused with a composition failure.
    FlushExposed,
}

// ---------------------------------------------------------------------------
// CPU twin — a line-by-line transcription of GEMM_F32_EFT_TWIN_SHADER
// ---------------------------------------------------------------------------

/// Execute the shipped twin sequence on the host for every output element.
///
/// Fidelity notes (each maps to one WGSL line):
/// * `prod = a * w`                     → `a * w`
/// * `ep   = fma(a, w, -prod)`          → `a.mul_add(w, -prod)`
/// * the `< 2^-101` guard and its `F32_MIN_NORMAL` charge
/// * `s = acc + prod` then the five-`fma` barriered TwoSum
/// * `rsum = rsum + eterm + abs(es)` (LEFT-associative, as in WGSL)
/// * the tile loop's zero-padded taps for `j >= k`
///
/// Rust never reassociates or contracts f32 arithmetic, and `mul_add` lowers to
/// a true single-rounded FMA on this host (pinned by `ny_core::eft`'s own
/// self-check, asserted by the caller).
pub(crate) fn cpu_twin(
    a: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TwinStats) {
    let mut v_out = vec![0.0f32; m * n];
    let mut r_out = vec![0.0f32; m * n];
    let mut r_exact_out: Vec<f64> = Vec::with_capacity(m * n);
    let mut stats = TwinStats {
        min_nonzero_intermediate: f32::INFINITY,
        ..Default::default()
    };
    let padded_k = k.div_ceil(TILE) * TILE;
    let taps = if mode == TwinMode::NoPadTaps {
        k
    } else {
        padded_k
    };

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            let mut rsum = 0.0f32;
            // Extra accumulators for the re-association hypotheses.
            let mut lanes = [0.0f32; 4];
            let mut tile_r = 0.0f32;
            let mut r_exact = 0.0f64;
            let mut r_comp = 0.0f64; // Neumaier compensation for r_exact
            for step in 0..taps {
                // Tile order: thread (row,col) sees j = t*16 + kk ascending.
                let j = if mode == TwinMode::ReverseK {
                    taps - 1 - step
                } else {
                    step
                };
                // Zero-padded out-of-range taps (`a_col < k` / `b_row < k`).
                let av = if j < k { a[row * k + j] } else { 0.0 };
                let wv = if j < k { w[j * n + col] } else { 0.0 };

                let prod = av * wv;
                let ep = av.mul_add(wv, -prod);
                let mut eterm = ep.abs();
                if av != 0.0 && wv != 0.0 && prod.abs() < TWO_PROD_EXACT_FLOOR_F32 {
                    eterm = F32_MIN_NORMAL;
                    stats.taps_floor_charged += 1;
                }

                let s = acc + prod;
                let bb = (-1.0f32).mul_add(acc, s);
                let sb = (-1.0f32).mul_add(bb, s);
                let da = (-1.0f32).mul_add(sb, acc);
                let db = (-1.0f32).mul_add(bb, prod);
                let es = da + db;

                let charged_ep = if mode == TwinMode::DropTwoProdResidual {
                    0.0
                } else {
                    eterm
                };
                let charged_es = if mode == TwinMode::DropTwoSumResidual {
                    0.0
                } else {
                    es.abs()
                };
                match mode {
                    TwinMode::RsumRightAssoc => rsum += charged_ep + charged_es,
                    TwinMode::RsumSplitChannels => {
                        lanes[0] += charged_ep;
                        lanes[1] += charged_es;
                    }
                    TwinMode::RsumFourWay => {
                        let b = (2 * step) % 4;
                        lanes[b] += charged_ep;
                        lanes[(b + 1) % 4] += charged_es;
                    }
                    TwinMode::RsumPerTile => {
                        tile_r = tile_r + charged_ep + charged_es;
                        if step % TILE == TILE - 1 || step + 1 == taps {
                            rsum += tile_r;
                            tile_r = 0.0;
                        }
                    }
                    _ => rsum = rsum + charged_ep + charged_es,
                }
                acc = s;

                if mode == TwinMode::Faithful {
                    stats.taps_total += 1;
                    if ep != 0.0 {
                        stats.taps_nonzero_ep += 1;
                    }
                    if es != 0.0 {
                        stats.taps_nonzero_es += 1;
                    }
                    // Neumaier-compensated f64 accumulation of the EXACT term
                    // stream: the reference the published radius must enclose.
                    for term in [f64::from(eterm), f64::from(es.abs())] {
                        let t = r_exact + term;
                        r_comp += if r_exact.abs() >= term.abs() {
                            (r_exact - t) + term
                        } else {
                            (term - t) + r_exact
                        };
                        r_exact = t;
                    }
                    for v in [prod, ep, es, acc, rsum] {
                        if v != 0.0 && v.abs() < F32_MIN_NORMAL {
                            stats.subnormal_intermediates += 1;
                        }
                        if v != 0.0 && v.abs() < stats.min_nonzero_intermediate {
                            stats.min_nonzero_intermediate = v.abs();
                        }
                    }
                }
            }
            match mode {
                TwinMode::RsumSplitChannels => rsum = lanes[0] + lanes[1],
                TwinMode::RsumFourWay => rsum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]),
                _ => {}
            }
            let r_exact_total = r_exact + r_comp;
            if mode == TwinMode::Faithful && rsum > 0.0 {
                let ratio = r_exact_total / f64::from(rsum);
                if ratio > stats.max_r_accum_deficit_ratio {
                    stats.max_r_accum_deficit_ratio = ratio;
                }
            }
            v_out[row * n + col] = acc;
            r_out[row * n + col] = rsum;
            if mode == TwinMode::Faithful {
                r_exact_out.push(r_exact_total);
            }
        }
    }
    stats.r_exact = r_exact_out;
    (v_out, r_out, stats)
}

// ---------------------------------------------------------------------------
// GPU side — the PRODUCTION pipeline, dispatch helper and grid
// ---------------------------------------------------------------------------

impl WgpuDevice {
    /// Dispatch the production `#eft-err` twin GEMM and read `(V, R)` back as
    /// raw bit patterns (no float load that could canonicalize a NaN).
    pub(crate) fn u1_run_production_twin(
        &self,
        a: &[f32],
        w: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        assert_eq!(a.len(), m * k, "A is m×k row-major");
        assert_eq!(w.len(), k * n, "W is k×n row-major");
        self.run_gpu_checked("u1_twin_composition", || {
            let out_len = m * n;
            let out_bytes = (out_len * size_of::<f32>()) as u64;

            let params = GemmParams {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _padding: 0,
            };
            let p_buf = create_buffer(
                &self.device,
                "u1_twin_params",
                size_of::<GemmParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            self.queue
                .write_buffer(&p_buf, 0, bytemuck::bytes_of(&params));

            let a_buf = create_buffer(
                &self.device,
                "u1_twin_a",
                size_of_val(a) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(a));
            let w_buf = create_buffer(
                &self.device,
                "u1_twin_w",
                size_of_val(w) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.queue.write_buffer(&w_buf, 0, bytemuck::cast_slice(w));

            // Sentinel pre-fill: an element the production grid never writes is
            // then a MISMATCH, not an accidental agreement on zero.
            let sentinel = vec![UNWRITTEN_SENTINEL; out_len];
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let v_buf = create_buffer(&self.device, "u1_twin_v", out_bytes, usage);
            let r_buf = create_buffer(&self.device, "u1_twin_r", out_bytes, usage);
            self.queue
                .write_buffer(&v_buf, 0, bytemuck::cast_slice(&sentinel));
            self.queue
                .write_buffer(&r_buf, 0, bytemuck::cast_slice(&sentinel));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("u1_twin_encoder"),
                });
            // PRODUCTION pipeline object + PRODUCTION dispatch helper + the
            // PRODUCTION grid (x over columns, y over rows).
            let pipes = self.resident_backward_pipelines();
            self.pass_simple_2d(
                &mut encoder,
                &pipes.eft_twin,
                &p_buf,
                &[&a_buf, &w_buf, &v_buf, &r_buf],
                (n as u32).div_ceil(16),
                (m as u32).div_ceil(16),
            );

            let v_stage = create_buffer(
                &self.device,
                "u1_twin_v_stage",
                out_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let r_stage = create_buffer(
                &self.device,
                "u1_twin_r_stage",
                out_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(&v_buf, 0, &v_stage, 0, out_bytes);
            encoder.copy_buffer_to_buffer(&r_buf, 0, &r_stage, 0, out_bytes);
            self.queue.submit(std::iter::once(encoder.finish()));

            let v = WgpuDevice::read_u32_buffer(&self.device, &v_stage, out_len)?;
            let r = WgpuDevice::read_u32_buffer(&self.device, &r_stage, out_len)?;
            Ok((v, r))
        })
    }
}

// ---------------------------------------------------------------------------
// Comparison + data generation
// ---------------------------------------------------------------------------

/// Bit-compare device words against a CPU twin, and classify every residual
/// disagreement by DIRECTION (under-charging is the unsound one).
pub(crate) fn compare(v_gpu: &[u32], r_gpu: &[u32], v_cpu: &[f32], r_cpu: &[f32]) -> TwinAgreement {
    let mut ag = TwinAgreement {
        elems: v_cpu.len(),
        v_bit_exact: 0,
        r_bit_exact: 0,
        r_under: 0,
        r_over: 0,
        worst_under_ratio: 1.0,
        r_nonzero: 0,
        max_abs_ulp_delta: 0,
        max_under_ulp: 0,
    };
    for i in 0..v_cpu.len() {
        if v_gpu[i] == v_cpu[i].to_bits() {
            ag.v_bit_exact += 1;
        }
        if r_gpu[i] == r_cpu[i].to_bits() {
            ag.r_bit_exact += 1;
        }
        if r_cpu[i] != 0.0 {
            ag.r_nonzero += 1;
        }
        let rg = f32::from_bits(r_gpu[i]);
        let rc = r_cpu[i];
        if !rg.is_finite() {
            // A NaN/Inf residual cannot be trusted to charge anything: count it
            // with the unsound direction (fail-closed accounting).
            ag.r_under += 1;
            ag.worst_under_ratio = f64::INFINITY;
        } else if rg < rc {
            ag.r_under += 1;
            if rg > 0.0 {
                let ratio = f64::from(rc) / f64::from(rg);
                if ratio > ag.worst_under_ratio {
                    ag.worst_under_ratio = ratio;
                }
            } else if rc > 0.0 {
                ag.worst_under_ratio = f64::INFINITY;
            }
        } else if rg > rc {
            ag.r_over += 1;
        }
        // ULP distance. Both residual sums are non-negative finite f32, so the
        // raw bit-pattern difference is exactly the number of representable
        // steps between them.
        if rg.is_finite() && rg >= 0.0 && rc >= 0.0 {
            let delta = i64::from(r_gpu[i]) - i64::from(rc.to_bits());
            if delta.abs() > ag.max_abs_ulp_delta {
                ag.max_abs_ulp_delta = delta.abs();
            }
            if -delta > ag.max_under_ulp {
                ag.max_under_ulp = -delta;
            }
        }
    }
    ag
}

/// Does the PUBLISHED residual, scaled by the host's `eft_r_slack(k)`, still
/// enclose the EXACT residual sum? This is the question that actually decides
/// soundness: the min-combine ships `(R + d)·r_slack + …`, and `R` is whatever
/// the device's accumulation order produced — bit-exact to the CPU model or not.
///
/// Returns `(violations, worst_margin)` where `worst_margin` is the smallest
/// `R_gpu·r_slack / R_exact` over elements with a nonzero exact residual
/// (`> 1` ⇒ encloses, with that much headroom to spare).
pub(crate) fn enclosure(r_gpu: &[u32], r_exact: &[f64], r_slack: f32) -> (usize, f64) {
    let mut violations = 0usize;
    let mut worst = f64::INFINITY;
    for (i, &exact) in r_exact.iter().enumerate() {
        let rg = f64::from(f32::from_bits(r_gpu[i]));
        if exact <= 0.0 {
            continue;
        }
        let margin = rg * f64::from(r_slack) / exact;
        if margin < worst {
            worst = margin;
        }
        if margin < 1.0 {
            violations += 1;
        }
    }
    (violations, worst)
}

/// Is the device's `R` inside the ORDER-INDEPENDENT Higham envelope of the exact
/// term sum, `|fl − exact| ≤ γ_{N−1}·exact` for `N = 2k` non-negative terms?
///
/// This is the assertion that survives a driver change: any binary summation
/// tree over the same multiset of `2k` non-negative terms satisfies it, so a
/// device that merely RE-ASSOCIATES stays inside, while one that drops,
/// duplicates or alters a term leaves it.
///
/// Returns `(outside_below, outside_above, worst_relative_deviation)`.
pub(crate) fn higham_envelope(r_gpu: &[u32], r_exact: &[f64], gamma: f64) -> (usize, usize, f64) {
    let mut below = 0usize;
    let mut above = 0usize;
    let mut worst = 0.0f64;
    for (i, &exact) in r_exact.iter().enumerate() {
        if exact <= 0.0 {
            continue;
        }
        let rg = f64::from(f32::from_bits(r_gpu[i]));
        let rel = (rg - exact) / exact;
        if rel.abs() > worst {
            worst = rel.abs();
        }
        if rel < -gamma {
            below += 1;
        } else if rel > gamma {
            above += 1;
        }
    }
    (below, above, worst)
}

/// Deterministic xorshift64* — reproducible operands, no dev-dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        (f64::from(bits) / f64::from(1u32 << 23) - 1.0) as f32
    }
    fn exp_scale(&mut self, lo: i32, hi: i32) -> f32 {
        let span = (hi - lo + 1) as u64;
        let e = lo + (self.next_u64() % span) as i32;
        f32::from_bits(((127 + e) as u32) << 23)
    }
}

/// CROWN-shaped operands for one case.
pub(crate) fn make_operands(case: &TwinCase) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(case.seed | 1);
    let mut a = vec![0.0f32; case.m * case.k];
    let mut w = vec![0.0f32; case.k * case.n];
    match case.family {
        Family::Normal => {
            for x in a.iter_mut() {
                *x = rng.unit() * rng.exp_scale(-6, 4);
            }
            for x in w.iter_mut() {
                *x = rng.unit() * rng.exp_scale(-8, 2);
            }
        }
        Family::Cancelling => {
            // Alternating signs with a wide exponent spread: `acc` repeatedly
            // crosses zero, so the barriered TwoSum residual is nonzero on
            // nearly every tap and catastrophic cancellation is the norm.
            for (i, x) in a.iter_mut().enumerate() {
                let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                *x = s * (0.5 + 0.5 * rng.unit().abs()) * rng.exp_scale(-10, 8);
            }
            for (i, x) in w.iter_mut().enumerate() {
                let s = if i % 3 == 0 { -1.0 } else { 1.0 };
                *x = s * (0.5 + 0.5 * rng.unit().abs()) * rng.exp_scale(-6, 6);
            }
        }
        Family::FlushExposed => {
            // Products land near 2^-126: the residuals themselves are subnormal.
            for x in a.iter_mut() {
                *x = rng.unit() * rng.exp_scale(-60, -55);
            }
            for x in w.iter_mut() {
                *x = rng.unit() * rng.exp_scale(-70, -64);
            }
        }
    }
    (a, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};

    /// CROWN-shaped `(m, k, n) = (num_specs, of, if_)`, plus the ragged shapes
    /// that exercise partial tiles, zero-padding and a `k` below one tile.
    const CASES: &[TwinCase] = &[
        TwinCase {
            name: "mnist-fc",
            m: 10,
            k: 256,
            n: 256,
            seed: 0x1111,
            family: Family::Normal,
        },
        TwinCase {
            name: "cifar-fc",
            m: 9,
            k: 512,
            n: 1024,
            seed: 0x2222,
            family: Family::Normal,
        },
        TwinCase {
            name: "conv-cols",
            m: 72,
            k: 1152,
            n: 576,
            seed: 0x3333,
            family: Family::Normal,
        },
        TwinCase {
            name: "deep-k",
            m: 8,
            k: 4096,
            n: 64,
            seed: 0x4444,
            family: Family::Normal,
        },
        TwinCase {
            name: "ragged",
            m: 13,
            k: 1000,
            n: 37,
            seed: 0x5555,
            family: Family::Normal,
        },
        TwinCase {
            name: "k1",
            m: 17,
            k: 1,
            n: 19,
            seed: 0x6666,
            family: Family::Normal,
        },
        TwinCase {
            name: "k15",
            m: 5,
            k: 15,
            n: 5,
            seed: 0x7777,
            family: Family::Normal,
        },
        TwinCase {
            name: "k17",
            m: 5,
            k: 17,
            n: 5,
            seed: 0x8888,
            family: Family::Normal,
        },
        TwinCase {
            name: "cancel-fc",
            m: 10,
            k: 512,
            n: 128,
            seed: 0x9999,
            family: Family::Cancelling,
        },
        TwinCase {
            name: "cancel-ragged",
            m: 23,
            k: 333,
            n: 71,
            seed: 0xAAAA,
            family: Family::Cancelling,
        },
    ];

    const FLUSH_CASE: TwinCase = TwinCase {
        name: "flush-exposed",
        m: 5,
        k: 64,
        n: 5,
        seed: 0xBBBB,
        family: Family::FlushExposed,
    };

    /// `#u1` THE SETTLING TEST. Per-element bit-compare of the production twin
    /// `(V, R)` against the CPU twin executing the identical sequence, at
    /// CROWN-shaped `(m, k, n)`, with negative controls and a repeat run.
    ///
    /// MEASURED VERDICT on Apple M5 Max / Metal: the VALUE word `V` is bit-exact
    /// everywhere, and the RESIDUAL word `R` is NOT — the device's `rsum`
    /// accumulation drifts by a bounded number of ULPs in BOTH directions. The
    /// value chain `acc = acc + prod` is a true serial dependency (its `acc` is
    /// consumed by the barriered TwoSum), so no reassociation is available to
    /// the compiler; `rsum` is a pure reduction with no other consumer, so it
    /// is free to be reassociated/vectorized — and measurably is.
    ///
    /// The assertions below are therefore split into the two questions that
    /// matter separately:
    ///   * COMPOSITION (reported, pinned): how far the device sequence drifts
    ///     from the modeled one, in ULPs and in element counts.
    ///   * SOUNDNESS (asserted): does `R_gpu · eft_r_slack(k)` still ENCLOSE the
    ///     exact residual sum `Σ|ep| + Σ|es|`? A reassociated sum of `2k`
    ///     non-negative terms still obeys Higham's order-independent
    ///     `|fl − exact| ≤ γ_{2k−1}·exact`, so it should — and the measured
    ///     margin says by how much.
    #[test]
    fn u1_production_twin_matches_cpu_twin_bit_exactly() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        // The host reference must itself be sound (true FMA, RN, no FTZ), else
        // the comparison means nothing.
        ny_core::eft::eft_self_check().expect("host EFT reference must be sound");

        println!(
            "\n=== #u1 production-twin composition, adapter: {} ({:?}) ===",
            device.adapter_info.name, device.adapter_info.backend
        );
        let mut all_exact = true;
        let mut total_elems = 0usize;
        let mut total_v_exact = 0usize;
        let mut total_r_exact_bits = 0usize;
        let mut total_r_under = 0usize;
        let mut worst_drift = 0i64;
        for case in CASES {
            let (a, w) = make_operands(case);
            let (v_gpu, r_gpu) = device
                .u1_run_production_twin(&a, &w, case.m, case.k, case.n)
                .expect("production twin dispatch");
            let (v_cpu, r_cpu, stats) =
                cpu_twin(&a, &w, case.m, case.k, case.n, TwinMode::Faithful);
            let ag = compare(&v_gpu, &r_gpu, &v_cpu, &r_cpu);

            // Normal-range certification: the flush question must not be able
            // to explain anything on these cases.
            assert_eq!(
                stats.subnormal_intermediates, 0,
                "{}: case is not normal-range; composition and flush are confounded",
                case.name
            );

            // Negative controls + re-association hypotheses on the SAME device
            // bytes. The controls must be REJECTED; a hypothesis that is
            // accepted everywhere identifies the sequence the device ran.
            let mut controls = Vec::new();
            for mode in [
                TwinMode::DropTwoSumResidual,
                TwinMode::DropTwoProdResidual,
                TwinMode::ReverseK,
                TwinMode::NoPadTaps,
                TwinMode::RsumRightAssoc,
                TwinMode::RsumSplitChannels,
                TwinMode::RsumFourWay,
                TwinMode::RsumPerTile,
            ] {
                let (vc, rc, _) = cpu_twin(&a, &w, case.m, case.k, case.n, mode);
                let cag = compare(&v_gpu, &r_gpu, &vc, &rc);
                controls.push((mode, cag));
            }

            // Repeat run: a dropped/mis-placed workgroup barrier shows up as
            // run-to-run instability on the tiled kernel.
            let (v2, r2) = device
                .u1_run_production_twin(&a, &w, case.m, case.k, case.n)
                .expect("production twin repeat dispatch");
            let stable = v2 == v_gpu && r2 == r_gpu;

            println!(
                "[{name:<14}] m={m:<3} k={k:<5} n={n:<5} elems={e:<7} \
                 V bit-exact {v}/{e}  R bit-exact {r}/{e}  R<cpu {u}  R>cpu {o}  \
                 R!=0 {nz}/{e}  repeat-stable={stable}",
                name = case.name,
                m = case.m,
                k = case.k,
                n = case.n,
                e = ag.elems,
                v = ag.v_bit_exact,
                r = ag.r_bit_exact,
                u = ag.r_under,
                o = ag.r_over,
                nz = ag.r_nonzero,
            );
            println!(
                "                 taps={t} nonzero-ep={ep} nonzero-es={es} floor-charged={fc} \
                 min|intermediate|={mn:e}  max|ULP drift|={ud} (under {uu})  \
                 f32-accum deficit x{def:.9}",
                t = stats.taps_total,
                ep = stats.taps_nonzero_ep,
                es = stats.taps_nonzero_es,
                fc = stats.taps_floor_charged,
                mn = stats.min_nonzero_intermediate,
                ud = ag.max_abs_ulp_delta,
                uu = ag.max_under_ulp,
                def = stats.max_r_accum_deficit_ratio,
            );
            for (mode, cag) in &controls {
                println!(
                    "                 alt {mode:?}: R bit-exact {r}/{e}  max|ULP|={u}",
                    r = cag.r_bit_exact,
                    e = cag.elems,
                    u = cag.max_abs_ulp_delta,
                );
            }

            // SOUNDNESS: does the PUBLISHED residual, scaled by the host slack
            // the min-combine actually applies, enclose the exact residual sum?
            let r_slack = crate::wgpu_device::sound_consts::eft_r_slack_f32(case.k)
                .expect("eft_r_slack for a CROWN-shaped k");
            let (violations, worst_margin) = enclosure(&r_gpu, &stats.r_exact, r_slack);
            // Order-independent Higham envelope for a sum of 2k non-negative
            // terms: ANY binary summation tree obeys |fl − exact| ≤ γ_{2k−1}.
            let terms = 2 * case.k;
            let gamma = f64::from(
                crate::wgpu_device::sound_consts::gamma_k_f32(terms - 1)
                    .expect("gamma for 2k-1 terms"),
            );
            let (below, above, worst_rel) = higham_envelope(&r_gpu, &stats.r_exact, gamma);
            println!(
                "                 ENCLOSURE R_gpu·r_slack vs exact: r_slack={rs:.9} \
                 violations={v}/{e} worst margin x{wm:.9}",
                rs = r_slack,
                v = violations,
                e = ag.elems,
                wm = worst_margin,
            );
            println!(
                "                 ENVELOPE |R_gpu/R_exact − 1| <= gamma_{{2k−1}}={g:.3e}: \
                 outside below={b} above={ab}  worst observed={wr:.3e} \
                 (uses {frac:.2}% of the budget)",
                g = gamma,
                b = below,
                ab = above,
                wr = worst_rel,
                frac = 100.0 * worst_rel / gamma,
            );

            assert!(
                stable,
                "{}: production twin is not run-to-run stable",
                case.name
            );
            if ag.v_bit_exact != ag.elems || ag.r_bit_exact != ag.elems {
                all_exact = false;
            }
            total_elems += ag.elems;
            total_v_exact += ag.v_bit_exact;
            total_r_exact_bits += ag.r_bit_exact;
            total_r_under += ag.r_under;
            worst_drift = worst_drift.max(ag.max_abs_ulp_delta);
            // The VALUE word has no reassociation freedom and must be exact.
            assert_eq!(
                ag.v_bit_exact, ag.elems,
                "{}: value word V is not bit-exact",
                case.name
            );
            // The published radius must enclose regardless of accumulation order.
            assert_eq!(
                violations, 0,
                "{}: R_gpu·r_slack FAILS to enclose the exact residual on {} elements \
                 (worst margin {:.9})",
                case.name, violations, worst_margin
            );
            // …and it must do so for the REASON claimed: the device is summing
            // the same 2k non-negative terms in some order. A dropped or
            // altered term leaves this envelope; a re-association cannot.
            assert_eq!(
                (below, above),
                (0, 0),
                "{}: R_gpu left the order-independent Higham envelope \
                 ({} below / {} above, worst relative deviation {:.3e} vs gamma {:.3e}) — \
                 the device is NOT summing the same term multiset",
                case.name,
                below,
                above,
                worst_rel,
                gamma
            );

            // Discriminating power: a case whose residuals are all zero proves
            // nothing, and a control the compare cannot reject proves nothing.
            assert!(
                ag.r_nonzero * 2 >= ag.elems,
                "{}: fewer than half the elements have a nonzero residual — \
                 this case cannot discriminate",
                case.name
            );
            let drop_es = controls
                .iter()
                .find(|(m, _)| *m == TwinMode::DropTwoSumResidual)
                .map(|(_, c)| c.r_bit_exact)
                .unwrap_or(usize::MAX);
            // Only meaningful where the sequence HAS TwoSum residuals: at k=1
            // the accumulator starts at 0, so every `es` is exactly 0 and the
            // control is legitimately indistinguishable.
            assert!(
                stats.taps_nonzero_es == 0 || drop_es * 10 < ag.elems,
                "{}: the drop-TwoSum-residual control still matches on {}/{} elements — \
                 the bit-compare is not discriminating",
                case.name,
                drop_es,
                ag.elems
            );
        }
        println!(
            "\n--- #u1 VERDICT over {c} CROWN-shaped cases ---\n\
             elements compared        : {total_elems}\n\
             V bit-exact              : {total_v_exact}/{total_elems}\n\
             R bit-exact              : {total_r_exact_bits}/{total_elems}\n\
             R below the CPU twin     : {total_r_under}/{total_elems}\n\
             worst |R| drift          : {worst_drift} ULP\n\
             every case fully exact   : {all_exact}",
            c = CASES.len(),
        );
        // PIN. The measured state on this adapter is: V bit-exact everywhere,
        // R re-associated within a couple of ULPs, always inside the
        // order-independent Higham envelope (asserted per case above). The
        // bound below is not a soundness claim — the envelope check is — but a
        // TRIPWIRE: a driver that starts drifting by orders of magnitude more
        // has changed the composition, and this finding must be re-derived
        // before the compensated channel is trusted on it.
        const PINNED_MAX_ULP_DRIFT: i64 = 8;
        assert!(
            worst_drift <= PINNED_MAX_ULP_DRIFT,
            "#u1: residual drift grew to {worst_drift} ULP (pinned <= {PINNED_MAX_ULP_DRIFT}); \
             the production kernel's composition changed — re-derive before trusting it"
        );
    }

    /// Companion conformance observation, NOT a composition claim: on an adapter with
    /// uncharged subnormal flushing, the twin's own residual channel is expected
    /// to disagree with a gradual-underflow host once residuals enter that band.
    /// Recorded (never asserted bit-exact) so the flush arc's verdict and this
    /// one can never be confused for each other.
    #[test]
    fn u1_flush_exposed_case_preserves_value_bits_and_exercises_subnormals() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let case = FLUSH_CASE;
        let (a, w) = make_operands(&case);
        let (v_gpu, r_gpu) = device
            .u1_run_production_twin(&a, &w, case.m, case.k, case.n)
            .expect("production twin dispatch");
        let (v_cpu, r_cpu, stats) = cpu_twin(&a, &w, case.m, case.k, case.n, TwinMode::Faithful);
        let ag = compare(&v_gpu, &r_gpu, &v_cpu, &r_cpu);
        assert!(
            ag.elems > 0,
            "flush-exposed fixture must exercise device output"
        );
        assert_eq!(
            ag.v_bit_exact, ag.elems,
            "the production value channel must remain bit-exact in the flush-exposed fixture"
        );
        assert!(
            stats.subnormal_intermediates > 0,
            "flush-exposed fixture must actually create subnormal intermediates"
        );
        println!(
            "[flush-exposed ] elems={e} V bit-exact {v}/{e} R bit-exact {r}/{e} \
             R<cpu {u} R>cpu {o} subnormal-intermediates={s} floor-charged={fc}",
            e = ag.elems,
            v = ag.v_bit_exact,
            r = ag.r_bit_exact,
            u = ag.r_under,
            o = ag.r_over,
            s = stats.subnormal_intermediates,
            fc = stats.taps_floor_charged,
        );
        if ag.r_under > 0 {
            println!(
                "                 UNDER-CHARGE on {} elements, worst ratio {:.6} \
                 (attributable to adapter flush, see ops/subnormal_selfcheck.rs)",
                ag.r_under, ag.worst_under_ratio
            );
        }
    }

    /// `#u1` LOCALIZATION. The bit-compare above says the device's residual
    /// accumulator drifts from the modeled one by at most a couple of ULPs; this
    /// test says WHERE and BY WHICH RULE.
    ///
    /// Mechanism: for `k ≤ 16` the tiled kernel always executes the SAME 16
    /// inner iterations, and every tap past `k` is a zero-padded `0·0` whose
    /// product, both residuals and both accumulator updates are exact zeros. So
    /// dispatching the PRODUCTION kernel at `k = 1, 2, …, 16` over prefixes of
    /// one fixed operand set reads out the device's own `rsum` AFTER EVERY TAP
    /// — a per-tap trace of the production kernel, obtained without editing a
    /// single character of the shipped shader.
    ///
    /// Each observed step is then matched against candidate rounding rules. The
    /// counts tell you exactly which sequence the device executed.
    #[test]
    fn u1_prefix_scan_localizes_the_residual_drift() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        ny_core::eft::eft_self_check().expect("host EFT reference must be sound");

        // 32x32 = four FULLY OCCUPIED 16x16 workgroups: the characterization
        // must hold at production occupancy, not just for a partial tile.
        const M: usize = 32;
        const N: usize = 32;
        /// 4 tiles: the scan crosses three tile boundaries, so a re-association
        /// that only appears once the tile loop is entered is visible too.
        const KMAX: usize = 64;
        let full = TwinCase {
            name: "prefix",
            m: M,
            k: KMAX,
            n: N,
            seed: 0xC0FFEE,
            family: Family::Cancelling,
        };
        let (a_full, w_full) = make_operands(&full);

        // Device trace: r_dev[j][e] = the kernel's rsum after j+1 taps.
        let mut r_dev: Vec<Vec<f32>> = Vec::with_capacity(KMAX);
        let mut v_dev: Vec<Vec<f32>> = Vec::with_capacity(KMAX);
        for k in 1..=KMAX {
            let mut a = Vec::with_capacity(M * k);
            for row in 0..M {
                a.extend_from_slice(&a_full[row * KMAX..row * KMAX + k]);
            }
            let w = w_full[..k * N].to_vec();
            let (v, r) = device
                .u1_run_production_twin(&a, &w, M, k, N)
                .expect("prefix dispatch");
            r_dev.push(r.iter().map(|&b| f32::from_bits(b)).collect());
            v_dev.push(v.iter().map(|&b| f32::from_bits(b)).collect());
        }

        // Host term stream (validated by V matching bit-exactly at every k).
        let mut faithful = 0u64;
        let mut right_assoc = 0u64;
        let mut swapped = 0u64;
        let mut single_rounding = 0u64;
        let mut unexplained = 0u64;
        let mut steps = 0u64;
        let mut v_exact_steps = 0u64;
        let mut first_divergence: Option<(usize, usize)> = None;

        for row in 0..M {
            for col in 0..N {
                let e = row * N + col;
                let mut acc = 0.0f32;
                for j in 0..KMAX {
                    let av = a_full[row * KMAX + j];
                    let wv = w_full[j * N + col];
                    let prod = av * wv;
                    let ep = av.mul_add(wv, -prod).abs();
                    let s = acc + prod;
                    let bb = (-1.0f32).mul_add(acc, s);
                    let sb = (-1.0f32).mul_add(bb, s);
                    let da = (-1.0f32).mul_add(sb, acc);
                    let db = (-1.0f32).mul_add(bb, prod);
                    let es = (da + db).abs();
                    acc = s;

                    if v_dev[j][e].to_bits() == acc.to_bits() {
                        v_exact_steps += 1;
                    }
                    // Prior device state (exact zero before the first tap).
                    let prev = if j == 0 { 0.0f32 } else { r_dev[j - 1][e] };
                    let got = r_dev[j][e];
                    steps += 1;
                    // Candidate rules, in order of specificity.
                    let r_faithful = (prev + ep) + es;
                    let r_right = prev + (ep + es);
                    let r_swap = (prev + es) + ep;
                    // One rounding of the exact 3-term sum (what a
                    // higher-precision accumulator would produce).
                    let r_single = (f64::from(prev) + f64::from(ep) + f64::from(es)) as f32;
                    if got.to_bits() == r_faithful.to_bits() {
                        faithful += 1;
                    } else if got.to_bits() == r_right.to_bits() {
                        right_assoc += 1;
                        first_divergence.get_or_insert((e, j));
                    } else if got.to_bits() == r_swap.to_bits() {
                        swapped += 1;
                        first_divergence.get_or_insert((e, j));
                    } else if got.to_bits() == r_single.to_bits() {
                        single_rounding += 1;
                        first_divergence.get_or_insert((e, j));
                    } else {
                        unexplained += 1;
                        first_divergence.get_or_insert((e, j));
                    }
                }
            }
        }
        println!(
            "\n=== #u1 per-tap prefix scan (production kernel, k=1..{KMAX}, {M}x{N} elements) ===\n\
             steps={steps}  V-exact steps={v_exact_steps}/{steps}\n\
             rule (rsum+ep)+es [SHIPPED]        : {faithful}\n\
             rule rsum+(ep+es) [right-assoc]    : {right_assoc}\n\
             rule (rsum+es)+ep [swapped]        : {swapped}\n\
             rule RN(rsum+ep+es) [one rounding] : {single_rounding}\n\
             UNEXPLAINED by any of the above    : {unexplained}\n\
             first divergence (element, tap)    : {first_divergence:?}",
        );
        assert_eq!(
            v_exact_steps, steps,
            "the value chain must be bit-exact at every prefix length"
        );
        // The three rules above ENUMERATE every binary association of the three
        // addends `{rsum, |ep|, |es|}`, so `unexplained == 0` says exactly this:
        // at every tap the device added the SAME two non-negative residual terms
        // to the running sum, in one of the orders a re-associating compiler may
        // choose. No term is dropped, altered or duplicated. That — not
        // bit-equality — is what makes the γ-based `eft_r_slack` recovery valid.
        assert_eq!(
            unexplained, 0,
            "#u1: {unexplained}/{steps} taps are not explained by ANY association of \
             (rsum, |ep|, |es|) — the device is charging a different term multiset, \
             which invalidates the order-independent recovery the min-combine relies on"
        );
        assert!(
            single_rounding == 0,
            "#u1: {single_rounding} taps match a SINGLE-ROUNDING 3-term sum — the device \
             accumulates at higher precision than the f32 model; re-derive"
        );
    }

    /// The CPU twin itself must be a faithful transcription: pinned scalar
    /// expectations, independent of any GPU.
    #[test]
    fn cpu_twin_transcription_is_pinned() {
        // k=1: R must be exactly the TwoProduct residual of (1+2^-12)^2, i.e.
        // 2^-24, plus a zero TwoSum residual (acc starts at 0).
        let a = [f32::from_bits(0x3F80_0800)];
        let w = [f32::from_bits(0x3F80_0800)];
        let (v, r, _) = cpu_twin(&a, &w, 1, 1, 1, TwinMode::Faithful);
        let (p, e) = ny_core::eft::two_prod_f32(a[0], w[0]);
        assert_eq!(v[0].to_bits(), p.to_bits());
        assert_eq!(r[0].to_bits(), e.abs().to_bits());
        assert_eq!(e.to_bits(), 0x3380_0000, "2^-24");

        // Padded taps are inert: k=1 padded to 16 must equal the unpadded run.
        let (v2, r2, _) = cpu_twin(&a, &w, 1, 1, 1, TwinMode::NoPadTaps);
        assert_eq!(v2[0].to_bits(), v[0].to_bits());
        assert_eq!(r2[0].to_bits(), r[0].to_bits());
    }
}
