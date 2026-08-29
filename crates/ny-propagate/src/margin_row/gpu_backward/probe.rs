// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M1 probe: the adversarial lane table, the HOST twins of both device
//! kernels, and the once-per-process self-check dispatch whose readback feeds
//! [`super::authority::DeviceParityProof::qualify`] (design §6 "admit", §7 M1).
//!
//! Two transactions, both drain-safe (executor contract, R2):
//!
//! 1. **ds probe** (`kernels/ds_probe.wgsl`): per lane the fixed composition
//!    `m = ds_mul_f32(x, w); s = ds_add_f32(m, b); t = ds_add(s, x)` —
//!    every EFT/ds primitive the production kernels use, composed, operands
//!    from storage. Host twin: [`ds`] op-for-op.
//! 2. **gate transform** (`kernels/gate_transform.wgsl`): the REAL production
//!    kernel over a synthetic tableau with both lane-direction gate styles
//!    and both coefficient signs, so the intercept-PAIR selection (review
//!    item 4) is exercised on device. Host twin: [`host_gate_transform`].
//!
//! The streams concatenate (ds: mul/addf/add planes; gate: coeff_out,
//! err_out-as-ds, bias_partial, partial_err-as-ds) into ONE host/device pair,
//! so a single `qualify` call covers everything and mints at most one proof.
//!
//! # Subnormal discipline (deliberate, pinned by `super::tests`)
//!
//! Every probe operand and every host-twin output is NORMAL-RANGE or exactly
//! zero — the same discipline as ny-gpu's `ops/eft_selfcheck.rs`, and for the
//! same reason: fma on this class of adapter may DAZ a subnormal OPERAND even
//! under the preserving channel, so a subnormal lane could kill the channel
//! falsely. The subnormal contract is enforced by the ADMISSION requirement
//! (resolved `denorm_preserve_enabled` ON) plus the charged flush floors of
//! design §4.1, not by this probe.

use ny_core::{LaneBinding, LaneTransactionSpec, MarginRowLaneGpu};

use super::ds::{ds_abs_bound, ds_add, ds_add_f32, ds_mul, ds_mul_f32, Ds, U_DS_F32};
use super::{DS_PRIMITIVES_WGSL, DS_PROBE_WGSL, GATE_TRANSFORM_WGSL};

/// ds-probe lane count. Together with the gate fixture the qualified stream
/// is far above `PARITY_MIN_LANES` (816); this alone already exceeds it.
pub(super) const DS_PROBE_LANES: usize = 1024;

/// Gate-transform fixture shape: `GATE_NEURONS x GATE_ROWS` elements.
pub(super) const GATE_NEURONS: usize = 64;
pub(super) const GATE_ROWS: usize = 16;

/// Deterministic xorshift64 (the repo's probe idiom) so failures reproduce.
struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A full-mantissa f32 in ±[2^-12, 2^12) — normal-range by construction,
    /// so every probe product/sum stays far above the flush band.
    fn coeff(&mut self) -> f32 {
        let r = self.next();
        let frac = (r & 0x7F_FFFF) as u32;
        let exp = 115u32 + ((r >> 24) % 25) as u32; // unbiased [-12, 12]
        let sign = ((r >> 60) & 1) as u32;
        f32::from_bits((sign << 31) | (exp << 23) | frac)
    }
}

/// Split an f64 into a ds32 pair (hi = fl32(v), lo = fl32(v - hi)) — the
/// standard downcast, so probe `x` values carry genuine nonzero `lo` parts.
fn ds_from_f64(v: f64) -> Ds {
    #[allow(clippy::cast_possible_truncation)]
    let hi = v as f32;
    #[allow(clippy::cast_possible_truncation)]
    let lo = (v - f64::from(hi)) as f32;
    Ds { hi, lo }
}

/// The ds-probe lane table: `(x_hi, x_lo, w, b)` per lane. Handpicked
/// discriminating classes (product residual exactly 2^-24, 2^24-boundary
/// sums, near-cancellation) plus deterministic adversarial bulk.
pub(super) fn build_ds_probe_lanes() -> Vec<[f32; 4]> {
    let mut lanes: Vec<[f32; 4]> = Vec::with_capacity(DS_PROBE_LANES);
    let c12 = f32::from_bits(0x3F80_0800); // 1 + 2^-12
                                           // Product residual exactly 2^-24; generic add.
    lanes.push([c12, 0.0, c12, 1e-8]);
    // ds value with a live lo part; near-cancelling add.
    lanes.push([1.0, f32::from_bits(0x3380_0000), 3.0, -3.0]);
    // 2^24 + 1: the add residual is exactly 1.0.
    lanes.push([16_777_216.0, 0.0, 1.0, 1.0]);
    // Irrational-ish product against a cancelling offset.
    lanes.push([std::f32::consts::PI, 0.0, std::f32::consts::E, -8.539_734]);
    // lo of the opposite sign to hi (a legal renormalized ds).
    lanes.push([
        1.0 + f32::from_bits(0x3A00_0000),
        -f32::from_bits(0x2800_0000),
        1.0 / 3.0,
        0.25,
    ]);
    // Near-4 cancellation through the whole chain.
    lanes.push([1.999_999_9, 0.0, 1.999_999_9, -4.0]);

    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    while lanes.len() < DS_PROBE_LANES {
        let x = ds_from_f64(f64::from(rng.coeff()) * (1.0 + 1.234_567e-9));
        let w = rng.coeff();
        let b = match rng.next() % 3 {
            // Generic offset at product scale.
            0 => rng.coeff() * x.hi * w,
            // Near-cancellation of the product (the CROWN regime; exercises
            // the barrier TwoSum residual hard).
            1 => -(x.hi * w) * (1.0 + f32::from_bits(0x3980_0000)), // *(1+2^-12)
            // Magnitude-split add: b far above the product.
            _ => rng.coeff() * 4096.0,
        };
        lanes.push([x.hi, x.lo, w, b]);
    }
    lanes
}

/// Host twin of `kernels/ds_probe.wgsl`: the three output planes,
/// concatenated `[mul lanes.., addf lanes.., add lanes..]` — exactly the
/// readback order of the device transaction.
pub(super) fn host_ds_probe(lanes: &[[f32; 4]]) -> Vec<Ds> {
    let mut mul_plane = Vec::with_capacity(lanes.len());
    let mut addf_plane = Vec::with_capacity(lanes.len());
    let mut add_plane = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let x = Ds {
            hi: lane[0],
            lo: lane[1],
        };
        let m = ds_mul_f32(x, lane[2]);
        let s = ds_add_f32(m, lane[3]);
        let t = ds_add(s, x);
        mul_plane.push(m);
        addf_plane.push(s);
        add_plane.push(t);
    }
    let mut out = mul_plane;
    out.append(&mut addf_plane);
    out.append(&mut add_plane);
    out
}

/// The gate-transform fixture: per-neuron gate ds-PAIRS (two vec4s, hi quad
/// then lo quad) in BOTH lane styles (even j lower `(alpha, s, 0, c)`, odd j
/// upper `(s, alpha, c, 0)` — the host-baked direction of design §5 kernel 2)
/// with LIVE lo parts (the gates ride as ds splits of f64 values), plus a
/// coefficient tableau with both signs and live lo parts, plus a nonnegative
/// error plane.
pub(super) struct GateFixture {
    /// `[hi quad, lo quad]` per neuron, flattened: quads `2j` and `2j + 1`.
    /// Quad layout: `(slope v>=0, slope v<0, icept v>=0, icept v<0)`.
    pub(super) gate: Vec<[f32; 4]>,
    /// ds coefficient per (neuron, row), row-major `j * rows + r`.
    pub(super) coeff: Vec<Ds>,
    /// Nonnegative error per element.
    pub(super) err: Vec<f32>,
}

impl GateFixture {
    /// The selected (slope, icept) ds pair for element `idx` plus the two
    /// Lipschitz bounds (m over the slope pair, mi over the INTERCEPT pair —
    /// mi, not the selected side, is what the E carry multiplies: the stored
    /// sign can differ from the exact one's across the kink), mirroring the
    /// kernel — shared by the host twin and the spec tests.
    pub(super) fn select(&self, idx: usize) -> (Ds, Ds, f32, f32) {
        let j = idx / GATE_ROWS;
        let (ghi, glo) = (self.gate[2 * j], self.gate[2 * j + 1]);
        let nonneg = self.coeff[idx].hi >= 0.0;
        let pick = |hi: &[f32; 4], lo: &[f32; 4], pos: usize, neg: usize| Ds {
            hi: if nonneg { hi[pos] } else { hi[neg] },
            lo: if nonneg { lo[pos] } else { lo[neg] },
        };
        let slope = pick(&ghi, &glo, 0, 1);
        let icept = pick(&ghi, &glo, 2, 3);
        let pair_bound = |a: usize, b: usize| {
            ds_abs_bound(Ds {
                hi: ghi[a],
                lo: glo[a],
            })
            .max(ds_abs_bound(Ds {
                hi: ghi[b],
                lo: glo[b],
            }))
        };
        (slope, icept, pair_bound(0, 1), pair_bound(2, 3))
    }
}

pub(super) fn build_gate_fixture() -> GateFixture {
    let mut rng = XorShift64(0x2545_F491_4F6C_DD1D);
    // f64 gate parameters with genuinely fractional (non-f32-exact) values,
    // so the ds splits carry live lo parts — the production shape.
    let unit = |rng: &mut XorShift64| -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (rng.next() % 1_000_000) as f64 / 1_000_000.0;
        v * (1.0 + 3.141_59e-11)
    };
    let mut gate = Vec::with_capacity(2 * GATE_NEURONS);
    for j in 0..GATE_NEURONS {
        let alpha = ds_from_f64(unit(&mut rng));
        let s = ds_from_f64(unit(&mut rng));
        let c = ds_from_f64(unit(&mut rng) + 1e-3); // strictly positive mass
        let zero = Ds { hi: 0.0, lo: 0.0 };
        let quad: [Ds; 4] = if j % 2 == 0 {
            [alpha, s, zero, c] // LOWER lane style
        } else {
            [s, alpha, c, zero] // UPPER lane style
        };
        gate.push([quad[0].hi, quad[1].hi, quad[2].hi, quad[3].hi]);
        gate.push([quad[0].lo, quad[1].lo, quad[2].lo, quad[3].lo]);
    }
    let total = GATE_NEURONS * GATE_ROWS;
    let mut coeff = Vec::with_capacity(total);
    let mut err = Vec::with_capacity(total);
    for _ in 0..total {
        let v = ds_from_f64(f64::from(rng.coeff()) * (1.0 + 7.654_321e-10));
        coeff.push(v);
        err.push(v.hi.abs() * 1e-6);
    }
    GateFixture { gate, coeff, err }
}

/// Host twin of `kernels/gate_transform.wgsl`, op-for-op: returns
/// `(coeff_out, err_out, bias_partial, partial_err)` planes in element order.
/// The err planes mirror the kernel's f32 sequence exactly (`mul_add` is the
/// fused `fma`, which the M1 gate has already proven fused on this adapter).
pub(super) type GatePlanes = (Vec<Ds>, Vec<f32>, Vec<Ds>, Vec<f32>);

pub(super) fn host_gate_transform(fixture: &GateFixture) -> GatePlanes {
    let total = GATE_NEURONS * GATE_ROWS;
    let mut coeff_out = Vec::with_capacity(total);
    let mut err_out = Vec::with_capacity(total);
    let mut bias_partial = Vec::with_capacity(total);
    let mut partial_err = Vec::with_capacity(total);
    for idx in 0..total {
        let (slope, icept, m, mi) = fixture.select(idx);
        let v = fixture.coeff[idx];
        let o = ds_mul(v, slope);
        coeff_out.push(o);
        err_out.push(fixture.err[idx].mul_add(m, 2.0 * U_DS_F32 * ds_abs_bound(o)));
        let p = ds_mul(v, icept);
        bias_partial.push(p);
        partial_err.push(fixture.err[idx].mul_add(mi, 2.0 * U_DS_F32 * ds_abs_bound(p)));
    }
    (coeff_out, err_out, bias_partial, partial_err)
}

/// Assembled host/device parity streams (plus telemetry facts).
pub(super) struct SelfCheckStreams {
    pub(super) host: Vec<Ds>,
    pub(super) device: Vec<(f32, f32)>,
    pub(super) device_ms: f64,
}

fn u32_uniform(words: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Reinterpret a readback byte buffer as `(hi, lo)` pairs. Pure bit
/// reinterpretation (`f32::from_le_bytes` is a transmute of the stored word)
/// — no float op that could canonicalize a bit pattern; the M1 compare is on
/// exactly what the shader stored.
pub(super) fn pairs_from_bytes(bytes: &[u8]) -> Result<Vec<(f32, f32)>, &'static str> {
    if !bytes.len().is_multiple_of(8) {
        return Err("lane readback is not a whole number of (hi, lo) pairs");
    }
    let floats = scalars_from_bytes(bytes);
    Ok(floats
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0], c[1]))
        .collect())
}

pub(super) fn scalars_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch both probe transactions on the lane device and assemble the
/// concatenated parity streams. `Err(reason)` is terminal for the process
/// (the caller latches the channel dead): a probe that cannot RUN cannot
/// qualify anything, and retrying a broken device is the flapping this lane
/// forbids.
pub(super) fn self_check_streams(
    gpu: &dyn MarginRowLaneGpu,
) -> Result<SelfCheckStreams, &'static str> {
    let started = std::time::Instant::now();

    // ---- transaction 1: ds primitives probe ----
    let lanes = build_ds_probe_lanes();
    let lane_bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|l| bytemuck::bytes_of(l).to_vec())
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let n = lanes.len() as u32;
    let ds_shader = format!("{DS_PRIMITIVES_WGSL}\n{DS_PROBE_WGSL}");
    let plane_bytes = (lanes.len() * 8) as u64;
    let ds_uniform = u32_uniform([n, 0, 0, 0]);
    let ds_spec = LaneTransactionSpec {
        label: "margin_row_gpu_backward_ds_probe",
        entry: "ds_probe",
        shader_wgsl: &ds_shader,
        uniforms: &ds_uniform,
        bindings: &[
            LaneBinding::Read(&lane_bytes),
            LaneBinding::Output(plane_bytes),
            LaneBinding::Output(plane_bytes),
            LaneBinding::Output(plane_bytes),
        ],
        workgroups: [n.div_ceil(256), 1, 1],
    };
    let ds_out = gpu
        .run_lane_transaction(&ds_spec)
        .map_err(|_| "ds probe transaction failed on the lane device")?;
    if ds_out.len() != 3 {
        return Err("ds probe returned the wrong number of output planes");
    }

    // ---- transaction 2: the production gate-transform kernel ----
    let fixture = build_gate_fixture();
    let total = GATE_NEURONS * GATE_ROWS;
    let gate_bytes: Vec<u8> = fixture
        .gate
        .iter()
        .flat_map(|g| bytemuck::bytes_of(g).to_vec())
        .collect();
    let coeff_bytes: Vec<u8> = fixture
        .coeff
        .iter()
        .flat_map(|d| {
            let pair = [d.hi, d.lo];
            bytemuck::bytes_of(&pair).to_vec()
        })
        .collect();
    let err_bytes: Vec<u8> = fixture.err.iter().flat_map(|e| e.to_le_bytes()).collect();
    let gate_shader = format!("{DS_PRIMITIVES_WGSL}\n{GATE_TRANSFORM_WGSL}");
    #[allow(clippy::cast_possible_truncation)]
    let gate_uniform = u32_uniform([GATE_NEURONS as u32, GATE_ROWS as u32, 0, 0]);
    #[allow(clippy::cast_possible_truncation)]
    let total_u32 = total as u32;
    let gate_spec = LaneTransactionSpec {
        label: "margin_row_gpu_backward_gate_transform",
        entry: "gate_transform",
        shader_wgsl: &gate_shader,
        uniforms: &gate_uniform,
        bindings: &[
            LaneBinding::Read(&gate_bytes),
            LaneBinding::Read(&coeff_bytes),
            LaneBinding::Output((total * 8) as u64),
            LaneBinding::Read(&err_bytes),
            LaneBinding::Output((total * 4) as u64),
            LaneBinding::Output((total * 8) as u64),
            LaneBinding::Output((total * 4) as u64),
        ],
        workgroups: [total_u32.div_ceil(256), 1, 1],
    };
    let gate_out = gpu
        .run_lane_transaction(&gate_spec)
        .map_err(|_| "gate-transform probe transaction failed on the lane device")?;
    if gate_out.len() != 4 {
        return Err("gate probe returned the wrong number of output planes");
    }
    let device_ms = started.elapsed().as_secs_f64() * 1e3;

    // ---- assemble the concatenated parity streams ----
    let mut host = host_ds_probe(&lanes);
    let (coeff_out, err_out, bias_partial, partial_err) = host_gate_transform(&fixture);
    host.extend(coeff_out);
    host.extend(err_out.iter().map(|&e| Ds { hi: e, lo: 0.0 }));
    host.extend(bias_partial);
    host.extend(partial_err.iter().map(|&e| Ds { hi: e, lo: 0.0 }));

    let mut device: Vec<(f32, f32)> = Vec::with_capacity(host.len());
    for plane in &ds_out {
        device.extend(pairs_from_bytes(plane)?);
    }
    device.extend(pairs_from_bytes(&gate_out[0])?);
    device.extend(
        scalars_from_bytes(&gate_out[1])
            .into_iter()
            .map(|e| (e, 0.0)),
    );
    device.extend(pairs_from_bytes(&gate_out[2])?);
    device.extend(
        scalars_from_bytes(&gate_out[3])
            .into_iter()
            .map(|e| (e, 0.0)),
    );

    Ok(SelfCheckStreams {
        host,
        device,
        device_ms,
    })
}
