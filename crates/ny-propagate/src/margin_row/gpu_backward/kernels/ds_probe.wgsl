// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// M1 parity PROBE kernel for the certified margin-row GPU backward
// (design section 7 M1; concatenated AFTER ds_primitives.wgsl at pipeline
// build). One thread per probe lane; per lane the FIXED composition
//
//     x = (inp.x, inp.y)            // a ds32 value
//     m = ds_mul_f32(x, inp.z)      // DWTimesFP3: eft_two_prod + fma fold + renorm
//     s = ds_add_f32(m, inp.w)      // DWPlusFP:   eft_two_sum + renorm
//     t = ds_add(s, x)              // DWPlusDW:   2x eft_two_sum + fast_two_sum + renorm
//
// exercises EVERY primitive the production kernels use, in composition, with
// all operands loaded from storage at runtime (the compiler cannot
// constant-fold anything away). The host twin (../probe.rs) runs the same
// three steps through ds.rs op-for-op; the M1 gate is BIT-IDENTITY of all
// three output planes.
//
// DISCRIMINATION (what a broken toolchain does to these lanes):
//   * de-fused fma  => eft_two_prod residual identically 0 => m.lo diverges
//     on every lane whose product has a nonzero rounding residual;
//   * reassociated TwoSum => the barrier residual folds to 0 => s.lo / t.lo
//     diverge on the cancellation lanes.
//
// SUBNORMAL DISCIPLINE (same as ny-gpu ops/eft_selfcheck.rs): every operand
// and every intermediate/output value of the probe lanes is NORMAL-RANGE or
// exactly zero — the probe is deliberately BLIND to flush-to-zero, because
// fma on this adapter may DAZ a subnormal OPERAND even under the preserving
// channel (shader_loading.rs module docs), so a subnormal probe lane could
// falsely kill the channel. The subnormal contract is carried by the
// ADMISSION requirement instead (resolved denorm_preserve_enabled ON,
// ../authority.rs) plus the host-charged flush floors of design section 4.1.
// ../tests.rs pins the lane table normal-range so this stays true.

struct ProbeParams {
    // Number of probe lanes.
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> probe_params: ProbeParams;
// Per lane: (x_hi, x_lo, w, b).
@group(0) @binding(1) var<storage, read> probe_in: array<vec4<f32>>;
// The three composition stages, one plane each.
@group(0) @binding(2) var<storage, read_write> probe_out_mul: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> probe_out_addf: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> probe_out_add: array<vec2<f32>>;

// Workgroup size 256: the chunk-256 recipe, same as the production kernels.
@compute @workgroup_size(256)
fn ds_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= probe_params.n {
        return;
    }
    let lane = probe_in[i];
    let x = vec2<f32>(lane.x, lane.y);
    let m = ds_mul_f32(x, lane.z);
    let s = ds_add_f32(m, lane.w);
    let t = ds_add(s, x);
    probe_out_mul[i] = m;
    probe_out_addf[i] = s;
    probe_out_add[i] = t;
}
