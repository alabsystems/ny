// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Residual merge for the certified margin-row GPU backward: the elementwise
// ds addition of two coefficient contributions at a residual `Add`'s common
// ancestor tensor, with the matching E-lane merge. Concatenated AFTER
// ds_primitives.wgsl at pipeline build.
//
// CPU reference: engine.rs merge_into —
//   L = L_a + L_b;  E = slack16(E_a + E_b + 2u * |L|)
// with the f64 unit swapped for DS_UNIT (the value add is the accurate
// DWPlusDW, <= 3u_ds^2 relative, charged at 2*DS_UNIT*|L| for headroom
// exactly like the CPU's double-weight carry). The f32 roundings of the
// E-side expression are host-widened at egress.

struct MergeParams {
    // Total elements (n * rows).
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> mp: MergeParams;
@group(0) @binding(1) var<storage, read> merge_a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> merge_b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> merge_out: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> merge_ea: array<f32>;
@group(0) @binding(5) var<storage, read> merge_eb: array<f32>;
@group(0) @binding(6) var<storage, read_write> merge_eout: array<f32>;

// Workgroup size 256 (the chunk-256 recipe); one thread per element.
@compute @workgroup_size(256)
fn merge_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= mp.total {
        return;
    }
    let s = ds_add(merge_a[idx], merge_b[idx]);
    merge_out[idx] = s;
    merge_eout[idx] = merge_ea[idx] + merge_eb[idx] + 2.0 * DS_UNIT * ds_abs_bound(s);
}
