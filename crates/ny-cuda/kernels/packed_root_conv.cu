// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Ordered paired root contraction. Capability requires actual-module known-answer
// tests before publication. The typed host adapter must validate finite inputs,
// all gather/inverse indices, nonaliasing buffers, checked allocation products,
// and complete disjoint owner coverage before publishing either output lane.
// Build with --fmad=false --ftz=false; never --use_fast_math.
// Existing CPU conv_error, support, FoldMag and physical bias/fold remain outside.

using u32 = unsigned int;
using u64 = unsigned long long;
static_assert(sizeof(u32) == 4 && sizeof(u64) == 8, "packed-root ABI widths");
constexpr u32 MISSING = 0xffffffffu;

extern "C" __global__ void packed_root_conv_ordered(
    const float* __restrict__ source_m,
    const float* __restrict__ source_d,
    const float* __restrict__ weights,
    const u32* __restrict__ gather,
    const u32* __restrict__ inverse,
    float* __restrict__ output_m,
    float* __restrict__ output_d,
    u32 source_stride,
    u32 output_stride,
    u32 output_spatial,
    u32 output_channels,
    u32 spatial_taps,
    u32 taps,
    u64 first_owner,
    u64 owner_count)
{
    const u64 local = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= owner_count) return;
    const u64 owner = first_owner + local;
    const u64 spatial_width = (u64)output_channels * output_stride;
    // Schedule all output channels at one spatial site together, so their
    // source patch and inverse-map rows can share the device cache.
    const u32 sp = (u32)(owner / spatial_width);
    const u64 within_spatial = owner % spatial_width;
    const u32 oc = (u32)(within_spatial / output_stride);
    const u32 column = (u32)(within_spatial % output_stride);
    const u32 output_slots = output_stride - 1;
    float m = 0.0f;
    float d = 0.0f;
    for (u32 t = 0; t < taps; ++t) {
        const u32 row = gather[(u64)t * output_spatial + sp];
        if (row == MISSING) continue;
        const float w = weights[(u64)oc * taps + t];
        // Preserve CPU skipping of both signed zero weights.
        if (w == 0.0f) continue;
        const u32 slot = column == output_slots ? source_stride - 1 :
            inverse[((u64)sp * spatial_taps + t % spatial_taps)
                    * output_slots + column];
        // Missing original coordinates contribute no arithmetic operation.
        if (slot == MISSING) continue;
        const u64 source = (u64)row * source_stride + slot;
        const float aw = __int_as_float(__float_as_int(w) & 0x7fffffff);
        // One thread owns each coefficient. No atomics, inter-thread reduction,
        // reassociation or FMA; the original ascending tap order is preserved.
        m = __fadd_rn(m, __fmul_rn(w, source_m[source]));
        d = __fadd_rn(d, __fmul_rn(aw, source_d[source]));
    }
    // CPU PackedRows uses channel-major row order. Scheduling order is private.
    const u64 destination = ((u64)oc * output_spatial + sp)
                                 * output_stride + column;
    output_m[destination] = m;
    output_d[destination] = d;
}
