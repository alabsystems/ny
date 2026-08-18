// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host-side contract tests for WGPU pipeline/shader invariants.
//!
//! These tests run in the default test suite (no `gpu-tests` feature required).
//! They verify structural invariants that prevent silent runtime failures:
//! - Param struct sizes match WGSL uniform expectations (16 bytes each)
//! - Shader binding declarations match pipeline bind group layouts
//! - All shaders define a `fn main` entry point
//! - WGSL source compiles on the host adapter
//! - NaN safety constants are consistent across shaders
//!
//! Part of #1908.

use super::ops::StridedGatherParams;
use super::params::{
    AddIbpParams, Conv2dIbpParams, ConvCol2imParams, ConvReshapeParams, GemmParams,
    LinearIbpParams, MatmulIbpParams, ReluIbpParams, ScaleIbpParams, SoftmaxIbpParams,
    TransposeIbpParams,
};
use super::shaders::{
    ADD_IBP_SHADER, CONV2D_IBP_IM2COL_SHADER, CONV_COL2IM_SHADER, CONV_COL2IM_TAINT_SHADER,
    CONV_RESHAPE_SHADER, CONV_RESHAPE_TAINT_SHADER, CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER,
    CROWN_ACTIVATION_RESIDENT_SHADER, CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
    CROWN_BIAS_ERR_ACCUMULATE_SHADER, CROWN_CONCRETIZE_SHADER, CROWN_CONCRETIZE_SOUND_SHADER,
    CROWN_STRIDED_GATHER_SHADER, GEMM_F32_EFT_TWIN_SHADER, GEMM_F32_SHADER,
    GEMM_F32_SMALL_K_SHADER, GEMM_F32_SMALL_K_TAINT_SHADER, LINEAR_IBP_SHADER, MATMUL_IBP_SHADER,
    RELU_IBP_SHADER, SCALE_IBP_SHADER, SOFTMAX_APPLY_SHADER, SOFTMAX_REDUCE_SHADER,
    TRANSPOSE_IBP_SHADER,
};
use std::mem::size_of;

/// TwoProdFMA's residual is guaranteed exact only once the rounded product is at
/// least `2^-101`. `F32_MIN_NORMAL` (`2^-126`) remains the sound charge for a
/// guarded tap, but is too small to be the guard threshold. Scan the Rust source,
/// not selected exported constants, so a new WGSL block or an alias such as
/// `F32_MIN_NORMAL_ACT` cannot silently reintroduce the old comparison. Arithmetic
/// FTZ charges and integer-classification uses of `F32_MIN_NORMAL*` remain allowed.
#[test]
fn test_wgsl_two_prod_guards_use_exactness_floor_not_ftz_charge() {
    // BOTH WGSL source modules. `shaders_taint.rs` was split out with the `#u4`
    // taint twins and was invisible to this scan, which defeats the stated
    // purpose above — "scan the Rust source, not selected exported constants, so
    // a NEW WGSL BLOCK ... cannot silently reintroduce the old comparison." A new
    // FILE is exactly such a block. Any further split must be added here.
    let source = concat!(include_str!("shaders.rs"), include_str!("shaders_taint.rs"));
    let compact: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();

    assert!(
        !compact.contains("<F32_MIN_NORMAL"),
        "WGSL must never use F32_MIN_NORMAL* as a TwoProd/FMA residual guard; \
         compare with TWO_PROD_EXACT_FLOOR_F32 and keep F32_MIN_NORMAL* only as \
         the charged radius"
    );

    // GEMM twin, scalar bias, activation base+twin, activation intercept, and
    // four concretize products. The U4 activation twin added the ninth site;
    // counting it separately prevents its arithmetic copy from drifting. The
    // twin carries the base shader's audited guard and charge verbatim.
    const EXPECTED_RESIDUAL_GUARDS: usize = 9;
    assert_eq!(
        compact.matches("<TWO_PROD_EXACT_FLOOR_F32").count(),
        EXPECTED_RESIDUAL_GUARDS,
        "review every added or removed WGSL TwoProd residual site and keep the \
         source-level guard inventory explicit"
    );
}

/// SENTINEL STICKINESS across the EFT arm (#gpu-typed-authority).
///
/// `CROWN_AW_ERROR_COMBINE_SHADER` deliberately degrades the certified error to
/// `1e30` when EITHER `s_prod = fl(|A|@|W|)` or `prop = fl(err@|W|)` saturates
/// at `FALLBACK_BOUND` — past saturation the true reduction is unknown and
/// strictly larger, so a measured charge would UNDER-cover.
/// `CROWN_EFT_MIN_COMBINE_SHADER` then runs `min(err_out, e_eft)` on the SAME
/// buffer, so it must observe BOTH saturation signals or it can silently lower
/// a deliberately-degraded sentinel back to a measured value.
///
/// This test pins that the EFT arm binds `s_prod` and refuses on it. Removing
/// either the binding or the guard would restore a path on which a certified
/// radius shrinks past the overflow sentinel — the exact class of defect the
/// WGPU verdict quarantine exists to prevent.
#[test]
fn test_eft_min_combine_refuses_on_both_saturation_sentinels() {
    use super::shaders::{CROWN_AW_ERROR_COMBINE_SHADER, CROWN_EFT_MIN_COMBINE_SHADER};

    let compact: String = CROWN_EFT_MIN_COMBINE_SHADER
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();

    assert!(
        compact.contains("s_prod:array<f32>"),
        "the EFT min-combine must BIND s_prod; without it the arm cannot see the \
         |A|@|W| saturation that the Higham combine degrades on"
    );
    assert!(
        compact.contains("if(s_prod[i]>=FALLBACK_BOUND){return;}"),
        "the EFT min-combine must REFUSE (keep the degraded Higham charge) when \
         s_prod saturates — a min() past the sentinel is a tightening on an \
         unproven path"
    );
    assert!(
        compact.contains("if(pr>=FALLBACK_BOUND){return;}"),
        "the EFT min-combine must also refuse on the propagated-term sentinel"
    );

    // The two shaders must degrade/refuse on the SAME set of signals. If the
    // Higham arm ever learns a third sentinel, this count forces the EFT arm to
    // be revisited rather than silently falling behind.
    let higham: String = CROWN_AW_ERROR_COMBINE_SHADER
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    assert!(
        higham.contains("if(s_prod[i]>=FALLBACK_BOUND||prop[i]>=FALLBACK_BOUND){e=1e30;}"),
        "the Higham combine's sentinel degrade changed shape; re-derive the EFT \
         arm's matching refusal before updating this pin"
    );
}

/// The EFT shaders are self-contained WGSL sources rather than sound-IBP bodies
/// that inherit `IBP_SOUND_PRELUDE`. Keep a local declaration beside every use;
/// otherwise the first live EFT pipeline creation fails after Rust compilation.
#[cfg(feature = "wgpu")]
#[test]
fn test_self_contained_eft_shaders_define_and_validate_two_prod_floor() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    const EXACT_FLOOR_DECL: &str = "constTWO_PROD_EXACT_FLOOR_F32:f32=3.9443045e-31;";
    assert_eq!(
        3.9443045e-31f32.to_bits(),
        ny_core::eft::TWO_PROD_EXACT_FLOOR_F32.to_bits(),
        "the WGSL decimal literal must round to the derived Rust constant"
    );

    let shaders = [
        ("gemm_eft_twin", GEMM_F32_EFT_TWIN_SHADER),
        ("bias_eft", CROWN_BIAS_ERR_ACCUMULATE_SHADER),
        ("activation_eft", CROWN_ACTIVATION_RESIDENT_SHADER),
        (
            "activation_taint_eft",
            CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
        ),
        (
            "activation_intercept_eft",
            CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER,
        ),
        ("concretize_eft", CROWN_CONCRETIZE_SOUND_SHADER),
    ];

    for (name, source) in shaders {
        let compact: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();
        assert!(
            compact.contains(EXACT_FLOOR_DECL),
            "{name} uses a TwoProd residual guard and must define the exact 2^-101 floor"
        );
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name}: WGSL parse failed: {error:?}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: WGSL validation failed: {error:?}"));
    }
}

/// Bias/error radii are accumulated across layers. A normal-sized existing
/// radius can absorb a tiny positive local/floor term under ordinary RN, which
/// would publish less than the sum of two individually sound bounds. Pin both
/// scalar writers to staged outward assembly and retain the concrete
/// `1.0 + tiny` counterexample that motivated it.
#[test]
fn test_bias_error_writers_assemble_every_nonnegative_term_outward() {
    const OUTWARD_FLUSH: &str =
        "letflush_scaled=round_up_pos(round_up_pos(sf[0]*p.slack)*F32_MIN_NORMAL);";
    const OUTWARD_UPDATE: &str =
        "bias_err_out[s]=round_up_pos(round_up_pos(old_err+local_err)+flush);";
    const LEGACY_REDUCTION_RECOVERY: &str =
        "letreduced_err=round_up_pos(p.gamma_k*sa[0]+se[0]);letlocal_err=round_up_pos(reduced_err*p.slack);";
    const EFT_PROPAGATED_RECOVERY: &str = "letpropagated_err=round_up_pos(se[0]*p.slack);";

    for (name, source) in [
        ("bias", CROWN_BIAS_ERR_ACCUMULATE_SHADER),
        (
            "activation intercept",
            CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER,
        ),
    ] {
        let compact: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();
        assert!(
            compact.contains(OUTWARD_FLUSH),
            "{name} writer must assemble its additive and amplified flush terms outward"
        );
        assert_eq!(
            compact.matches(OUTWARD_UPDATE).count(),
            2,
            "{name} writer must use staged outward accumulation in both legacy and EFT arms"
        );
        assert!(
            compact.contains(LEGACY_REDUCTION_RECOVERY),
            "{name} legacy arm must recover the non-negative sa/se reductions with p.slack"
        );
        assert!(
            compact.contains(EFT_PROPAGATED_RECOVERY),
            "{name} EFT arm must recover its propagated-error reduction independently of r_slack"
        );
    }

    let old = 1.0f32;
    let tiny = f32::MIN_POSITIVE;
    assert_eq!(old + tiny, old, "the ordinary-RN swallowing repro drifted");
    let round_up_pos = |value: f32| {
        if value <= 0.0 || !value.is_finite() {
            value
        } else if value < f32::MIN_POSITIVE {
            f32::MIN_POSITIVE
        } else {
            f32::from_bits(value.to_bits() + 1)
        }
    };
    let published = round_up_pos(round_up_pos(old + 0.0) + tiny);
    assert!(
        published > old,
        "staged outward assembly must preserve a positive floor beside an O(1) radius"
    );

    // Mirror the two kernels' shared 256-lane strided reduction on the
    // deterministic non-negative row [2^24, 1, 1, ...]. At k=1024, thread 0
    // loses three unit terms before the tree. A single final next-up therefore
    // remains below the exact f64 sum; the k-scaled combine recovery covers it.
    const REDUCTION_K: usize = 1024;
    const WORKGROUP_SIZE: usize = 256;
    let large = 16_777_216.0f32; // 2^24
    let mut lanes = [0.0f32; WORKGROUP_SIZE];
    for (lane_id, lane) in lanes.iter_mut().enumerate() {
        let mut j = lane_id;
        while j < REDUCTION_K {
            *lane += if j == 0 { large } else { 1.0 };
            j += WORKGROUP_SIZE;
        }
    }
    let mut stride = WORKGROUP_SIZE / 2;
    while stride > 0 {
        let (left, right) = lanes.split_at_mut(stride);
        for (lhs, rhs) in left.iter_mut().zip(right.iter()) {
            *lhs += *rhs;
        }
        stride >>= 1;
    }
    let reduced = lanes[0];
    let exact = f64::from(large) + (REDUCTION_K - 1) as f64;
    let one_final_ulp = round_up_pos(reduced);
    assert!(
        f64::from(one_final_ulp) < exact,
        "the reduction counterexample must remain under-covered by one final ULP"
    );
    let slack = super::sound_consts::combine_slack_f32(REDUCTION_K)
        .expect("the contract reduction length must admit finite recovery");
    let recovered = round_up_pos(reduced * slack);
    assert!(
        f64::from(recovered) >= exact,
        "the shared p.slack recovery must cover the exact propagated-error sum"
    );
}

/// A rung-3 residual can flush before the EFT residual lane is multiplied by
/// its recovery factor. Every reduction consumer must therefore construct its
/// host additive with that same factor; the activation coefficient update is
/// the sole elementwise consumer and intentionally keeps the unscaled base.
#[test]
fn test_rung3_flush_floor_is_scaled_at_every_reduction_consumer() {
    let resident: String = include_str!("ops/crown_backward_sound_resident.rs")
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    let concretize: String = include_str!("ops/crown_concretize_sound.rs")
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();

    assert_eq!(
        resident
            .matches("rung3_flush_safe_additive_scaled(")
            .count(),
        4,
        "review every added/removed resident reduction floor; each EFT residual consumer needs its recovery multiplier"
    );
    assert_eq!(
        concretize
            .matches("rung3_flush_safe_additive_scaled(")
            .count(),
        1,
        "sound concretize must have exactly one scaled rung-3 floor call"
    );

    for required in [
        "rung3_flush_safe_additive_scaled(nn_u32,eft_slack,)",
        "rung3_flush_safe_additive_scaled(conv_reduction_u32,conv_eft_slack,)",
        "rung3_flush_safe_additive_scaled(bias_reduction_u32,bias_eft_slack,)",
        "rung3_flush_safe_additive_scaled(reduction_u32,eft_slack,)",
    ] {
        assert!(
            resident.contains(required),
            "resident reduction floor lost its matching residual recovery: {required}"
        );
    }
    assert!(
        concretize.contains("rung3_flush_safe_additive_scaled(shape.input_dim_u32,eft_r_slack,)"),
        "concretize floor must be scaled by the exact eft_r_slack uniform"
    );

    const ELEMENTWISE_BASE: &str =
        "letadd_e=super::super::sound_consts::rung3_flush_safe_additive(1)?;";
    assert_eq!(
        resident.matches(ELEMENTWISE_BASE).count(),
        1,
        "activation's elementwise coefficient update is the one intentional unscaled rung-3 site"
    );
}

/// GB10/Vulkan with DenormPreserve preserves core multiplication but
/// DAZ-zeroes subnormal multiplicands in `fma(a,b,0)`, even when amplification
/// should make the result normal. Primary EFT products must therefore use the
/// rung-3-qualified core multiply; FMA is reserved for residual calculations,
/// where operand DAZ either over-charges `|ep|` or falls under the explicit
/// small-product/barrier floor.
#[test]
fn test_eft_primary_products_avoid_the_measured_fma_operand_daz_form() {
    let source = include_str!("shaders.rs");
    let compact: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();

    for required in [
        "letprod=a*w;",
        "letprod=aj*bj;",
        "letprod=a_v*sel;",
        "varp=a_l_pos*x_l;",
        "p=a_l_neg*x_u;",
        "p=a_u_pos*x_u;",
        "p=a_u_neg*x_l;",
    ] {
        assert!(
            compact.contains(required),
            "reviewed core-multiply EFT product `{required}` disappeared"
        );
    }
    for (name, activation) in [
        ("activation", CROWN_ACTIVATION_RESIDENT_SHADER),
        ("activation taint", CROWN_ACTIVATION_RESIDENT_TAINT_SHADER),
    ] {
        let activation: String = activation
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect();
        assert_eq!(
            activation.matches("letbase=a*sel;").count(),
            1,
            "{name} value path must keep its primary product on core multiply"
        );
        assert!(
            !activation.contains("letbase=fma(a,sel,0.0);"),
            "{name} value path regressed to the measured FMA-operand-DAZ form"
        );
    }
    for forbidden in [
        "letprod=fma(a,w,0.0);",
        "letprod=fma(aj,bj,0.0);",
        "letprod=fma(a_v,sel,0.0);",
        "letbase=fma(a,sel,0.0);",
        "varp=fma(a_l_pos,x_l,0.0);",
        "p=fma(a_l_neg,x_u,0.0);",
        "p=fma(a_u_pos,x_u,0.0);",
        "p=fma(a_u_neg,x_l,0.0);",
    ] {
        assert!(
            !compact.contains(forbidden),
            "primary product regressed to the measured FMA-operand-DAZ form: {forbidden}"
        );
    }
}

/// Once the primary product is computed by qualified core multiplication, a
/// residual FMA that DAZ-zeroes the subnormal multiplicand evaluates to
/// `-prod`. Above the exactness threshold `abs(-prod)` trivially dominates the
/// true RN product residual; below it the shader replaces the observation with
/// `F32_MIN_NORMAL`. Exercise both subnormal endpoints, signs, exponent range,
/// and non-power-of-two significands against the exact f64 product.
#[test]
fn test_measured_fma_operand_daz_makes_product_residual_conservative() {
    const SUBNORMAL_BITS: [u32; 5] = [1, 2, 0x003f_ffff, 0x0040_0001, 0x007f_ffff];
    for subnormal_bits in SUBNORMAL_BITS {
        for subnormal_sign in [0u32, 0x8000_0000] {
            let subnormal = f32::from_bits(subnormal_bits | subnormal_sign);
            for exponent in 1u32..=254 {
                for fraction in [0u32, 0x0040_0001, 0x007f_ffff] {
                    for normal_sign in [0u32, 0x8000_0000] {
                        let normal = f32::from_bits((exponent << 23) | fraction | normal_sign);
                        for (a, b) in [(subnormal, normal), (normal, subnormal)] {
                            let prod = a * b;
                            if !prod.is_finite() {
                                continue;
                            }
                            let exact = f64::from(a) * f64::from(b);
                            let true_residual = (exact - f64::from(prod)).abs();
                            let published = if prod.abs() < ny_core::eft::TWO_PROD_EXACT_FLOOR_F32 {
                                f32::MIN_POSITIVE
                            } else {
                                // Modeled FMA operand DAZ in either order:
                                // fma(a,b,-prod) = -prod.
                                prod.abs()
                            };
                            assert!(
                                f64::from(published) >= true_residual,
                                "a=0x{:08x} b=0x{:08x}: DAZ residual charge {published:e} \
                                 < exact RN product residual {true_residual:e}",
                                a.to_bits(),
                                b.to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Param struct size invariants
// ============================================================================
// Every params struct is passed as a uniform buffer to WGSL shaders.
// WGSL struct Params has 4 u32 fields = 16 bytes. If the Rust struct
// drifts from this, the GPU will read garbage.

#[test]
fn test_linear_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<LinearIbpParams>(),
        16,
        "LinearIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_matmul_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<MatmulIbpParams>(),
        16,
        "MatmulIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_gemm_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<GemmParams>(),
        16,
        "GemmParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_softmax_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<SoftmaxIbpParams>(),
        16,
        "SoftmaxIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_transpose_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<TransposeIbpParams>(),
        16,
        "TransposeIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_scale_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ScaleIbpParams>(),
        16,
        "ScaleIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_conv_reshape_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ConvReshapeParams>(),
        16,
        "ConvReshapeParams must be exactly 16 bytes to match WGSL Params struct (4 x u32)"
    );
}

#[test]
fn test_conv2d_ibp_params_size_is_64_bytes() {
    assert_eq!(
        size_of::<Conv2dIbpParams>(),
        64,
        "Conv2dIbpParams must be exactly 64 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_conv_col2im_params_size_is_64_bytes() {
    assert_eq!(
        size_of::<ConvCol2imParams>(),
        64,
        "ConvCol2imParams must be exactly 64 bytes to match WGSL Params struct (14 x u32 + vec2)"
    );
}

#[test]
fn test_relu_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ReluIbpParams>(),
        16,
        "ReluIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_add_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<AddIbpParams>(),
        16,
        "AddIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_strided_gather_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<StridedGatherParams>(),
        16,
        "StridedGatherParams must match the four-u32 WGSL Params layout"
    );
}

// ============================================================================
// Shader entry point invariant
// ============================================================================
// Every compute pipeline specifies `entry_point: Some("main")`.
// If a shader doesn't define `fn main`, pipeline creation will fail at runtime.

#[test]
fn test_all_shaders_define_main_entry_point() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("GEMM_F32_SMALL_K_SHADER", GEMM_F32_SMALL_K_SHADER),
        (
            "GEMM_F32_SMALL_K_TAINT_SHADER",
            GEMM_F32_SMALL_K_TAINT_SHADER,
        ),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_RESHAPE_TAINT_SHADER", CONV_RESHAPE_TAINT_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("CONV_COL2IM_TAINT_SHADER", CONV_COL2IM_TAINT_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
        ("CROWN_STRIDED_GATHER_SHADER", CROWN_STRIDED_GATHER_SHADER),
    ];
    for (name, source) in shaders {
        assert!(
            source.contains("fn main("),
            "{name} must define `fn main(` entry point"
        );
    }
}

// ============================================================================
// Shader binding count invariants
// ============================================================================
// Each pipeline's bind group layout must match the number of @binding(N)
// declarations in the WGSL shader. Drift between these causes runtime panics.

/// Count the number of `@binding(` declarations in a WGSL shader source.
fn count_bindings(shader_source: &str) -> usize {
    shader_source.matches("@binding(").count()
}

/// Extract the maximum binding index from a WGSL shader source.
fn max_binding_index(shader_source: &str) -> Option<u32> {
    let mut max_idx = None;
    for (i, _) in shader_source.match_indices("@binding(") {
        let after = &shader_source[i + "@binding(".len()..];
        if let Some(end) = after.find(')') {
            if let Ok(idx) = after[..end].trim().parse::<u32>() {
                max_idx = Some(max_idx.map_or(idx, |m: u32| m.max(idx)));
            }
        }
    }
    max_idx
}

#[test]
fn test_linear_ibp_shader_has_8_bindings() {
    // pipelines.rs: create_linear_ibp_pipeline has 8 entries (0..7)
    assert_eq!(count_bindings(LINEAR_IBP_SHADER), 8);
    assert_eq!(max_binding_index(LINEAR_IBP_SHADER), Some(7));
}

#[test]
fn test_matmul_ibp_shader_has_7_bindings() {
    // pipelines.rs: create_matmul_ibp_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(MATMUL_IBP_SHADER), 7);
    assert_eq!(max_binding_index(MATMUL_IBP_SHADER), Some(6));
}

#[test]
fn test_softmax_reduce_shader_has_8_bindings() {
    // pipelines.rs: create_softmax_reduce_pipeline has 8 entries (0..7)
    assert_eq!(count_bindings(SOFTMAX_REDUCE_SHADER), 8);
    assert_eq!(max_binding_index(SOFTMAX_REDUCE_SHADER), Some(7));
}

#[test]
fn test_softmax_apply_shader_has_7_bindings() {
    // pipelines.rs: create_softmax_apply_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(SOFTMAX_APPLY_SHADER), 7);
    assert_eq!(max_binding_index(SOFTMAX_APPLY_SHADER), Some(6));
}

#[test]
fn test_transpose_ibp_shader_has_5_bindings() {
    // pipelines.rs: create_transpose_ibp_pipeline has 5 entries (0..4)
    assert_eq!(count_bindings(TRANSPOSE_IBP_SHADER), 5);
    assert_eq!(max_binding_index(TRANSPOSE_IBP_SHADER), Some(4));
}

#[test]
fn test_scale_ibp_shader_has_5_bindings() {
    // pipelines.rs: create_scale_ibp_pipeline has 5 entries (0..4)
    assert_eq!(count_bindings(SCALE_IBP_SHADER), 5);
    assert_eq!(max_binding_index(SCALE_IBP_SHADER), Some(4));
}

#[test]
fn test_gemm_f32_shader_has_4_bindings() {
    // pipelines.rs: create_gemm_f32_pipeline has 4 entries (0..3)
    assert_eq!(count_bindings(GEMM_F32_SHADER), 4);
    assert_eq!(max_binding_index(GEMM_F32_SHADER), Some(3));
}

#[test]
fn test_conv2d_ibp_shader_has_8_bindings() {
    assert_eq!(count_bindings(CONV2D_IBP_IM2COL_SHADER), 8);
    assert_eq!(max_binding_index(CONV2D_IBP_IM2COL_SHADER), Some(7));
}

#[test]
fn test_conv_reshape_shader_has_3_bindings() {
    // conv_pipelines.rs: create_conv_reshape_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(CONV_RESHAPE_SHADER), 3);
    assert_eq!(max_binding_index(CONV_RESHAPE_SHADER), Some(2));
}

#[test]
fn test_conv_col2im_shader_has_3_bindings() {
    // conv_pipelines.rs: create_conv_col2im_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(CONV_COL2IM_SHADER), 3);
    assert_eq!(max_binding_index(CONV_COL2IM_SHADER), Some(2));
}

#[test]
fn test_conv_taint_twins_have_5_bindings() {
    for shader in [CONV_RESHAPE_TAINT_SHADER, CONV_COL2IM_TAINT_SHADER] {
        assert_eq!(count_bindings(shader), 5);
        assert_eq!(max_binding_index(shader), Some(4));
    }
}

#[test]
fn test_small_k_taint_twin_has_7_bindings() {
    assert_eq!(count_bindings(GEMM_F32_SMALL_K_TAINT_SHADER), 7);
    assert_eq!(max_binding_index(GEMM_F32_SMALL_K_TAINT_SHADER), Some(6));
}

#[test]
fn test_relu_ibp_shader_has_3_bindings() {
    // pipelines.rs: create_relu_ibp_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(RELU_IBP_SHADER), 3);
    assert_eq!(max_binding_index(RELU_IBP_SHADER), Some(2));
}

#[test]
fn test_add_ibp_shader_has_7_bindings() {
    // pipelines.rs: create_add_ibp_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(ADD_IBP_SHADER), 7);
    assert_eq!(max_binding_index(ADD_IBP_SHADER), Some(6));
}

#[test]
fn test_crown_strided_gather_shader_has_4_bindings() {
    assert_eq!(count_bindings(CROWN_STRIDED_GATHER_SHADER), 4);
    assert_eq!(max_binding_index(CROWN_STRIDED_GATHER_SHADER), Some(3));
}

// ============================================================================
// Binding index sequential invariant
// ============================================================================
// All pipelines use sequential binding indices starting from 0 with no gaps.
// Missing indices cause wgpu validation errors at pipeline creation.

/// Verify that binding indices in a shader are sequential 0..N.
fn verify_sequential_bindings(shader_source: &str, shader_name: &str) {
    let mut indices = Vec::new();
    for (i, _) in shader_source.match_indices("@binding(") {
        let after = &shader_source[i + "@binding(".len()..];
        if let Some(end) = after.find(')') {
            if let Ok(idx) = after[..end].trim().parse::<u32>() {
                indices.push(idx);
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    let expected: Vec<u32> = (0..indices.len() as u32).collect();
    assert_eq!(
        indices,
        expected,
        "{shader_name}: binding indices must be sequential 0..{}, got {indices:?}",
        indices.len()
    );
}

#[test]
fn test_all_shaders_have_sequential_binding_indices() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("GEMM_F32_SMALL_K_SHADER", GEMM_F32_SMALL_K_SHADER),
        (
            "GEMM_F32_SMALL_K_TAINT_SHADER",
            GEMM_F32_SMALL_K_TAINT_SHADER,
        ),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_RESHAPE_TAINT_SHADER", CONV_RESHAPE_TAINT_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("CONV_COL2IM_TAINT_SHADER", CONV_COL2IM_TAINT_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
        ("CROWN_STRIDED_GATHER_SHADER", CROWN_STRIDED_GATHER_SHADER),
    ];
    for (name, source) in shaders {
        verify_sequential_bindings(source, name);
    }
}

// ============================================================================
// Binding 0 is uniform invariant
// ============================================================================
// Every pipeline has binding 0 as a uniform buffer for the Params struct.

#[test]
fn test_all_shaders_use_binding_0_as_uniform() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("GEMM_F32_SMALL_K_SHADER", GEMM_F32_SMALL_K_SHADER),
        (
            "GEMM_F32_SMALL_K_TAINT_SHADER",
            GEMM_F32_SMALL_K_TAINT_SHADER,
        ),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_RESHAPE_TAINT_SHADER", CONV_RESHAPE_TAINT_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("CONV_COL2IM_TAINT_SHADER", CONV_COL2IM_TAINT_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
        ("CROWN_STRIDED_GATHER_SHADER", CROWN_STRIDED_GATHER_SHADER),
    ];
    for (name, source) in shaders {
        assert!(
            source.contains("@binding(0) var<uniform>"),
            "{name}: binding 0 must be var<uniform> for Params struct"
        );
    }
}

// ============================================================================
// Fallback bound constant consistency (#2258, #2390)
// ============================================================================
// All shaders that perform arithmetic on bounds define FALLBACK_BOUND.
// These must match crate::FALLBACK_BOUND to ensure GPU and CPU paths produce
// consistent sanitization behavior.

#[test]
fn test_fallback_bound_consistent_across_shaders() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    // WGSL literal must match the Rust constant (1e10).
    let expected = "1e10";
    for (name, source) in shaders {
        assert!(
            source.contains(expected),
            "{name} must define FALLBACK_BOUND = {expected} (must match crate::FALLBACK_BOUND)"
        );
    }
}

#[test]
fn test_nan_safety_functions_present_in_ibp_shaders() {
    // IBP shaders that compute bounds via multiplication/accumulation must have
    // is_non_finite + nan_safe_lower + nan_safe_upper guards.
    // Part of #2390: added SCALE_IBP_SHADER.
    for (name, source) in [
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
    ] {
        assert!(
            source.contains("fn is_non_finite("),
            "{name} must define is_non_finite helper"
        );
        assert!(
            source.contains("fn nan_safe_lower("),
            "{name} must define nan_safe_lower helper"
        );
        assert!(
            source.contains("fn nan_safe_upper("),
            "{name} must define nan_safe_upper helper"
        );
    }
}

#[test]
fn test_nan_safety_functions_present_in_softmax_shaders() {
    // SOFTMAX_REDUCE_SHADER: must guard max reduction (max(x, NaN) is
    // implementation-defined in WGSL) and exp output overflow.
    // Part of #2390.
    assert!(
        SOFTMAX_REDUCE_SHADER.contains("fn is_non_finite("),
        "SOFTMAX_REDUCE_SHADER must define is_non_finite helper"
    );
    assert!(
        SOFTMAX_REDUCE_SHADER.contains("fn safe_exp("),
        "SOFTMAX_REDUCE_SHADER must define safe_exp helper for guarded exponentiation"
    );

    // SOFTMAX_APPLY_SHADER: must guard division output to catch NaN/Inf
    // and clamp probability bounds to [0, 1].
    // Part of #2390.
    assert!(
        SOFTMAX_APPLY_SHADER.contains("fn is_non_finite("),
        "SOFTMAX_APPLY_SHADER must define is_non_finite helper"
    );
}

#[test]
fn test_nan_safety_functions_present_in_gemm_shader() {
    // GEMM shader uses nan_safe_clamp (NaN → preserved for downstream detection,
    // Inf → clamp to ±FALLBACK_BOUND) via x != x check. #2708: NaN is no longer
    // replaced with 0.0 (which was unsound — silently drops coefficient contribution).
    // Part of #2366, #2708.
    assert!(
        GEMM_F32_SHADER.contains("fn nan_safe_clamp("),
        "GEMM_F32_SHADER must define nan_safe_clamp helper"
    );
    // Verify the NaN-specific check pattern (x != x distinguishes NaN from Inf).
    assert!(
        GEMM_F32_SHADER.contains("x != x"),
        "GEMM_F32_SHADER nan_safe_clamp must use x != x for NaN detection"
    );
}

#[test]
fn test_crown_concretize_shader_degrades_fallback_bound_coefficients_2708() {
    assert!(
        CROWN_CONCRETIZE_SHADER.contains("abs(a_l) >= FALLBACK_BOUND"),
        "CROWN_CONCRETIZE_SHADER must treat exact +/ -FALLBACK_BOUND lower coefficients as degraded"
    );
    assert!(
        CROWN_CONCRETIZE_SHADER.contains("abs(a_u) >= FALLBACK_BOUND"),
        "CROWN_CONCRETIZE_SHADER must treat exact +/ -FALLBACK_BOUND upper coefficients as degraded"
    );
}

#[path = "contract_plan_cache_tests.rs"]
mod contract_plan_cache_tests;
#[path = "contract_runtime_tests.rs"]
mod contract_runtime_tests;
