// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Source-level contract tests for the GPU CROWN plan cache key (#3397).
//!
//! `crown_plan_key(...)` is the aliasing boundary for cached GPU CROWN plans.
//! These tests pin the exact static/dynamic split in `ops/crown_plan_key.rs`
//! without widening the production visibility of the helper itself.
//!
//! #perf-plan-cache: the static-weight key was changed from a full content
//! hash (`hash_f32_slice` over every `f32`) to a pointer-identity + length
//! hash (`hash_arc_identity` over `Arc::as_ptr` + `len`). These tests are
//! updated to pin the new contract: static weights are still mixed into
//! `static_data` (so distinct weights still map to distinct keys), dynamic
//! slopes/routing are still excluded, and the batch-shape fields still
//! participate.

fn crown_plan_source() -> &'static str {
    include_str!("ops/crown_plan_key.rs")
}

fn crown_plan_key_source(source: &str) -> &str {
    source_block_after(source, "fn crown_plan_key(", "fn hash_arc_identity")
}

fn source_block_after<'a>(source: &'a str, anchor: &str, next_anchor: &str) -> &'a str {
    let start = source
        .find(anchor)
        .unwrap_or_else(|| panic!("expected to find anchor `{anchor}` in crown_plan_key.rs"));
    let after_start = &source[start..];
    let end = after_start.find(next_anchor).unwrap_or_else(|| {
        panic!("expected to find next anchor `{next_anchor}` after `{anchor}` in crown_plan_key.rs")
    });
    &after_start[..end]
}

#[test]
fn test_crown_plan_key_linear_arm_hashes_static_weights_and_bias_3397() {
    let source = crown_plan_source();
    let key_fn = crown_plan_key_source(source);
    let linear_arm = source_block_after(
        key_fn,
        "GpuCrownLayer::Linear {",
        "GpuCrownLayer::Activation { num_neurons, .. } => {",
    );

    for required in [
        "out_features.hash(&mut topology);",
        "in_features.hash(&mut topology);",
        "bias.is_some().hash(&mut topology);",
        // #perf-plan-cache: static weights keyed by Arc pointer identity, not
        // by hashing every f32. Still mixed into `static_data` so distinct
        // weight tensors map to distinct keys.
        "hash_arc_identity(&mut static_data, weight);",
        "hash_arc_identity(&mut static_data, bias);",
    ] {
        assert!(
            linear_arm.contains(required),
            "linear cache-key arm must contain `{required}`",
        );
    }
}

#[test]
fn test_crown_plan_key_activation_arm_hashes_only_topology_3397() {
    let source = crown_plan_source();
    let key_fn = crown_plan_key_source(source);
    let activation_arm = source_block_after(
        key_fn,
        "GpuCrownLayer::Activation { num_neurons, .. } => {",
        "GpuCrownLayer::MaxPool2d {",
    );

    assert!(
        activation_arm.contains("num_neurons.hash(&mut topology);"),
        "activation cache-key arm must hash neuron count into topology",
    );
    assert!(
        !activation_arm.contains("static_data"),
        "activation cache-key arm must not hash dynamic slopes/intercepts as static data",
    );
}

#[test]
fn test_crown_plan_key_maxpool_arm_hashes_only_topology_4211() {
    let source = crown_plan_source();
    let key_fn = crown_plan_key_source(source);
    let maxpool_arm = source_block_after(
        key_fn,
        "GpuCrownLayer::MaxPool2d {",
        "GpuCrownLayer::Conv2d {",
    );

    for required in [
        "input_dim.hash(&mut topology);",
        "output_dim.hash(&mut topology);",
    ] {
        assert!(
            maxpool_arm.contains(required),
            "maxpool cache-key arm must contain `{required}`",
        );
    }
    assert!(
        !maxpool_arm.contains("static_data"),
        "maxpool cache-key arm must not hash dynamic routing/bounds as static data",
    );
}

#[test]
fn test_crown_plan_key_conv_arm_hashes_geometry_and_static_weights_3397() {
    let source = crown_plan_source();
    let key_fn = crown_plan_key_source(source);
    let conv_arm = source_block_after(key_fn, "GpuCrownLayer::Conv2d {", "CrownPlanKey {");

    for required in [
        "stride_h.hash(&mut topology);",
        "stride_w.hash(&mut topology);",
        "pad_h.hash(&mut topology);",
        "pad_w.hash(&mut topology);",
        "out_h.hash(&mut topology);",
        "out_w.hash(&mut topology);",
        "in_h.hash(&mut topology);",
        "in_w.hash(&mut topology);",
        // #perf-plan-cache: conv kernel/bias keyed by Arc pointer identity.
        "hash_arc_identity(&mut static_data, weight_col);",
        "hash_arc_identity(&mut static_data, bias_expanded);",
    ] {
        assert!(
            conv_arm.contains(required),
            "conv cache-key arm must contain `{required}`",
        );
    }
}

#[test]
fn test_crown_plan_key_includes_batch_shape_fields_3397() {
    let source = crown_plan_source();
    let key_fn = crown_plan_key_source(source);

    for required in ["num_specs,", "first_dim,"] {
        assert!(
            key_fn.contains(required),
            "cache key construction must include `{required}`",
        );
    }
}
