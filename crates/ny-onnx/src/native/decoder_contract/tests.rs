// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::certify::work::{WorkBudget, WorkRefusal};
use std::time::{Duration, Instant};

#[path = "observation_tests.rs"]
mod observation_tests;

#[test]
fn decoder_n1_observation_identity_binds_point_bits_output_selection_and_completed_arms() {
    use ny_propagate::layers::PointLinearArm;
    use ny_propagate::network::GraphPointArm;
    let options = DecoderOptions::default();
    let model = model(false, &[0, 1]);
    let graph = model.graph_for("logits", &options).unwrap();
    let input = model.embedded_tokens(&[1, 3], &options).unwrap();
    let ordinary = graph.point(input.values(), &options).unwrap();
    let repeated = graph.point(input.values(), &options).unwrap();
    assert_eq!(ordinary.execution_digest(), repeated.execution_digest());
    assert_eq!(
        ordinary.lowering_digest(),
        &model.manifest().lowering_digest
    );
    assert!(ordinary.execution_trace().steps().iter().any(|s|matches!(s.arm,GraphPointArm::PointLinear {receipt} if receipt.arm==PointLinearArm::OriginalRowMajorFaerMatvecF32)));
    let finite_options = DecoderOptions {
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..options.clone()
    };
    let finite = graph.point(input.values(), &finite_options).unwrap();
    assert_eq!(ordinary.input_digest(), finite.input_digest());
    assert_ne!(ordinary.execution_digest(), finite.execution_digest());
    assert!(finite
        .execution_trace()
        .steps()
        .iter()
        .filter_map(|s| if let GraphPointArm::PointLinear { receipt } = s.arm {
            Some(receipt)
        } else {
            None
        })
        .all(|receipt| receipt.arm == PointLinearArm::ScalarF64OnceF32));
    let mut changed = input.values().clone();
    changed[[0, 0]] = f32::from_bits(changed[[0, 0]].to_bits() ^ 1);
    let changed = graph.point(&changed, &options).unwrap();
    assert_ne!(changed.input_digest(), ordinary.input_digest());
    assert_ne!(changed.execution_digest(), ordinary.execution_digest());
    let hidden = model
        .graph_for("hidden", &options)
        .unwrap()
        .point(input.values(), &options)
        .unwrap();
    assert_ne!(hidden.selected_output(), ordinary.selected_output());
    assert_ne!(hidden.execution_digest(), ordinary.execution_digest());
}

#[test]
fn decoder_n1_point_propagates_original_optional_lease_deadline_without_replacement() {
    use ny_propagate::layers::PointLinearArm;
    use ny_propagate::network::GraphPointArm;
    let deadline = Instant::now() + Duration::from_secs(30);
    let budget = WorkBudget::new(
        Some(deadline),
        64 << 20,
        128 << 20,
        1 << 32,
        1000,
        Duration::from_secs(5),
    )
    .unwrap();
    let options = DecoderOptions {
        budget: Some(budget.clone()),
        ..DecoderOptions::default()
    };
    let model = lower_decoder(
        DecoderWeights::from_store(weights(false), &options).unwrap(),
        &contract(false),
        &[0, 1],
        &options,
    )
    .unwrap();
    let graph = model.graph_for("logits", &options).unwrap();
    let input = model.embedded_tokens(&[1, 3], &options).unwrap();
    let before = budget.receipt().unwrap();
    let point = graph.point(input.values(), &options).unwrap();
    let linear: Vec<_> = point
        .execution_trace()
        .steps()
        .iter()
        .filter_map(|s| {
            if let GraphPointArm::PointLinear { receipt } = s.arm {
                Some(receipt)
            } else {
                None
            }
        })
        .collect();
    assert_ne!((linear).len(), 0);
    assert!(
        linear
            .iter()
            .all(|r| r.arm == PointLinearArm::ScalarF64OnceF32),
        "options have no deadline; the original lease's cutoff must still reach kernels"
    );
    let after = budget.receipt().unwrap();
    assert_eq!(
        after.work_reserved - before.work_reserved,
        graph.point_work_bound()
    );
    assert_eq!(
        after.live_persistent_bytes - before.live_persistent_bytes,
        point.reserved_bytes()
    );
    drop(point);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    let expired = DecoderOptions {
        deadline: Some(Instant::now()),
        ..options
    };
    let before = budget.receipt().unwrap();
    assert!(graph.point(input.values(), &expired).is_err());
    assert_eq!(
        budget.receipt().unwrap(),
        before,
        "expired authority must refuse before another allocation/lease"
    );
}

fn config(qwen: bool) -> serde_json::Value {
    if qwen {
        serde_json::json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],
        "hidden_size":8,"intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,
        "num_hidden_layers":2,"vocab_size":11,"max_position_embeddings":32768,
        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":1000000.0,
        "tie_word_embeddings":true,"use_sliding_window":false,"sliding_window":32768,
        "max_window_layers":24,"use_mrope":false})
    } else {
        serde_json::json!({"model_type":"gpt_neox","architectures":["GPTNeoXForCausalLM"],
        "hidden_size":8,"intermediate_size":12,"num_attention_heads":2,
        "num_hidden_layers":2,"vocab_size":11,"max_position_embeddings":32768,
        "hidden_act":"gelu","layer_norm_eps":0.00001,"rotary_pct":0.5,
        "rotary_emb_base":10000,"use_parallel_residual":true,"attention_bias":true,
        "tie_word_embeddings":false})
    }
}
fn contract(qwen: bool) -> DecoderContract {
    DecoderContract::from_json(
        &serde_json::to_vec(&config(qwen)).unwrap(),
        &DecoderOptions::default(),
    )
    .unwrap()
}
fn insert(ws: &mut WeightStore, name: &str, shape: &[usize], scale: f32) {
    let len: usize = shape.iter().product();
    let values = (0..len)
        .map(|k| ((k * 37 + name.len() * 11) % 103) as f32 / 103.0 * scale - scale / 2.0)
        .collect();
    ws.insert(
        name.into(),
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
    );
}
fn norm(ws: &mut WeightStore, prefix: &str, rms: bool) {
    ws.insert(
        format!("{prefix}.weight"),
        ArrayD::from_elem(IxDyn(&[8]), 1.0),
    );
    if !rms {
        insert(ws, &format!("{prefix}.bias"), &[8], 0.1);
    }
}
fn weights(qwen: bool) -> WeightStore {
    let mut ws = WeightStore::new();
    let embed = if qwen {
        "model.embed_tokens.weight"
    } else {
        "gpt_neox.embed_in.weight"
    };
    insert(&mut ws, embed, &[11, 8], 0.7);
    if !qwen {
        insert(&mut ws, "embed_out.weight", &[11, 8], 0.2);
    }
    norm(
        &mut ws,
        if qwen {
            "model.norm"
        } else {
            "gpt_neox.final_layer_norm"
        },
        qwen,
    );
    for i in 0..2 {
        let p = if qwen {
            format!("model.layers.{i}")
        } else {
            format!("gpt_neox.layers.{i}")
        };
        norm(&mut ws, &format!("{p}.input_layernorm"), qwen);
        norm(&mut ws, &format!("{p}.post_attention_layernorm"), qwen);
        if qwen {
            for (which, out) in [("q", 8), ("k", 4), ("v", 4)] {
                insert(
                    &mut ws,
                    &format!("{p}.self_attn.{which}_proj.weight"),
                    &[out, 8],
                    0.25,
                );
                insert(
                    &mut ws,
                    &format!("{p}.self_attn.{which}_proj.bias"),
                    &[out],
                    0.03,
                );
            }
            insert(
                &mut ws,
                &format!("{p}.self_attn.o_proj.weight"),
                &[8, 8],
                0.25,
            );
            for which in ["gate", "up"] {
                insert(
                    &mut ws,
                    &format!("{p}.mlp.{which}_proj.weight"),
                    &[12, 8],
                    0.25,
                );
            }
            insert(
                &mut ws,
                &format!("{p}.mlp.down_proj.weight"),
                &[8, 12],
                0.25,
            );
        } else {
            insert(
                &mut ws,
                &format!("{p}.attention.query_key_value.weight"),
                &[24, 8],
                0.25,
            );
            insert(
                &mut ws,
                &format!("{p}.attention.query_key_value.bias"),
                &[24],
                0.03,
            );
            insert(
                &mut ws,
                &format!("{p}.attention.dense.weight"),
                &[8, 8],
                0.25,
            );
            insert(&mut ws, &format!("{p}.attention.dense.bias"), &[8], 0.03);
            insert(
                &mut ws,
                &format!("{p}.mlp.dense_h_to_4h.weight"),
                &[12, 8],
                0.25,
            );
            insert(&mut ws, &format!("{p}.mlp.dense_h_to_4h.bias"), &[12], 0.03);
            insert(
                &mut ws,
                &format!("{p}.mlp.dense_4h_to_h.weight"),
                &[8, 12],
                0.25,
            );
            insert(&mut ws, &format!("{p}.mlp.dense_4h_to_h.bias"), &[8], 0.03);
        }
    }
    ws
}
fn model(qwen: bool, positions: &[u64]) -> LoweredDecoder {
    let options = DecoderOptions::default();
    lower_decoder(
        DecoderWeights::from_store(weights(qwen), &options).unwrap(),
        &contract(qwen),
        positions,
        &options,
    )
    .unwrap()
}

#[test]
fn decoder_n1_closed_config_refuses_unknown_semantics_and_wrong_architecture() {
    let options = DecoderOptions::default();
    for (key, value) in [
        ("mystery_attention", serde_json::json!(true)),
        (
            "rope_scaling",
            serde_json::json!({"type":"linear","factor":2.0}),
        ),
        ("quantization_config", serde_json::json!({"bits":4})),
        ("hidden_act", serde_json::json!("gelu_new")),
        ("architectures", serde_json::json!(["LlamaForCausalLM"])),
    ] {
        let mut c = config(false);
        c[key] = value;
        assert!(
            DecoderContract::from_json(&serde_json::to_vec(&c).unwrap(), &options).is_err(),
            "{key}"
        );
    }
    let mut q = config(true);
    q["use_sliding_window"] = serde_json::json!(true);
    assert!(DecoderContract::from_json(&serde_json::to_vec(&q).unwrap(), &options).is_err());
    q = config(true);
    q.as_object_mut().unwrap().remove("num_key_value_heads");
    assert!(DecoderContract::from_json(&serde_json::to_vec(&q).unwrap(), &options).is_err());
    q = config(true);
    q["num_key_value_heads"] = serde_json::json!(3);
    assert!(DecoderContract::from_json(&serde_json::to_vec(&q).unwrap(), &options).is_err());
    q = config(true);
    q["head_dim"] = serde_json::json!(6);
    assert_eq!(
        DecoderContract::from_json(&serde_json::to_vec(&q).unwrap(), &options)
            .unwrap()
            .parameters()
            .head_dim,
        6
    );
    let bytes = serde_json::to_string(&config(false)).unwrap();
    let duplicate = bytes.replacen('{', "{\"layer_norm_eps\":0.123,", 1);
    assert!(DecoderContract::from_json(duplicate.as_bytes(), &options).is_err());
}

#[test]
fn decoder_n1_nonstandard_qwen_head_dimension_reaches_actual_graph() {
    let options = DecoderOptions::default();
    let mut c = config(true);
    c["head_dim"] = serde_json::json!(6);
    let contract = DecoderContract::from_json(&serde_json::to_vec(&c).unwrap(), &options).unwrap();
    let mut ws = weights(true);
    for index in 0..2 {
        for (which, out) in [("q", 12), ("k", 6), ("v", 6)] {
            insert(
                &mut ws,
                &format!("model.layers.{index}.self_attn.{which}_proj.weight"),
                &[out, 8],
                0.25,
            );
            insert(
                &mut ws,
                &format!("model.layers.{index}.self_attn.{which}_proj.bias"),
                &[out],
                0.03,
            );
        }
        insert(
            &mut ws,
            &format!("model.layers.{index}.self_attn.o_proj.weight"),
            &[8, 12],
            0.25,
        );
    }
    let model = lower_decoder(
        DecoderWeights::from_store(ws, &options).unwrap(),
        &contract,
        &[1024, 1025],
        &options,
    )
    .unwrap();
    let input = model.embedded_tokens(&[1, 3], &options).unwrap();
    let output = model
        .graph_for("logits", &options)
        .unwrap()
        .point(input.values(), &options)
        .unwrap();
    assert_eq!(output.values().shape(), [1, 2, 11]);
    assert!(output.values().iter().all(|x| x.is_finite()));
}

#[test]
fn decoder_n1_safetensors_load_hashes_original_encoding_and_exact_decoded_bits() {
    use std::io::Write;
    let options = DecoderOptions::default();
    let header = serde_json::to_vec(&serde_json::json!({
        "a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},
        "b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}
    }))
    .unwrap();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for bits in [
        half::f16::from_f32(0.1).to_bits(),
        half::f16::from_f32(-3.0).to_bits(),
        half::bf16::from_f32(0.1).to_bits(),
        half::bf16::from_f32(-3.0).to_bits(),
    ] {
        file.write_all(&bits.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    let loaded = load_decoder_safetensors(file.path(), &options).unwrap();
    assert_eq!(loaded.coefficients, 4);
    assert_eq!(loaded.source_bytes, 8);
    let a = loaded.weights.get("a").unwrap();
    let b = loaded.weights.get("b").unwrap();
    assert_eq!(
        a[IxDyn(&[0])].to_bits(),
        half::f16::from_f32(0.1).to_f32().to_bits()
    );
    assert_eq!(
        b[IxDyn(&[0])].to_bits(),
        half::bf16::from_f32(0.1).to_f32().to_bits()
    );
    assert_ne!(a[IxDyn(&[0])].to_bits(), b[IxDyn(&[0])].to_bits());
    assert_eq!(loaded.identities["a"].source_dtype, "F16");
    assert_eq!(loaded.identities["b"].source_dtype, "BF16");
    assert_ne!(
        loaded.identities["a"].source_digest,
        loaded.identities["a"].decoded_f32_digest
    );
    let limited = DecoderOptions {
        max_file_bytes: 1,
        ..options
    };
    assert!(matches!(
        load_decoder_safetensors(file.path(), &limited),
        Err(DecoderContractRefusal::Resource(_))
    ));
}

#[test]
fn decoder_n1_named_graphs_execute_all_blocks_logits_and_causal_prefix() {
    let options = DecoderOptions::default();
    for qwen in [false, true] {
        let model = model(qwen, &[0, 1, 2]);
        let input = model.embedded_tokens(&[1, 3, 7], &options).unwrap();
        let graph = model.graph_for("logits", &options).unwrap();
        let original = graph.point(input.values(), &options).unwrap();
        assert_eq!(original.values().shape(), [1, 3, 11]);
        let mut changed = input.values().clone();
        for k in 0..8 {
            changed[IxDyn(&[2, k])] += 0.25 + k as f32 * 0.1;
        }
        let future = graph.point(&changed, &options).unwrap();
        for token in 0..11 {
            assert_eq!(
                original.values()[IxDyn(&[0, 0, token])].to_bits(),
                future.values()[IxDyn(&[0, 0, token])].to_bits(),
                "future affected prefix, qwen={qwen}"
            );
        }
        assert!(original
            .values()
            .iter()
            .zip(future.values())
            .any(|(a, b)| (a - b).abs() > 1e-5));
        for output in ["block.0", "block.1", "hidden"] {
            let graph = model.graph_for(output, &options).unwrap();
            let result = graph.point(input.values(), &options).unwrap();
            assert_eq!(result.values().shape(), [1, 3, 8]);
            assert!(result.values().iter().all(|x| x.is_finite()));
        }
    }
}

#[test]
fn decoder_n1_manifest_binds_epsilon_coefficients_positions_and_gqa_mapping() {
    let options = DecoderOptions::default();
    let original = model(false, &[0, 1]);
    assert_eq!(original.manifest().parameters.reference_profile,
        "decoded-f32-coefficients/ny-graph-point-mixed-f32-f64/linear-original-weight-f32-or-deadline-f64/causal-max-shift-libm-f64-one-f32-store/completed-arm-observation-v1/half-split-pinned-table/v3");
    let mut identity = original.manifest().clone();
    identity.lowering_digest = [0; 32];
    assert_eq!(
        manifest::digest_serialized(b"lowering", &identity).unwrap(),
        original.manifest().lowering_digest
    );
    identity.parameters.reference_profile =
        "decoded-coefficients/binary32-cpu-eager/half-split-pinned-table/v1".into();
    assert_ne!(
        manifest::digest_serialized(b"lowering", &identity).unwrap(),
        original.manifest().lowering_digest,
        "chosen diagnostic point dispatch is part of the content identity"
    );
    let changed_positions = model(false, &[1024, 1025]);
    assert_ne!(
        original.manifest().lowering_digest,
        changed_positions.manifest().lowering_digest
    );
    let mut c = config(false);
    c["layer_norm_eps"] = serde_json::json!(0.001);
    let c = DecoderContract::from_json(&serde_json::to_vec(&c).unwrap(), &options).unwrap();
    let changed = lower_decoder(
        DecoderWeights::from_store(weights(false), &options).unwrap(),
        &c,
        &[0, 1],
        &options,
    )
    .unwrap();
    assert_ne!(
        original.manifest().lowering_digest,
        changed.manifest().lowering_digest
    );
    let mut ws = weights(false);
    ws.insert(
        "gpt_neox.layers.0.input_layernorm.weight".into(),
        ArrayD::from_elem(IxDyn(&[8]), 1.01),
    );
    let changed = lower_decoder(
        DecoderWeights::from_store(ws, &options).unwrap(),
        &contract(false),
        &[0, 1],
        &options,
    )
    .unwrap();
    assert_ne!(
        original.manifest().blocks[1].digest,
        changed.manifest().blocks[1].digest
    );
    assert_eq!(
        original.manifest().blocks[2].digest,
        changed.manifest().blocks[2].digest,
        "unaffected block coefficient identities stable"
    );
    let q = model(true, &[1024, 1025]);
    let ops = &q.manifest().blocks[1].operations;
    assert!(ops
        .iter()
        .any(|op| op.name.ends_with("key_gqa.repeat_groups") && op.parameters["reps"] == 2));
    assert!(ops
        .iter()
        .any(|op| op.name.ends_with("value_gqa.repeat_groups")));
    assert!(ops.iter().any(|op| op.name.ends_with("swiglu")));
    assert!(original.manifest().blocks[1]
        .operations
        .iter()
        .any(|op| op.name.ends_with("parallel_branches")));
    assert!(q
        .manifest()
        .original_tensors
        .iter()
        .all(|t| t.source_dtype == "F32-decoded-input"));
}

#[test]
fn decoder_n1_original_tensor_shape_tied_head_and_unused_tensor_refuse() {
    let options = DecoderOptions::default();
    let mut wrong = weights(false);
    insert(
        &mut wrong,
        "gpt_neox.layers.0.attention.query_key_value.weight",
        &[8, 24],
        0.1,
    );
    assert!(lower_decoder(
        DecoderWeights::from_store(wrong, &options).unwrap(),
        &contract(false),
        &[0],
        &options
    )
    .is_err());
    let mut tied = weights(true);
    insert(&mut tied, "lm_head.weight", &[11, 8], 0.9);
    assert!(lower_decoder(
        DecoderWeights::from_store(tied, &options).unwrap(),
        &contract(true),
        &[0],
        &options
    )
    .is_err());
    let mut unused = weights(false);
    insert(
        &mut unused,
        "gpt_neox.layers.0.extra_semantic.weight",
        &[8],
        0.1,
    );
    assert!(lower_decoder(
        DecoderWeights::from_store(unused, &options).unwrap(),
        &contract(false),
        &[0],
        &options
    )
    .is_err());
    let mut nan = weights(false);
    nan.insert(
        "embed_out.weight".into(),
        ArrayD::from_elem(IxDyn(&[11, 8]), f32::NAN),
    );
    assert!(DecoderWeights::from_store(nan, &options).is_err());
}

#[test]
fn decoder_n1_resource_refusal_precedes_graph_copy_and_preserves_residency() {
    let budget = WorkBudget::new(None, 8 << 20, 32 << 20, 1 << 30, 100, Duration::ZERO).unwrap();
    let options = DecoderOptions {
        budget: Some(budget.clone()),
        ..DecoderOptions::default()
    };
    let model = lower_decoder(
        DecoderWeights::from_store(weights(false), &options).unwrap(),
        &contract(false),
        &[0, 1],
        &options,
    )
    .unwrap();
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let limited = DecoderOptions {
        max_scratch_bytes: 1,
        ..options
    };
    assert!(matches!(
        model.graph_for("logits", &limited),
        Err(DecoderContractRefusal::Resource(_))
    ));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    let starved = WorkBudget::new(None, 1 << 20, 1 << 20, 0, 10, Duration::ZERO).unwrap();
    let opts = DecoderOptions {
        budget: Some(starved),
        ..DecoderOptions::default()
    };
    assert!(matches!(
        DecoderWeights::from_store(weights(false), &opts),
        Err(DecoderContractRefusal::Budget(WorkRefusal::WorkLimit))
    ));
    drop(model);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n1_linear_causal_point_work_is_admitted_and_charged_before_execution() {
    let options = DecoderOptions::default();
    let positions = (0..64u64).collect::<Vec<_>>();
    let model = model(true, &positions);
    let input = model.embedded_tokens(&vec![1; 64], &options).unwrap();
    let graph = model.graph_for("logits", &options).unwrap();
    // The actual diagnostic causal evaluator is O(visible pairs). Its work
    // joins projection/QK/PV work in the original point lease; no certificate
    // checker or separate work allowance runs in this point-only route.
    let actual_work = graph.point_work_bound();
    let causal_work =
        ny_propagate::layers::CausalSoftmaxLayer::point_work_bound(&[2, 64, 64], None).unwrap() * 2;
    let legacy_cubic_work = 2 * 2 * 64u64.pow(3) * 8;
    assert!(causal_work < legacy_cubic_work);
    let unchanged_other_work = actual_work.checked_sub(causal_work).unwrap();
    assert!(actual_work < unchanged_other_work + legacy_cubic_work);
    let limited = DecoderOptions {
        max_terms: actual_work,
        ..options.clone()
    };
    let point = graph.point(input.values(), &limited).unwrap();
    assert_eq!(point.values().shape(), &[1, 64, 11]);
    assert!(point.values().iter().all(|x| x.is_finite()));
    let starved = DecoderOptions {
        max_terms: graph.point_work_bound() - 1,
        ..options
    };
    assert!(matches!(
        graph.point(input.values(), &starved),
        Err(DecoderContractRefusal::Resource(_))
    ));
}

#[test]
fn decoder_n10_realistic_original_owners_are_admitted_before_any_graph_allocation() {
    let (hidden, intermediate, vocab) = (256, 1024, 4096);
    let budget = WorkBudget::new(None, 256 << 20, 512 << 20, 1 << 32, 128, Duration::ZERO).unwrap();
    let opts = DecoderOptions {
        budget: Some(budget.clone()),
        max_resident_bytes: 512 << 20,
        max_scratch_bytes: 256 << 20,
        ..DecoderOptions::default()
    };
    let mut c = config(false);
    c["hidden_size"] = serde_json::json!(hidden);
    c["intermediate_size"] = serde_json::json!(intermediate);
    c["vocab_size"] = serde_json::json!(vocab);
    c["num_hidden_layers"] = serde_json::json!(1);
    c["num_attention_heads"] = serde_json::json!(4);
    c["rotary_pct"] = serde_json::json!(0.25);
    let contract = DecoderContract::from_json(&serde_json::to_vec(&c).unwrap(), &opts).unwrap();
    let mut ws = WeightStore::new();
    insert(&mut ws, "gpt_neox.embed_in.weight", &[vocab, hidden], 0.1);
    insert(&mut ws, "embed_out.weight", &[vocab, hidden], 0.01);
    for prefix in [
        "gpt_neox.layers.0.input_layernorm",
        "gpt_neox.layers.0.post_attention_layernorm",
        "gpt_neox.final_layer_norm",
    ] {
        ws.insert(
            format!("{prefix}.weight"),
            ArrayD::from_elem(IxDyn(&[hidden]), 1.0),
        );
        insert(&mut ws, &format!("{prefix}.bias"), &[hidden], 0.01);
    }
    for (name, out, inner) in [
        (
            "gpt_neox.layers.0.attention.query_key_value",
            3 * hidden,
            hidden,
        ),
        ("gpt_neox.layers.0.attention.dense", hidden, hidden),
        ("gpt_neox.layers.0.mlp.dense_h_to_4h", intermediate, hidden),
        ("gpt_neox.layers.0.mlp.dense_4h_to_h", hidden, intermediate),
    ] {
        insert(&mut ws, &format!("{name}.weight"), &[out, inner], 0.01);
        insert(&mut ws, &format!("{name}.bias"), &[out], 0.001);
    }
    let model = lower_decoder(
        DecoderWeights::from_store(ws, &opts).unwrap(),
        &contract,
        &[0, 1],
        &opts,
    )
    .unwrap();
    let linear_coefficients: usize = model
        .graph_model()
        .network
        .layers
        .iter()
        .filter(|s| s.layer_type == ny_core::LayerType::Linear)
        .map(|s| model.graph_model().weights.get(&s.inputs[1]).unwrap().len())
        .sum();
    let admission = model.graph_admission(&opts).unwrap();
    assert!(admission.persistent_bytes >= 4 * linear_coefficients);
    assert!(admission.persistent_bytes < 8 * linear_coefficients);
    // The source owner remains resident while the complete graph is built.
    let before = budget.receipt().unwrap();
    let cap = DecoderOptions {
        max_resident_bytes: before.live_persistent_bytes + admission.persistent_bytes - 1,
        ..opts.clone()
    };
    assert!(matches!(
        model.graph_for("block.0", &cap),
        Err(DecoderContractRefusal::Resource(_))
    ));
    assert_eq!(
        budget.receipt().unwrap(),
        before,
        "preflight refusal must precede graph allocation/lease"
    );
    let graph = model.graph_for("block.0", &opts).unwrap();
    let initial = graph.linear_retained_bytes().unwrap();
    assert!(
        initial >= 4 * linear_coefficients && initial < 8 * linear_coefficients,
        "one original copy, bias, and owner controls include the unselected head"
    );
    assert!(initial <= graph.reserved_bytes());
    assert_eq!(graph.reserved_bytes(), admission.persistent_bytes);
    assert!(
        admission.work > linear_coefficients as u64 * 2,
        "original coefficient copy, finite checks, and hashing are charged"
    );
    assert!(admission.work < linear_coefficients as u64 * 8);
    // The point matvec needs no padded coefficient matrix or derived caches.
    assert!(graph.point_scratch_bytes() >= 2 * vocab * 16);
    let compact_nodes = graph
        .graph()
        .node_names()
        .iter()
        .filter(|name| {
            matches!(
                graph.graph().node(name).unwrap().layer(),
                ny_propagate::Layer::PointLinear(_)
            )
        })
        .count();
    assert_eq!(compact_nodes, 5);
    assert!(graph.graph().node_names().iter().all(|name| {
        !matches!(
            graph.graph().node(name).unwrap().layer(),
            ny_propagate::Layer::Linear(_)
        )
    }));
    let input = model.embedded_tokens(&[1, 3], &opts).unwrap();
    let observed = graph.point(input.values(), &opts).unwrap();
    assert_eq!(observed.values().shape(), [1, 2, hidden]);
    assert_eq!(
        graph.linear_retained_bytes().unwrap(),
        initial,
        "the original-W point contraction needs no absolute-weight cache"
    );
    // Point-only storage cannot acquire proof eligibility from degenerate
    // supplied intervals or fill unpriced derived caches through raw access.
    let interval_input =
        ny_propagate::prelude::BoundedTensor::concrete(input.values().clone()).unwrap();
    let receipt_before_interval = budget.receipt().unwrap();
    assert!(graph.graph().propagate_ibp(&interval_input).is_err());
    assert_eq!(budget.receipt().unwrap(), receipt_before_interval);
    drop(interval_input);
    let filled = graph.linear_retained_bytes().unwrap();
    assert_eq!(
        filled, initial,
        "refusal creates no derived coefficient cache"
    );
    let repeated = graph.point(input.values(), &opts).unwrap();
    assert_eq!(graph.linear_retained_bytes().unwrap(), filled);
    assert_eq!(observed.values(), repeated.values());
    eprintln!("[decoder-n10-memory] linear_coefficients={linear_coefficients} initial_linear_bytes={initial} filled_linear_bytes={filled} graph_reserved={} construction_scratch={} point_scratch={} construction_work={} point_work={}",
        graph.reserved_bytes(), admission.scratch_bytes, graph.point_scratch_bytes(), admission.work, graph.point_work_bound());
    drop(repeated);
    drop(observed);
    drop(input);
    drop(graph);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(model);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}
