// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::{DecoderContract, DecoderWeights};
use super::*;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{WorkBudget, WorkLimits, WorkReserve},
};
use std::cell::RefCell;
use std::time::Duration;

thread_local! {
    static CANCEL_ROW: RefCell<Option<(usize,CancellationToken)>>=const{RefCell::new(None)};
}
pub(super) fn after_affine_row() {
    CANCEL_ROW.with(|hook| {
        let mut hook = hook.borrow_mut();
        if let Some((remaining, token)) = hook.as_mut() {
            *remaining -= 1;
            if *remaining == 0 {
                token.cancel();
                *hook = None;
            }
        }
    });
}

fn options() -> DecoderOptions {
    let at = Instant::now() + Duration::from_secs(30);
    DecoderOptions {
        deadline: Some(at),
        budget: Some(
            WorkBudget::new(Some(at), 16 << 20, 16 << 20, 1 << 32, 100, Duration::ZERO).unwrap(),
        ),
        ..DecoderOptions::default()
    }
}
fn insert(ws: &mut WeightStore, name: &str, shape: &[usize], values: Vec<f32>) {
    ws.insert(
        name.into(),
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
    );
}
fn zero(ws: &mut WeightStore, name: &str, shape: &[usize]) {
    insert(ws, name, shape, vec![0.0; shape.iter().product()]);
}
fn fixture(options: &DecoderOptions) -> PretrainedDecoder {
    let contract=DecoderContract::from_json(br#"{"model_type":"gpt_neox","hidden_size":2,"intermediate_size":2,"vocab_size":3,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":32,"hidden_act":"gelu","layer_norm_eps":0.00001,"rotary_pct":1.0,"rotary_emb_base":10000,"use_parallel_residual":true,"attention_bias":true,"tie_word_embeddings":false}"#,options).unwrap();
    let mut ws = WeightStore::new();
    insert(
        &mut ws,
        "gpt_neox.embed_in.weight",
        &[3, 2],
        vec![0.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    );
    insert(
        &mut ws,
        "embed_out.weight",
        &[3, 2],
        vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    );
    for norm in [
        "gpt_neox.final_layer_norm",
        "gpt_neox.layers.0.input_layernorm",
        "gpt_neox.layers.0.post_attention_layernorm",
    ] {
        insert(&mut ws, &format!("{norm}.weight"), &[2], vec![1.0, 1.0]);
        zero(&mut ws, &format!("{norm}.bias"), &[2]);
    }
    for (suffix, out, inner) in [
        ("attention.query_key_value", 6, 2),
        ("attention.dense", 2, 2),
        ("mlp.dense_h_to_4h", 2, 2),
        ("mlp.dense_4h_to_h", 2, 2),
    ] {
        zero(
            &mut ws,
            &format!("gpt_neox.layers.0.{suffix}.weight"),
            &[out, inner],
        );
        zero(&mut ws, &format!("gpt_neox.layers.0.{suffix}.bias"), &[out]);
    }
    // Nonzero original attention and MLP residuals ensure neither is omitted.
    insert(
        &mut ws,
        "gpt_neox.layers.0.attention.dense.bias",
        &[2],
        vec![0.5, -0.5],
    );
    insert(
        &mut ws,
        "gpt_neox.layers.0.mlp.dense_4h_to_h.bias",
        &[2],
        vec![0.25, -0.25],
    );
    super::super::pretrained::authored_fixture(
        DecoderWeights::from_store(ws, options).unwrap(),
        contract.into_parameters(),
        options.deadline.unwrap(),
    )
    .unwrap()
}

#[test]
fn reachable_decoder_genuine_complete_original_nonzero_residual_positive_radius_rational_bounds() {
    let options = options();
    let model = fixture(&options);
    let result = certify_reachable_next_token(&model, &[1], 1.0 / 1024.0, 0, &options).unwrap();
    assert!(result.proven().unwrap());
    assert!(result.margin().unwrap().lo > 0.99);
    let hidden = result.final_hidden().unwrap();
    let logits = result.logits().unwrap();
    // Whole original function is [a,-a]/sqrt(a²+1/100000),
    // a=7/4+(delta0-delta1)/2. Independently squared rational brackets:
    // .99²*(a²+1/100000)<a²<1²*(a²+1/100000).
    for numerator in [1791i64, 1792, 1793] {
        let a = numerator as f64 / 1024.0;
        assert!(
            99i128 * 99 * (100000 * (numerator as i128) * (numerator as i128) + 1024i128 * 1024)
                < 10000i128 * 100000 * (numerator as i128) * (numerator as i128)
        );
        // Independently bisect the rational square comparison in integer
        // arithmetic; no decoder helper or host sqrt supplies this oracle.
        let n = 100000u128 * (numerator as u128) * (numerator as u128);
        let d = n + 1024u128 * 1024;
        let scale = 1u128 << 40;
        let (mut low, mut high) = (0u128, scale);
        while low + 1 < high {
            let mid = low.midpoint(high);
            if mid * mid * d <= n * scale * scale {
                low = mid;
            } else {
                high = mid;
            }
        }
        let exact_lower = low as f64 / scale as f64;
        let exact_upper = high as f64 / scale as f64;
        assert!(
            hidden[0].lo <= exact_lower && hidden[0].hi >= exact_upper,
            "{hidden:?}; a={a}"
        );
    }
    assert!(hidden[0].lo > 0.99 && hidden[0].hi < 1.01);
    assert!(hidden[1].lo < -0.99 && hidden[1].hi > -1.01);
    assert!(logits[0].lo > 0.99 && logits[2].hi < -0.99);
    let r = result.receipt();
    assert_eq!(
        (
            r.complete_blocks,
            r.layer_normalizations,
            r.causal_attention_heads,
            r.gelu_values,
            r.vocabulary_rows,
            r.horizon
        ),
        (1, 3, 1, 2, 3, 1)
    );
    assert!(r.serialized_affine_terms > 0);
    assert_eq!(result.token_ids(), [1]);
    assert_eq!(result.embedding_radius(), 1.0 / 1024.0);
}

#[test]
fn reachable_decoder_original_causal_coordinates_uniform_attention_and_half_split_rope() {
    let options = options();
    let model = fixture(&options);
    let mut math = CheckedDecoderMath::new(
        model.budget(),
        options.deadline,
        DecoderMathCounts {
            elementary: 4096,
            trig: 8,
            roots: 8,
            ..DecoderMathCounts::default()
        },
    )
    .unwrap();
    // Original packed [Q0,Q1,K0,K1,V0,V1] for both positions. Every
    // score is0; complete causal means t0V0, t1(V0+V1)/2.
    let qkv = vec![
        I::point(0.0),
        I::point(0.0),
        I::point(0.0),
        I::point(0.0),
        I::point(1.0),
        I::point(3.0),
        I::point(0.0),
        I::point(0.0),
        I::point(0.0),
        I::point(0.0),
        I::point(5.0),
        I::point(7.0),
    ];
    let output = attention(&model, &mut math, &qkv, 2, I::point(1.0)).unwrap();
    for (x, v) in output.iter().zip([1.0, 3.0, 3.0, 5.0]) {
        assert!(x.lo <= v && v <= x.hi);
    }
    let mut packed = qkv.clone();
    packed[0] = I::point(1.0);
    packed[1] = I::point(2.0);
    packed[2] = I::point(3.0);
    packed[3] = I::point(4.0);
    let identity = (I::point(0.0), I::point(1.0));
    rotary(&mut math, &mut packed, 2, 2, 1, 2, 2, &[identity, identity]).unwrap();
    for k in 0..4 {
        let v = (k + 1) as f64;
        assert!(packed[k].lo <= v && v <= packed[k].hi);
    }
    assert_eq!(packed[4], qkv[4]);
    assert_eq!(packed[5], qkv[5]);
}

#[test]
fn reachable_decoder_complete_vocabulary_wrong_target_ties_and_zero_token() {
    let options = options();
    let model = fixture(&options);
    let negative = certify_reachable_next_token(&model, &[1], 0.0001, 2, &options).unwrap();
    assert!(!negative.proven().unwrap());
    let reversed = certify_reachable_next_token(&model, &[2], 0.0001, 2, &options).unwrap();
    // Nonzero residual shifts do not relabel embeddings; actual a=-1+.75<0.
    assert!(reversed.proven().unwrap());
    let result = certify_reachable_next_token(&model, &[0], 0.0001, 0, &options).unwrap();
    assert_eq!(result.token_ids(), [0]);
    assert_eq!(result.logits().unwrap().len(), 3);
}

#[test]
fn reachable_decoder_refuses_unpaid_foreign_scope_invalid_shape_and_nonfinite_threat() {
    let options = options();
    let model = fixture(&options);
    let mut foreign = options.clone();
    foreign.budget = Some(
        WorkBudget::new(
            options.deadline,
            16 << 20,
            16 << 20,
            1 << 32,
            100,
            Duration::ZERO,
        )
        .unwrap(),
    );
    assert!(certify_reachable_next_token(&model, &[1], 0.01, 0, &foreign).is_err());
    foreign.budget = None;
    assert!(certify_reachable_next_token(&model, &[1], 0.01, 0, &foreign).is_err());
    for (tokens, radius, target) in [
        (&[][..], 0.1, 0),
        (&[3][..], 0.1, 0),
        (&[1][..], f64::NAN, 0),
        (&[1][..], -0.1, 0),
        (&[1][..], 0.1, 3),
    ] {
        assert!(certify_reachable_next_token(&model, tokens, radius, target, &options).is_err());
    }
    let long = vec![1; 33];
    assert!(certify_reachable_next_token(&model, &long, 0.1, 0, &options).is_err());
}

#[test]
fn reachable_decoder_full_source_peak_refusal_is_before_any_new_claim() {
    let options = options();
    let model = fixture(&options);
    let (request, counts, _) = quote(&model, 1).unwrap();
    let arithmetic = counts.request().unwrap();
    let before = model.budget().receipt().unwrap();
    let refused = DecoderOptions {
        max_resident_bytes: model.reserved_bytes()
            + request.persistent_bytes
            + arithmetic.persistent_bytes
            - 1,
        ..options.clone()
    };
    assert!(certify_reachable_next_token(&model, &[1], 0.01, 0, &refused).is_err());
    assert_eq!(model.budget().receipt().unwrap(), before);
    let refused = DecoderOptions {
        max_scratch_bytes: request.scratch_bytes + arithmetic.scratch_bytes - 1,
        ..options
    };
    assert!(certify_reachable_next_token(&model, &[1], 0.01, 0, &refused).is_err());
    assert_eq!(model.budget().receipt().unwrap(), before);
}

#[test]
fn reachable_decoder_preserves_original_earliest_clock_and_last_alias_claims() {
    let mut options = options();
    let model = fixture(&options);
    let original = model.deadline();
    let earlier = Instant::now() + Duration::from_secs(10);
    options.deadline = Some(earlier);
    let baseline = model.budget().receipt().unwrap().live_persistent_bytes;
    let result = certify_reachable_next_token(&model, &[1], 0.001, 0, &options).unwrap();
    assert!(result.deadline() <= earlier && result.deadline() <= original);
    let alias = result.clone();
    let after = model.budget().receipt().unwrap();
    drop(result);
    assert_eq!(
        model.budget().receipt().unwrap().live_persistent_bytes,
        after.live_persistent_bytes
    );
    drop(alias);
    assert_eq!(
        model.budget().receipt().unwrap().live_persistent_bytes,
        baseline
    );
    options.deadline = Some(Instant::now());
    assert!(certify_reachable_next_token(&model, &[1], 0.001, 0, &options).is_err());
}

#[test]
fn reachable_decoder_actual_same_ledger_cancellation_releases_interpreter_outputs() {
    let cancel = CancellationToken::default();
    let memory = MemoryAuthority::new(32 << 20);
    let mut options = options();
    options.budget = Some(
        WorkBudget::new_in(
            &memory,
            &cancel,
            WorkLimits {
                deadline: options.deadline,
                scratch_bytes: 16 << 20,
                persistent_bytes: 16 << 20,
                max_work: 1 << 32,
                max_proposals: 100,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap(),
    );
    let model = fixture(&options);
    let result = certify_reachable_next_token(&model, &[1], 0.001, 0, &options).unwrap();
    cancel.cancel();
    assert!(result.proven().is_err());
    assert!(certify_reachable_next_token(&model, &[1], 0.001, 0, &options).is_err());
    drop(result);
    drop(model);
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_persistent_bytes,
        0
    );
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_scratch_bytes,
        0
    );
}

#[test]
fn reachable_decoder_mid_original_affine_cancellation_publishes_no_partial_state() {
    let cancel = CancellationToken::default();
    let memory = MemoryAuthority::new(32 << 20);
    let mut options = options();
    options.budget = Some(
        WorkBudget::new_in(
            &memory,
            &cancel,
            WorkLimits {
                deadline: options.deadline,
                scratch_bytes: 16 << 20,
                persistent_bytes: 16 << 20,
                max_work: 1 << 32,
                max_proposals: 100,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap(),
    );
    let model = fixture(&options);
    let baseline = model.budget().receipt().unwrap().live_persistent_bytes;
    CANCEL_ROW.with(|hook| *hook.borrow_mut() = Some((3, cancel.clone())));
    assert!(certify_reachable_next_token(&model, &[1], 0.001, 0, &options).is_err());
    assert!(
        CANCEL_ROW.with(|hook| hook.borrow().is_none()),
        "must cancel during genuine original QKV rows"
    );
    assert_eq!(
        model.budget().receipt().unwrap().live_persistent_bytes,
        baseline
    );
    assert_eq!(model.budget().receipt().unwrap().live_scratch_bytes, 0);
    drop(model);
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_persistent_bytes,
        0
    );
}

#[test]
fn reachable_decoder_missing_or_changed_original_pretrained_assets_refuse() {
    let options = options();
    let path = std::env::temp_dir().join(format!("ny-missing-pythia-{}", std::process::id()));
    assert!(super::super::pretrained::load_pretrained_pythia14m(&path, &options).is_err());
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_persistent_bytes,
        0
    );
}

#[test]
fn reachable_decoder_same_held_checkpoint_digest_binds_original_bits_and_refuses_wrong_pin() {
    use sha2::{Digest as _, Sha256};
    let options = options();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.safetensors");
    let mut header = br#"{"original":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&(-0.0f32).to_bits().to_le_bytes());
    bytes.extend_from_slice(&f32::from_bits(1).to_bits().to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let expected: [u8; 32] = Sha256::digest(&bytes).into();
    let owner =
        super::super::load::load_pinned_decoder_safetensors(&path, expected, &options).unwrap();
    assert_eq!(owner.pinned_file_digest, Some(expected));
    let loaded = owner.weights.get("original").unwrap().as_slice().unwrap();
    assert_eq!(loaded[0].to_bits(), 0x80000000);
    assert_eq!(loaded[1].to_bits(), 1);
    drop(owner);
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_persistent_bytes,
        0
    );
    let mut wrong = expected;
    wrong[0] ^= 1;
    assert!(super::super::load::load_pinned_decoder_safetensors(&path, wrong, &options).is_err());
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_persistent_bytes,
        0
    );
    assert_eq!(
        options
            .budget
            .as_ref()
            .unwrap()
            .receipt()
            .unwrap()
            .live_scratch_bytes,
        0
    );
}

#[test]
fn reachable_decoder_authentic_checkpoint_hash_work_cap_is_preallocated() {
    let options = options();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.safetensors");
    let bytes = vec![0u8; 32];
    std::fs::write(&path, &bytes).unwrap();
    let strict = DecoderOptions {
        max_terms: 32 + 16384,
        ..options.clone()
    };
    let before = options.budget.as_ref().unwrap().receipt().unwrap();
    assert!(super::super::load::load_pinned_decoder_safetensors(&path, [0; 32], &strict).is_err());
    assert_eq!(options.budget.as_ref().unwrap().receipt().unwrap(), before);
}

#[cfg(feature = "external-decoders")]
#[test]
fn reachable_decoder_actual_pinned_pythia_all_six_blocks_positive_radius_h1() {
    let at = Instant::now() + Duration::from_secs(300);
    let options = DecoderOptions {
        deadline: Some(at),
        budget: Some(
            WorkBudget::new(Some(at), 4 << 30, 8 << 30, 1 << 34, 1024, Duration::ZERO).unwrap(),
        ),
        ..DecoderOptions::default()
    };
    let model=super::super::pretrained::load_pretrained_pythia14m("<home>/ny-model-cache/high-impact/EleutherAI--pythia-14m/cf967c0a9a04383db6f7b1108d86b2962634b4ac",&options).unwrap();
    // Original pinned prompt IDs; actual embedding radius is POSITIVE. Target
    //253 is only the requested objective; no supplied hidden/logit authority.
    let proof =
        certify_reachable_next_token(&model, &[510, 5347, 273, 6181, 310], 1e-12, 253, &options)
            .unwrap();
    assert!(
        proof.proven().unwrap(),
        "full original H1 margin {:?}",
        proof.margin().unwrap()
    );
    let r = proof.receipt();
    assert_eq!(
        (
            r.complete_blocks,
            r.layer_normalizations,
            r.causal_attention_heads,
            r.vocabulary_rows,
            r.horizon
        ),
        (6, 13, 24, 50304, 1)
    );
    assert!(proof.embedding_radius() > 0.0);
    assert_eq!(
        proof.model().manifest().revision,
        super::super::pretrained::PYTHIA14M_REVISION
    );
    assert_eq!(proof.final_hidden().unwrap().len(), 5 * 128);
    assert_eq!(proof.logits().unwrap().len(), 50304);
}
