// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Explicit external-vnncomp original-fixture probe. No historical status is an oracle and no
//! JSON receipt mints authority. The checked private OriginalReplay token is
//! inspected before any positive observation; LP remains explicitly unpriced.
use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

const REQUEST_SCHEMA: &str = "ny.original-safenlp-hull-resource-request.v1";
const MODEL_MEDICAL: &str = "5c6b9f0d3e78db73f260a1d3a76bd36543030a1f3dfe54b103fcbf4ec08b762b";
const MODEL_ROBOT: &str = "a9857f1f8f443ec03f4ede912146bf0aa3ba20d2ecb84301a8e631524576d5ff";
const ROWS: &[(u16, usize, &str, &str, usize)] = &[
    (
        2025,
        1,
        "vnnlib/medical/hyperrectangle_418.vnnlib",
        "af9def297e5cc0a38940efc5dfef2a193395e1befcba1ae6252aadcc2e36eb60",
        3306,
    ),
    (
        2025,
        2,
        "vnnlib/ruarobot/hyperrectangle_992.vnnlib",
        "e58db85b37e745f79a59cab9a170198def5d775810fdf7a2755e54a5aa0b9772",
        3319,
    ),
    (
        2025,
        3,
        "vnnlib/ruarobot/hyperrectangle_132.vnnlib",
        "792bafd8355df1f52d3c71727a47ffbde9d5cec408d92bee964740ce200054f0",
        3320,
    ),
    (
        2025,
        4,
        "vnnlib/ruarobot/hyperrectangle_2674.vnnlib",
        "505cc544866749e7478cbd965254c6d1db945bbbeb5434bdf6dcfe6ac9a24144",
        3310,
    ),
    (
        2025,
        5,
        "vnnlib/medical/hyperrectangle_2839.vnnlib",
        "f11b08e0e591198965a425f4e18f8f4d368489347a61739f3aa95e317256a9bb",
        3295,
    ),
    (
        2025,
        8,
        "vnnlib/ruarobot/hyperrectangle_3558.vnnlib",
        "f4d77a33f45be35ac0f68065ded8753ebc8971260f87f8796ea1698d20306a33",
        3316,
    ),
    (
        2025,
        9,
        "vnnlib/medical/hyperrectangle_2884.vnnlib",
        "e03b422f8cbfc47407c87fac0fd4ea8d501be90fa9f225ca8c4acee0d52af7aa",
        3314,
    ),
    (
        2025,
        10,
        "vnnlib/ruarobot/hyperrectangle_6394.vnnlib",
        "375e8854909f3f1491d7ac7f2398117db9d27247ff52f940de305b058e50559d",
        3305,
    ),
    (
        2025,
        28,
        "vnnlib/medical/hyperrectangle_2660.vnnlib",
        "a2cc845f7ccd64db8263608b497bb05f5313a2364fe88c7ed8d2676a05bc060b",
        3303,
    ),
    (
        2025,
        33,
        "vnnlib/ruarobot/hyperrectangle_859.vnnlib",
        "e4fedf3b3d7cbd485b4870e528be6cffd824378e9dc62e051ba6d7dd50aee460",
        3310,
    ),
    (
        2025,
        76,
        "vnnlib/medical/hyperrectangle_2672.vnnlib",
        "852b9e069ec6dec0ab8fca8d75c220fcc12f711949d23488fbeee6de9f9f6815",
        3306,
    ),
    (
        2025,
        94,
        "vnnlib/medical/hyperrectangle_1228.vnnlib",
        "8cbed8bc7e5443b3e459c2714180355e1253f3d03fefa35ab98018842437ed83",
        3301,
    ),
    (
        2026,
        1,
        "vnnlib/ruarobot/hyperrectangle_257.vnnlib",
        "b215efcb630e7bb9f89bca5261f5072b2450f6289a5179d6faf1ec7a6fd5a722",
        3307,
    ),
    (
        2026,
        2,
        "vnnlib/medical/hyperrectangle_355.vnnlib",
        "e6285ee94f332c9cea0ed3cbeb965943572c264587fb074a86345914a64d8432",
        3299,
    ),
    (
        2026,
        3,
        "vnnlib/medical/hyperrectangle_557.vnnlib",
        "90e1f13e8e4805dbf18739259d9d75e3ab47a3b8a1127fc7aec2e8ba6707bd4f",
        3314,
    ),
    (
        2026,
        4,
        "vnnlib/medical/hyperrectangle_265.vnnlib",
        "da42a108174b30cb240ba363579c6b3a626e95bd4ac079f1b85e9e76bf4af9ab",
        3315,
    ),
    (
        2026,
        5,
        "vnnlib/ruarobot/hyperrectangle_6763.vnnlib",
        "3d0bcfc92ce354bdce5f17cebeccfaebce19e913d1061c44a1d6c482ffa5d5e9",
        3303,
    ),
    (
        2026,
        17,
        "vnnlib/ruarobot/hyperrectangle_4617.vnnlib",
        "e3db5d134055a443214678498727877d0e48bdc692ad0061d7fe3137ccbd444b",
        3299,
    ),
    (
        2026,
        20,
        "vnnlib/medical/hyperrectangle_2532.vnnlib",
        "73c88d6d31d1d656a419a9eaccfce67c2006f9aa0e0ef4547f9f974be66f657e",
        3312,
    ),
    (
        2026,
        29,
        "vnnlib/ruarobot/hyperrectangle_2999.vnnlib",
        "dec21ae7e91135954c6d6bdc592c3b9ee2dff80e3ed1799674e60ab907b1d1fb",
        3312,
    ),
    (
        2026,
        31,
        "vnnlib/ruarobot/hyperrectangle_4779.vnnlib",
        "c62b36372314d113406c4b5dcb854e253d087351c99fda9ee7c6ec6c1c0b3f26",
        3298,
    ),
    (
        2026,
        38,
        "vnnlib/ruarobot/hyperrectangle_3400.vnnlib",
        "d84cb348b1325f4c86091ed35377d6971087719f5bc81420251de9165bac6422",
        3321,
    ),
    (
        2026,
        43,
        "vnnlib/medical/hyperrectangle_963.vnnlib",
        "e0efca54735ac93b7d60e962abf27b1e6caf57ed0a0b3f6dee310670145eca02",
        3306,
    ),
    (
        2026,
        80,
        "vnnlib/medical/hyperrectangle_2303.vnnlib",
        "f3bd428ad76daaf1d11dc8b442e16275ac15da0fe5b7b1219d7aa73e76843c26",
        3289,
    ),
    (
        2026,
        146,
        "vnnlib/ruarobot/hyperrectangle_5249.vnnlib",
        "e3669c9392363549da4206aa078f25106c2d21ba4e131472c57669ae392e3691",
        3313,
    ),
    (
        2026,
        578,
        "vnnlib/ruarobot/hyperrectangle_455.vnnlib",
        "b0d573e51f8bc011fa7d456aed5b87398aa5e97ece7bdbb09a92ac4fc05f9a65",
        3317,
    ),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalRequest {
    schema: String,
    year: u16,
    source_commit: String,
    source_index: usize,
    occurrence: usize,
    source_csv_sha256: String,
    model: String,
    model_sha256: String,
    property: String,
    property_sha256: String,
    timeout_seconds: u64,
    preset_path: String,
    preset_sha256: String,
    receipt_path: String,
}

fn read_original(
    path: &Path,
    max_bytes: usize,
    expected: Option<&str>,
    lease: &WorkLease,
) -> Vec<u8> {
    lease
        .checkpoint()
        .expect("original source admission/deadline");
    let before =
        fs::metadata(path).expect("pinned original unavailable: refusal, never skipped pass");
    assert!(
        before.is_file() && before.len() <= max_bytes as u64,
        "original size/profile refusal"
    );
    let mut file = fs::File::open(path).expect("original read");
    let mut result = Vec::with_capacity(before.len() as usize);
    let mut buffer = [0_u8; 1024];
    let mut hash = Sha256::new();
    loop {
        lease
            .checkpoint()
            .expect("original source scan cancelled/deadline");
        let count = file.read(&mut buffer).expect("original source read");
        if count == 0 {
            break;
        }
        assert!(
            result.len() + count <= max_bytes,
            "original grew beyond prepaid profile"
        );
        hash.update(&buffer[..count]);
        result.extend_from_slice(&buffer[..count]);
    }
    let after = fs::metadata(path).expect("original post-read metadata");
    assert_eq!(before.len(), after.len());
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    assert_eq!(result.len() as u64, before.len());
    if let Some(expected) = expected {
        assert_eq!(
            format!("{:x}", hash.finalize()),
            expected,
            "original digest mismatch"
        );
    }
    lease
        .checkpoint()
        .expect("original source final checkpoint");
    result
}

fn authenticate_model(model: &ny_onnx::OnnxModel, lease: &WorkLease) {
    lease.checkpoint().unwrap();
    let original = &model.network;
    assert_eq!(original.inputs.len(), 1, "one original input required");
    assert_eq!(original.outputs.len(), 1, "one original output required");
    // The SHA-authenticated original authors declare a symbolic input/output
    // batch. The ordinary loader resolves only runtime input axis0 to the
    // verification batch of one; output metadata remains authored/symbolic.
    // The original 30-input/2-output occurrence selects exactly one vector.
    // This probe makes no claim about other symbolic batch sizes.
    assert_eq!(original.inputs[0].shape, vec![1, 30]);
    assert_eq!(original.outputs[0].shape, vec![-1, 2]);
    assert_eq!(original.inputs[0].dtype, ny_onnx::DataType::Float32);
    assert_eq!(original.outputs[0].dtype, ny_onnx::DataType::Float32);
    assert_eq!(original.layers.len(), 5);
    let kinds = [
        ny_core::LayerType::MatMul,
        ny_core::LayerType::Add,
        ny_core::LayerType::ReLU,
        ny_core::LayerType::MatMul,
        ny_core::LayerType::Add,
    ];
    let mut source = original.inputs[0].name.as_str();
    for (index, (layer, kind)) in original.layers.iter().zip(kinds).enumerate() {
        lease.checkpoint().unwrap();
        assert_eq!(layer.layer_type, kind, "original authored operator changed");
        assert!(
            layer.attributes.is_empty(),
            "no scale/transpose/custom profile"
        );
        assert_eq!(layer.inputs.len(), if index == 2 { 1 } else { 2 });
        assert_eq!(layer.outputs.len(), 1);
        assert_eq!(
            layer.inputs[0], source,
            "original single live edge must be preserved"
        );
        if index != 2 {
            let parameter = model
                .weights
                .get(&layer.inputs[1])
                .expect("original constant operand");
            let expected: &[usize] = match index {
                0 => &[30, 128],
                1 => &[128],
                3 => &[128, 2],
                4 => &[2],
                _ => unreachable!(),
            };
            assert_eq!(parameter.shape(), expected);
            assert!(
                parameter.as_slice().is_some(),
                "original row-major parameter profile"
            );
        }
        source = &layer.outputs[0];
    }
    assert_eq!(source, original.outputs[0].name);
    lease.checkpoint().unwrap();
}

fn authenticate_lowered(model: &ny_onnx::OnnxModel, network: &Network, lease: &WorkLease) {
    let [Layer::Linear(first), Layer::AddConstant(b1), Layer::ReLU(_), Layer::Linear(last), Layer::AddConstant(b2)] =
        network.layers()
    else {
        panic!("ordinary original five-node lowering required");
    };
    assert!(first.bias().is_none() && last.bias().is_none());
    let layers = &model.network.layers;
    let first_source = model
        .weights
        .get(&layers[0].inputs[1])
        .unwrap()
        .as_slice()
        .unwrap();
    let last_source = model
        .weights
        .get(&layers[3].inputs[1])
        .unwrap()
        .as_slice()
        .unwrap();
    assert_eq!(first.weight().dim(), (128, 30));
    assert_eq!(last.weight().dim(), (2, 128));
    for h in 0..128 {
        lease.checkpoint().unwrap();
        for i in 0..30 {
            assert_eq!(
                first.weight()[[h, i]].to_bits(),
                first_source[i * 128 + h].to_bits(),
                "original W1 reindex bits"
            );
        }
        for o in 0..2 {
            assert_eq!(
                last.weight()[[o, h]].to_bits(),
                last_source[h * 2 + o].to_bits(),
                "original W2 reindex bits"
            );
        }
    }
    for (bias, index) in [(b1, 1_usize), (b2, 4_usize)] {
        let original = model.weights.get(&layers[index].inputs[1]).unwrap();
        assert_eq!(bias.constant().shape(), original.shape());
        for (a, b) in bias.constant().iter().zip(original) {
            lease.checkpoint().unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "authored bias provenance");
        }
    }
    lease.checkpoint().unwrap();
}

#[test]
fn hull_resource_cli_original_safenlp_strict20s() {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(20);
    // This single authority precedes all source IO. The fixed loader envelope
    // covers ONLY the two SHA-pinned18KiB originals/≤8KiB properties, their
    // protobuf/WeightStore/ordinary Linear owners,16KiB report and small views.
    // Static authored shapes only: ORT's native inference workspace is not
    // covered by this envelope and is deliberately not entered by this probe.
    // It is capacity admission, not whole-process RSS or an AY workspace quota.
    let memory = MemoryAuthority::new(32 << 20);
    let cancel = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(deadline),
            scratch_bytes: 16 << 20,
            persistent_bytes: 32 << 20,
            max_work: 1 << 30,
            max_proposals: 1,
            certificate_reserve: Duration::from_millis(100),
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let mut loader = ledger
        .lease_for(
            LeaseClass::Optional,
            WorkRequest {
                scratch_bytes: 0,
                persistent_bytes: 16 << 20,
                work: 64 << 20,
                proposals: 0,
            },
        )
        .expect("finite source/model loader admission");
    loader.mark_materialized().unwrap();
    let inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
    let request_path = inputs
        .get(&ny_levers::decls::diagnostics::HULL_RESOURCE_ORIGINAL_REQUEST)
        .expect("closed original request required; not a skipped gate");
    assert!(request_path.len() <= 4096, "bounded request path");
    let request_bytes = read_original(Path::new(&request_path), 8192, None, &loader);
    let request_sha = format!("{:x}", Sha256::digest(&request_bytes));
    let request: OriginalRequest =
        serde_json::from_slice(&request_bytes).expect("closed request fields");
    assert_eq!(request.schema, REQUEST_SCHEMA);
    assert_eq!(request.timeout_seconds, 20);
    // Direct VNN-COMP currently selects this same newest-year preset for both
    // years. This immutable copy changes ONLY the typed hull mode to Root.
    let preset_path = Path::new("<home>/ny-experiments/high-impact/2026-10-01/safenlp-hull-capsule-v1/root/vnncomp26/safenlp_2024.yaml");
    let preset_sha = "5c089111daa8e095fc4a666d16e95684c3ffe93e86f464d285eca51bdb050604";
    assert_eq!(Path::new(&request.preset_path), preset_path);
    assert_eq!(request.preset_sha256, preset_sha);
    let preset_bytes = read_original(preset_path, 16 << 10, Some(preset_sha), &loader);
    let preset: crate::preset::PresetConfig =
        serde_yaml::from_slice(&preset_bytes).expect("closed current loaded preset");
    assert_eq!(
        preset.bab.single_hidden_hull,
        Some(SingleHiddenHullMode::Root)
    );
    loader
        .checkpoint()
        .expect("after typed current preset parse");
    assert_eq!(
        request.occurrence, 1,
        "these exact original occurrences each occur once"
    );
    let (commit, csv_sha) = match request.year {
        2025 => (
            "8b7b811b78ce6a329dc96f04ae6652da3c245948",
            "2c6a1a33ef085eabb0431595214aaedd5357e52fbbf86b564418f5accec0b15d",
        ),
        2026 => (
            "303f2327a8e8411efc4457da03d2016ce42733cf",
            "e225fb8ffe11e39457894e75f2a053de446943510f821b12a4501a191d37a59d",
        ),
        _ => panic!("pinned original year required"),
    };
    assert_eq!(request.source_commit, commit);
    assert_eq!(request.source_csv_sha256, csv_sha);
    let row = ROWS
        .iter()
        .find(|r| r.0 == request.year && r.1 == request.source_index)
        .expect("exact selected official occurrence");
    assert_eq!(request.property, row.2);
    assert_eq!(request.property_sha256, row.3);
    let medical = request.property.starts_with("vnnlib/medical/");
    let model_key = if medical {
        "onnx/medical/perturbations_0.onnx"
    } else {
        "onnx/ruarobot/perturbations_0.onnx"
    };
    let model_sha = if medical { MODEL_MEDICAL } else { MODEL_ROBOT };
    let model_bytes_len = if medical { 18443 } else { 18423 };
    assert_eq!(request.model, model_key);
    assert_eq!(request.model_sha256, model_sha);
    let fixture = std::path::PathBuf::from(format!(
        "<home>/ny-model-cache/high-impact/vnncomp{}-safenlp/{commit}",
        request.year
    ));
    let model_path = fixture.join(model_key);
    let property_path = fixture.join(row.2);
    let model_bytes = read_original(&model_path, model_bytes_len, Some(model_sha), &loader);
    assert_eq!(model_bytes.len(), model_bytes_len);
    let property_bytes = read_original(&property_path, row.4, Some(row.3), &loader);
    assert_eq!(property_bytes.len(), row.4);
    let load_config = crate::preset::build_onnx_load_config(&preset)
        .expect("current typed original loader configuration")
        .with_shape_inference_policy(ny_onnx::ShapeInferencePolicy::Skip);
    let model = ny_onnx::load_onnx_bytes_with_config(
        "pinned-original-safenlp-hull",
        &model_bytes,
        &load_config,
    )
    .expect("original ONNX loader");
    loader.checkpoint().expect("after bounded protobuf loading");
    authenticate_model(&model, &loader);
    let spec = parse_vnnlib(std::str::from_utf8(&property_bytes).unwrap())
        .expect("original unchanged property");
    loader.checkpoint().expect("after original property parse");
    assert_eq!((spec.num_inputs, spec.num_outputs), (30, 2));
    let network = model
        .to_propagate_network()
        .expect("ordinary original sequential builder");
    loader
        .checkpoint()
        .expect("after ordinary original builder");
    authenticate_lowered(&model, &network, &loader);
    let (lower, upper) = spec.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 30]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 30]), upper).unwrap(),
    )
    .unwrap();
    loader
        .checkpoint()
        .expect("after original outward input box");
    let context = hull_context_for_budget(
        &ledger,
        hull_root_deadline(
            Instant::now(),
            Some(deadline.saturating_duration_since(Instant::now())),
        ),
    );
    let checked = try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &network,
        &input,
        &spec,
        &context,
        started,
    );
    // These four original/hash-bound rows have independent exact-rational
    // positive triangle controls (receipt2e2a06ac...); this is a mathematical
    // arrival assertion, never a historical bank-status oracle or LP win.
    if matches!(
        (request.year, request.source_index),
        (2025, 5) | (2025, 9) | (2026, 2) | (2026, 3)
    ) {
        assert!(
            checked.is_ok(),
            "independently positive original triangle must reach sealed replay"
        );
    }
    let (observation, reason, scope, lower_bound) = match &checked {
        Ok(result) => {
            assert_eq!(result.proof.origin(), HullProofOrigin::OriginalReplay);
            assert!(result.proof.verified() && result.proof.lower_bound() > 0.0);
            assert!(matches!(
                result.result.result,
                BabVerificationStatus::Verified
            ));
            // The ingress supports exactly one conjunctive relational unsafe
            // row. A disjunction is explicitly refused, never partly proved.
            assert!(!spec.is_disjunction && spec.output_constraint_clauses.len() <= 1);
            (
                "strict_positive_original_replay",
                "checked",
                Some(format!("{:x?}", result.proof.scope())),
                Some(result.proof.lower_bound()),
            )
        }
        Err(reason) => ("open_or_refused", *reason, None, None),
    };
    // The same prepaid loader envelope covers these bounded original rereads.
    assert_eq!(
        read_original(&model_path, model_bytes_len, Some(model_sha), &loader),
        model_bytes
    );
    assert_eq!(
        read_original(&property_path, row.4, Some(row.3), &loader),
        property_bytes
    );
    assert_eq!(
        read_original(Path::new(&request_path), 8192, Some(&request_sha), &loader),
        request_bytes
    );
    assert_eq!(
        read_original(preset_path, 16 << 10, Some(preset_sha), &loader),
        preset_bytes
    );
    let mut export = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: 16384,
                persistent_bytes: 0,
                work: 16384,
                proposals: 0,
            },
        )
        .expect("same-ledger bounded observation export");
    export.mark_materialized().unwrap();
    let receipt = ledger.receipt().unwrap();
    let report = serde_json::json!({"schema":"ny.original-safenlp-hull-resource-observation.v1","authority":"source-qualified in-process original replay observation; no LP/official score/external arithmetic replay claim","request_sha256":request_sha,"year":request.year,"source_commit":commit,"source_csv_sha256":csv_sha,"source_index":request.source_index,"occurrence":request.occurrence,"model_sha256":model_sha,"property_sha256":row.3,"preset_sha256":preset_sha,"preset_component_use":"current typed options except explicit static-only shape inference; explicit Root; no whole PGD/CROWN/fallback pipeline claim","shape_inference_policy":"skip_authored_static_shapes_only","production_equivalent_full_route":false,"timeout_seconds":20,"elapsed_seconds":started.elapsed().as_secs_f64(),"observation":observation,"reason":reason,"private_proof_origin":if lower_bound.is_some(){Some("OriginalReplay")}else{None},"scope":scope,"checked_lower_bound":lower_bound,"full_lp_engaged":false,"unsafe_alternatives_fully_closed":lower_bound.is_some(),"external_independent_arithmetic_replay":false,"work_reserved":receipt.work_reserved,"peak_leased_bytes":receipt.peak_leased_bytes,"live_persistent_bytes":receipt.live_persistent_bytes,"live_scratch_bytes":receipt.live_scratch_bytes,"memory_live_materialized_bytes":memory.receipt().unwrap().live_materialized_bytes});
    let bytes = serde_json::to_vec(&report).unwrap();
    assert!(bytes.len() <= 16384);
    export.checkpoint().expect("before publication");
    context
        .checkpoint(HullLeaseClass::Replay)
        .expect("whole query final deadline");
    export
        .complete()
        .expect("complete before observational publication");
    let path = Path::new(&request.receipt_path);
    assert!(
        path.starts_with("<home>/ny-experiments/high-impact/2026-10-01"),
        "external exclusive observation archive only"
    );
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .expect("never overwrite observation");
    output.write_all(&bytes).unwrap();
    output.write_all(b"\n").unwrap();
    eprintln!("[hull-resource-original] year={} source_index={} observation={} reason={} full_lp_engaged=false request_sha={}",request.year,request.source_index,observation,reason,request_sha);
    // Owners and both completed-query/replay claims stay resident through the
    // final publication. The receipt is telemetry; the token remains private.
    drop(checked);
    drop(input);
    drop(network);
    drop(spec);
    drop(model);
    drop(load_config);
    drop(preset);
    drop(preset_bytes);
    drop(property_bytes);
    drop(model_bytes);
    drop(request_bytes);
    drop(report);
    drop(bytes);
    drop(request);
    drop(output);
    drop(loader);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}
