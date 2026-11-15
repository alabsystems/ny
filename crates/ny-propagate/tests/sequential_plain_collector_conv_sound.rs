// Regression for #seq-plain-collector-sound: the PLAIN sequential IBP collectors
// (`Network::collect_ibp_bounds` and `Network::collect_ibp_bounds_with_deadline`) are
// the base of every sequential verdict path that is not `collect_ibp_bounds_sound`:
// the input-split per-domain layer bounds (`beta_crown/engine/input_split.rs`), the
// sequential root IBP fast path and root layer bounds (`beta_crown/engine/bounds.rs`,
// `verify_phases/initial.rs`), the CROWN-IBP base (`network/ibp/crown_ibp.rs`) and
// the CROWN-fast / batched intermediate bounds (`core/sequential/crown/{fast,batched}.rs`).
// Until this repair they called the plain conv arms — round-to-nearest f32 window
// sums with no directed rounding and no Higham term — whose box can sit INSIDE the true
// output under cancellation. `conv_family_sound_ibp_dispatch.rs` pins the SOUND
// dispatchers; this file pins the plain entries and the paths that read them.
//
// Fixture A (order-independent): 1x1 Conv2d over in_c=2 with W = [2^24, 1], bias -2^24,
// input x0 = 1 (point), x1 in [0.5, 1]. Any f32 summation order of the two products
// gives fl(2^24 + x1) = 2^24 for x1 in [0.5, 1] (below the half-ULP 1.0 at 2^24, tie at
// exactly 1.0 rounds to even), so the plain conv box is exactly [0, 0] while the true
// pre-activation is [0.5, 1]. The follow-on Linear (w = 1, b = -0.5) then reads
// [-0.5, -0.5] from the plain box: the second ReLU is classified DEAD (u <= 0) while the
// true pre-activation is [0, 0.5] — a false-stable neuron, the exact mechanism that
// turns into a false VERIFIED domain.
//
// Fixture B: the K=3 cancellation window from `conv_family_sound_ibp_dispatch.rs`,
// W = [2^24, -1, -2^24], point input [1,1,1], true output -1, plain output exactly 0,
// for every conv variant (Conv1d / Conv2d / ConvTranspose1d / ConvTranspose2d).
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_propagate::{
    layers::{
        Conv1dLayer, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer, FlattenLayer,
        LinearLayer, ReLULayer,
    },
    BoundPropagation, Layer, Network,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

const TWO24: f32 = 16_777_216.0; // 2^24

// ---------------------------------------------------------------------------
// Fixture A: Conv2d -> ReLU -> Flatten -> Linear -> ReLU with the [2^24, 1] kernel.
// ---------------------------------------------------------------------------

/// Layer indices in the fixture-A network.
const CONV: usize = 0;
const LINEAR: usize = 3;

/// True pre-activation interval of the Linear layer (= second ReLU input).
const LINEAR_TRUE_LOWER: f32 = 0.0;
const LINEAR_TRUE_UPPER: f32 = 0.5;

fn fixture_a_network() -> Network {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1]), vec![TWO24, 1.0]).unwrap();
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(arr1(&[-TWO24])), (1, 1), (0, 0), 1, 1).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Conv2d(conv));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[-0.5_f32]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer));
    net
}

/// x0 = 1 (point), x1 in [0.5, 1]; layout (in_c=2, H=1, W=1).
fn fixture_a_input() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![1.0, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap()
}

fn first_lower(b: &BoundedTensor) -> f32 {
    b.lower().iter().copied().next().unwrap()
}
fn first_upper(b: &BoundedTensor) -> f32 {
    b.upper().iter().copied().next().unwrap()
}

fn assert_linear_preactivation_not_dead(bounds: &[BoundedTensor], label: &str) {
    assert_eq!(
        bounds.len(),
        5,
        "{label}: five layers => five collected bounds"
    );
    let conv_l = first_lower(&bounds[CONV]);
    let conv_u = first_upper(&bounds[CONV]);
    assert!(
        conv_l <= 0.5 && conv_u >= 1.0,
        "{label}: conv box [{conv_l}, {conv_u}] must enclose the true [0.5, 1]; \
         [0, 0] is the plain-arm cancellation signature"
    );
    let lin_l = first_lower(&bounds[LINEAR]);
    let lin_u = first_upper(&bounds[LINEAR]);
    assert!(
        lin_l <= LINEAR_TRUE_LOWER && lin_u >= LINEAR_TRUE_UPPER,
        "{label}: Linear pre-activation [{lin_l}, {lin_u}] must enclose the true \
         [{LINEAR_TRUE_LOWER}, {LINEAR_TRUE_UPPER}]"
    );
    assert!(
        lin_u > 0.0,
        "{label}: second ReLU read as DEAD (u = {lin_u} <= 0) from an inward conv box; \
         the true pre-activation reaches {LINEAR_TRUE_UPPER}"
    );
}

/// The plain per-layer arms really do collapse this fixture: without this control the
/// enclosure tests above would pass vacuously on a fixture that never cancelled.
#[test]
fn plain_layer_arms_classify_the_second_relu_dead_on_fixture_a() {
    let net = fixture_a_network();
    let mut current = fixture_a_input();
    let mut per_layer = Vec::new();
    for layer in net.layers().iter() {
        current = layer.propagate_ibp(&current).unwrap();
        per_layer.push(current.clone());
    }
    assert_eq!(first_lower(&per_layer[CONV]), 0.0, "plain conv lower");
    assert_eq!(first_upper(&per_layer[CONV]), 0.0, "plain conv upper");
    let lin_u = first_upper(&per_layer[LINEAR]);
    assert!(
        lin_u <= 0.0,
        "plain chain should classify the second ReLU dead (got u = {lin_u})"
    );
    // The whole-network plain forward is the same inward box (fallback paths return it).
    let out = net.propagate_ibp(&fixture_a_input()).unwrap();
    assert_eq!(first_upper(&out), 0.0, "plain network output upper");
}

#[test]
fn collect_ibp_bounds_encloses_fixture_a() {
    let net = fixture_a_network();
    let bounds = net.collect_ibp_bounds(&fixture_a_input()).unwrap();
    assert_linear_preactivation_not_dead(&bounds, "Network::collect_ibp_bounds");
}

#[test]
fn collect_ibp_bounds_with_none_deadline_encloses_fixture_a() {
    let net = fixture_a_network();
    let bounds = net
        .collect_ibp_bounds_with_deadline(&fixture_a_input(), None)
        .unwrap();
    assert_linear_preactivation_not_dead(&bounds, "collect_ibp_bounds_with_deadline(None)");
}

#[test]
fn collect_ibp_bounds_with_far_deadline_encloses_fixture_a() {
    let net = fixture_a_network();
    let far = Instant::now() + Duration::from_secs(3600);
    let bounds = net
        .collect_ibp_bounds_with_deadline(&fixture_a_input(), Some(far))
        .unwrap();
    assert_linear_preactivation_not_dead(&bounds, "collect_ibp_bounds_with_deadline(Some)");
}

/// CROWN-fast (`core/sequential/crown/fast.rs:100`) builds its ReLU relaxations from the
/// plain collector and intersects its concretized output with the collector's last
/// box. With the dead second ReLU the output was exactly [0, 0]; the true output is
/// [0, 0.5].
#[test]
fn crown_fast_output_encloses_fixture_a() {
    let net = fixture_a_network();
    let out = net.propagate_crown_fast(&fixture_a_input()).unwrap();
    let u = first_upper(&out);
    let l = first_lower(&out);
    assert!(
        l <= 0.0 && u >= 0.5,
        "CROWN-fast output [{l}, {u}] must enclose the true [0, 0.5]; u == 0 is the \
         dead-ReLU signature from an inward conv box"
    );
}

/// `propagate_crown_with_linear` (`fast.rs:190`) is the production entry used by the
/// input-split clip and joint-margin closers (`relaxed_clip.rs`, `joint_margin.rs`).
#[test]
fn crown_with_linear_output_encloses_fixture_a() {
    let net = fixture_a_network();
    let (out, _linear) = net.propagate_crown_with_linear(&fixture_a_input()).unwrap();
    let u = first_upper(&out);
    let l = first_lower(&out);
    assert!(
        l <= 0.0 && u >= 0.5,
        "CROWN-with-linear output [{l}, {u}] must enclose the true [0, 0.5]"
    );
}

/// Batched CROWN (`core/sequential/crown/batched.rs:86`) collects the same plain bounds.
#[test]
fn crown_batched_output_encloses_fixture_a() {
    let net = fixture_a_network();
    let out = net.propagate_crown_batched(&fixture_a_input()).unwrap();
    let u = first_upper(&out);
    let l = first_lower(&out);
    assert!(
        l <= 0.0 && u >= 0.5,
        "batched CROWN output [{l}, {u}] must enclose the true [0, 0.5]"
    );
}

/// `propagate_crown` (the CROWN-IBP base at `network/ibp/crown_ibp.rs:159`) rests on
/// `collect_ibp_bounds_with_deadline` too.
#[test]
fn crown_output_encloses_fixture_a() {
    let net = fixture_a_network();
    let out = net.propagate_crown(&fixture_a_input()).unwrap();
    let u = first_upper(&out);
    let l = first_lower(&out);
    assert!(
        l <= 0.0 && u >= 0.5,
        "CROWN output [{l}, {u}] must enclose the true [0, 0.5]"
    );
}

// ---------------------------------------------------------------------------
// Fixture B: every conv variant, K=3 cancellation window (true -1, plain 0).
// ---------------------------------------------------------------------------

const W3: [f32; 3] = [TWO24, -1.0, -TWO24];
const TRUE_OUT_B: f64 = -1.0;

fn zero_bias() -> Option<Array1<f32>> {
    Some(arr1(&[0.0_f32]))
}

fn point_3ch(shape: &[usize]) -> BoundedTensor {
    let pt = ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32; 3]).unwrap();
    BoundedTensor::new(pt.clone(), pt).unwrap()
}

fn conv_cases() -> Vec<(&'static str, Layer, BoundedTensor)> {
    vec![
        (
            "Conv1d",
            Layer::Conv1d(
                Conv1dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), W3.to_vec()).unwrap(),
                    zero_bias(),
                    1,
                    0,
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1]),
        ),
        (
            "Conv2d",
            Layer::Conv2d(
                Conv2dLayer::with_input_shape(
                    ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1]), W3.to_vec()).unwrap(),
                    zero_bias(),
                    (1, 1),
                    (0, 0),
                    1,
                    1,
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1, 1]),
        ),
        (
            "ConvTranspose1d",
            Layer::ConvTranspose1d(
                ConvTranspose1dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), W3.to_vec()).unwrap(),
                    zero_bias(),
                    1,
                    0,
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1]),
        ),
        (
            "ConvTranspose2d",
            Layer::ConvTranspose2d(
                ConvTranspose2dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), W3.to_vec()).unwrap(),
                    zero_bias(),
                    (1, 1),
                    (0, 0),
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1, 1]),
        ),
    ]
}

fn single_layer_network(layer: Layer) -> Network {
    let mut net = Network::new();
    net.add_layer(layer);
    net
}

fn assert_encloses_b(lower: f32, label: &str) {
    assert!(
        (lower as f64) <= TRUE_OUT_B + 1e-4,
        "{label}: collected lower {lower} must enclose the true output {TRUE_OUT_B}; \
         a value near 0 means the conv took the plain forward"
    );
}

#[test]
fn plain_collect_encloses_for_every_conv_variant() {
    for (label, layer, input) in conv_cases() {
        let net = single_layer_network(layer);
        let bounds = net.collect_ibp_bounds(&input).unwrap();
        assert_eq!(bounds.len(), 1, "{label}: one layer => one collected bound");
        assert_encloses_b(
            first_lower(&bounds[0]),
            &format!("Network::collect_ibp_bounds {label}"),
        );
    }
}

#[test]
fn plain_collect_with_deadline_encloses_for_every_conv_variant() {
    let far = Instant::now() + Duration::from_secs(3600);
    for (label, layer, input) in conv_cases() {
        let net = single_layer_network(layer);
        for (dl_label, dl) in [("None", None), ("Some", Some(far))] {
            let bounds = net.collect_ibp_bounds_with_deadline(&input, dl).unwrap();
            assert_eq!(bounds.len(), 1, "{label}: one layer => one collected bound");
            assert_encloses_b(
                first_lower(&bounds[0]),
                &format!("Network::collect_ibp_bounds_with_deadline({dl_label}) {label}"),
            );
        }
    }
}

/// Linear-only networks must be BYTE-IDENTICAL between the plain collector and the
/// plain per-layer arms: the repair touches the conv-family / AveragePool arms only, so
/// no MLP bank moves.
#[test]
fn plain_collect_is_byte_identical_to_plain_layer_arms_on_linear_only_nets() {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[TWO24, 1.0], [0.3, -0.7]]),
            Some(arr1(&[-TWO24, 0.25])),
        )
        .unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0]]), Some(arr1(&[-0.5]))).unwrap(),
    ));
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 0.5]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let collected = net.collect_ibp_bounds(&input).unwrap();
    let with_deadline = net
        .collect_ibp_bounds_with_deadline(&input, Some(Instant::now() + Duration::from_secs(3600)))
        .unwrap();
    let mut current = input;
    for (i, layer) in net.layers().iter().enumerate() {
        current = layer.propagate_ibp(&current).unwrap();
        for (label, got) in [
            ("collect", &collected[i]),
            ("collect_with_deadline", &with_deadline[i]),
        ] {
            assert_eq!(
                got.lower().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                current
                    .lower()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{label}: layer {i} lower must be bit-identical to the plain Linear arm"
            );
            assert_eq!(
                got.upper().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                current
                    .upper()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{label}: layer {i} upper must be bit-identical to the plain Linear arm"
            );
        }
    }
}
