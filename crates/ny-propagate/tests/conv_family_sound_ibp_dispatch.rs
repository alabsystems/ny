// Regression for #vnncomp-aw-soundness: the SOUND IBP DISPATCHERS must route EVERY
// conv variant to its certified Higham forward, not just Conv2d.
// `conv_family_sound_ibp_encloses.rs` pins the layer methods themselves; this pins the
// WIRING — `GraphNetwork::propagate_ibp_sound`, `Network::propagate_ibp_sound` and
// `Network::collect_ibp_bounds_sound` — where a conv variant that misses its match arm
// falls into the generic `propagate_ibp` + 1-ULP widening.
//
// Cancellation window over K=3 input channels: W = [2^24, -1, -2^24], point input [1,1,1].
//   plain IBP: the W+ pass sums to 2^24, the W- pass to -(2^24 + 1) -> -2^24 (the -1 is
//              below half-ULP at magnitude 2^24); their sum is exactly 0.
//   true:      2^24 - 1 - 2^24 = -1, exact.
// 1 ULP of 0 cannot reach -1, so an unrouted conv returns a "sound" box EXCLUDING the
// true output — and a downstream ReLU `l >= 0 => identity` guard then reads the truly
// unstable neuron as stable-active.
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_propagate::{
    layers::{Conv1dLayer, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer},
    GraphNetwork, Layer, Network,
};
use ny_tensor::BoundedTensor;

const TWO24: f32 = 16_777_216.0; // 2^24
const W: [f32; 3] = [TWO24, -1.0, -TWO24];
const TRUE_OUT: f64 = -1.0;

fn zero_bias() -> Option<Array1<f32>> {
    Some(arr1(&[0.0_f32]))
}

fn point_3ch(shape: &[usize]) -> BoundedTensor {
    let pt = ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32; 3]).unwrap();
    BoundedTensor::new(pt.clone(), pt).unwrap()
}

/// Each conv variant as a single-layer network over the cancellation window, paired
/// with the point input its layout expects.
fn conv_cases() -> Vec<(&'static str, Layer, BoundedTensor)> {
    vec![
        (
            // kernel (out_c=1, in_c=3, kw=1); input (in_c=3, length=1)
            "Conv1d",
            Layer::Conv1d(
                Conv1dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), W.to_vec()).unwrap(),
                    zero_bias(),
                    1,
                    0,
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1]),
        ),
        (
            // kernel (out_c=1, in_c=3, kh=1, kw=1); input (in_c=3, H=1, W=1)
            "Conv2d",
            Layer::Conv2d(
                Conv2dLayer::with_input_shape(
                    ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1]), W.to_vec()).unwrap(),
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
            // transpose kernel (in_c=3, out_c/groups=1, kw=1); input (in_c=3, length=1)
            "ConvTranspose1d",
            Layer::ConvTranspose1d(
                ConvTranspose1dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), W.to_vec()).unwrap(),
                    zero_bias(),
                    1,
                    0,
                )
                .unwrap(),
            ),
            point_3ch(&[3, 1]),
        ),
        (
            // transpose kernel (in_c=3, out_c=1, kh=1, kw=1); input (in_c=3, H=1, W=1)
            "ConvTranspose2d",
            Layer::ConvTranspose2d(
                ConvTranspose2dLayer::new(
                    ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), W.to_vec()).unwrap(),
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

fn assert_encloses(lower: f32, label: &str) {
    assert!(
        (lower as f64) <= TRUE_OUT + 1e-4,
        "{label}: sound lower {lower} must enclose the true output {TRUE_OUT} (be <= it); \
         a value near 0 means the conv took the plain forward + 1-ULP widening"
    );
}

fn only_lower(bounds: &BoundedTensor) -> f32 {
    bounds.lower().iter().copied().next().unwrap()
}

#[test]
fn graph_sound_ibp_encloses_for_every_conv_variant() {
    for (label, layer, input) in conv_cases() {
        let graph = GraphNetwork::from_sequential(&single_layer_network(layer)).unwrap();
        let out = graph.propagate_ibp_sound(&input).unwrap();
        assert_encloses(
            only_lower(&out),
            &format!("GraphNetwork::propagate_ibp_sound {label}"),
        );
    }
}

#[test]
fn sequential_sound_ibp_encloses_for_every_conv_variant() {
    for (label, layer, input) in conv_cases() {
        let net = single_layer_network(layer);
        let out = net.propagate_ibp_sound(&input).unwrap();
        assert_encloses(
            only_lower(&out),
            &format!("Network::propagate_ibp_sound {label}"),
        );
    }
}

#[test]
fn sequential_collect_sound_ibp_encloses_for_every_conv_variant() {
    for (label, layer, input) in conv_cases() {
        let net = single_layer_network(layer);
        let bounds = net.collect_ibp_bounds_sound(&input).unwrap();
        assert_eq!(bounds.len(), 1, "{label}: one layer => one collected bound");
        assert_encloses(
            only_lower(&bounds[0]),
            &format!("Network::collect_ibp_bounds_sound {label}"),
        );
    }
}

/// The plain (non-sound) forward really does exhibit the cancellation false-tightness
/// these dispatchers must not inherit: without it the tests above would pass vacuously.
#[test]
fn plain_ibp_excludes_the_true_output_for_every_conv_variant() {
    for (label, layer, input) in conv_cases() {
        let net = single_layer_network(layer);
        let out = net.propagate_ibp(&input).unwrap();
        let lower = only_lower(&out);
        assert!(
            (lower as f64) > TRUE_OUT,
            "{label}: plain forward should EXCLUDE the true {TRUE_OUT} (cancellation), \
             got {lower}"
        );
    }
}
