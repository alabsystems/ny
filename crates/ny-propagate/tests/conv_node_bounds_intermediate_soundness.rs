// Regression for the intermediate-node-bound conv false-proof (#vnncomp-aw-soundness
// self-audit). The production CROWN/BaB verdict path takes per-neuron pre-activation
// boxes from `GraphNetwork::collect_node_bounds`, which feed the ReLU `l >= 0 => identity`
// stability guard (no epsilon margin). Conv2d's PLAIN propagate_ibp is a round-to-nearest
// f32 GEMM that can compute a box EXCLUDING the true pre-activation under cancellation, so
// a truly-unstable ReLU could be mis-classified stable-active -> a FALSE VERIFIED.
//
// Counterexample: 1x1 Conv2d over K=3 channels, W = [2^24, -1, -2^24], bias 0, point input
// [1,1,1]. f32: 2^24 + (-1) + (-2^24) = 0 (the -1 is below half-ULP at magnitude 2^24, lost
// in ANY accumulation order). True (f64): 2^24 - 1 - 2^24 = -1 < 0. The fix routes conv node
// bounds through the SOUND (abssum-Higham, directed) forward, so the production box now
// ENCLOSES -1 and the neuron is correctly unstable.
use ndarray::{arr1, ArrayD, IxDyn};
use ny_propagate::{
    layers::{Conv2dLayer, ReLULayer},
    GraphNetwork, Layer, Network,
};
use ny_tensor::BoundedTensor;

const TWO24: f32 = 16_777_216.0; // 2^24

fn build() -> (Conv2dLayer, BoundedTensor) {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1]), vec![TWO24, -1.0, -TWO24]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.0_f32])), (1, 1), (0, 0), 1, 1)
        .unwrap();
    let pt = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), vec![1.0_f32, 1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(pt.clone(), pt).unwrap();
    (conv, input)
}

#[test]
fn conv_intermediate_node_bound_encloses_true_pre_activation() {
    let (conv, input) = build();

    // The f32 cancellation that defeated the round-to-nearest path.
    let computed = TWO24 + ((-1.0f32) + (-TWO24));
    assert_eq!(
        computed, 0.0,
        "round-to-nearest conv lower computes exactly 0"
    );
    let true_lower: f64 = -1.0; // 2^24 - 1 - 2^24, exact in f64; < 0 => truly UNSTABLE

    // PRODUCTION node-bound path: GraphNetwork::collect_node_bounds. After the fix the
    // Conv2d node bound is the SOUND forward, so it must ENCLOSE the true pre-activation.
    let mut net = Network::new();
    net.add_layer(Layer::Conv2d(conv.clone()));
    net.add_layer(Layer::ReLU(ReLULayer));
    let graph = GraphNetwork::from_sequential(&net).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    // The conv pre-activation node is the (1,1,1) box with lower < 0 (the ReLU input);
    // the ReLU OUTPUT box, also (1,1,1), has lower >= 0. (The input is (3,1,1).) After the
    // sound fix the conv box is WIDENED outward ([-~10, +~10]), no longer a point box.
    let mut conv_lower: Option<f32> = None;
    for bt in node_bounds.values() {
        let lo = bt.lower();
        if lo.shape() == [1usize, 1, 1] {
            let v = lo.iter().copied().next().unwrap();
            if v < 0.0 {
                conv_lower = Some(v);
            }
        }
    }
    let prod_lower = conv_lower.expect(
        "production conv node bound must be negative (enclosing true -1); a non-negative \
         value means the round-to-nearest false-proof regressed",
    );

    // SOUND enclosure: the production lower must be <= the true pre-activation min (-1).
    assert!(
        (prod_lower as f64) <= true_lower + 1e-4,
        "production conv node lower {prod_lower} must enclose true {true_lower} (be <= it)"
    );
    // And therefore the neuron is correctly classified UNSTABLE (lower < 0), not the
    // mis-classified stable-active (lower >= 0) that produced the false verdict.
    assert!(
        prod_lower < 0.0,
        "conv pre-activation must be unstable (lower < 0), got {prod_lower}"
    );

    // The production path now agrees with the sound twin it routes through.
    let sound_lower = conv
        .propagate_ibp_sound_with_engine(&input, None)
        .unwrap()
        .lower()
        .iter()
        .copied()
        .next()
        .unwrap();
    assert_eq!(
        prod_lower, sound_lower,
        "production node bound must equal the sound IBP it now dispatches to"
    );
}
