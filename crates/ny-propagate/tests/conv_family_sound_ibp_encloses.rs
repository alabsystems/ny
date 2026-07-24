// Regression for #vnncomp-aw-soundness: Conv1d / ConvTranspose1d / ConvTranspose2d
// SOUND IBP forward must ENCLOSE the true output under f32 cancellation, where the plain
// round-to-nearest forward (used as a production node bound before this fix) EXCLUDES it.
//
// Cancellation kernel over K=3 input channels: W=[2^24, -1, -2^24], point input [1,1,1].
//   plain f32:  2^24 + (-1) + (-2^24) = 0   (the -1 is below half-ULP at 2^24, lost)
//   true (f64): 2^24 - 1 - 2^24 = -1  (< 0)
// The plain node box [0, .] excludes -1 (would mis-classify a downstream ReLU as
// stable-active -> false VERIFIED); the sound box must contain -1.
use ndarray::{arr1, ArrayD, IxDyn};
use ny_propagate::layers::{Conv1dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer};
use ny_tensor::BoundedTensor;

const TWO24: f32 = 16_777_216.0; // 2^24
const W: [f32; 3] = [TWO24, -1.0, -TWO24];
const TRUE_LOWER: f64 = -1.0; // 2^24 - 1 - 2^24, exact

fn point_3ch(shape: &[usize]) -> BoundedTensor {
    let pt = ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32, 1.0, 1.0]).unwrap();
    BoundedTensor::new(pt.clone(), pt).unwrap()
}

fn assert_encloses(plain_lower: f32, sound_lower: f32, label: &str) {
    // Sanity: the plain forward exhibits the cancellation false-tightness.
    assert!(
        plain_lower as f64 > TRUE_LOWER,
        "{label}: plain forward should EXCLUDE the true {TRUE_LOWER} (cancellation), got {plain_lower}"
    );
    // The fix: the sound forward ENCLOSES the true value (lower <= true).
    assert!(
        (sound_lower as f64) <= TRUE_LOWER + 1e-4,
        "{label}: sound lower {sound_lower} must enclose true {TRUE_LOWER} (be <= it)"
    );
}

#[test]
fn conv1d_sound_ibp_encloses_under_cancellation() {
    // kernel (out_c=1, in_c=3, kw=1); input (in_c=3, length=1).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), W.to_vec()).unwrap();
    let layer = Conv1dLayer::new(kernel, Some(arr1(&[0.0_f32])), 1, 0).unwrap();
    let input = point_3ch(&[3, 1]);

    let plain = layer.propagate_ibp_with_engine(&input, None).unwrap();
    let sound = layer.propagate_ibp_sound_with_engine(&input, None).unwrap();
    assert_encloses(
        plain.lower().iter().copied().next().unwrap(),
        sound.lower().iter().copied().next().unwrap(),
        "Conv1d",
    );
}

#[test]
fn conv_transpose1d_sound_ibp_encloses_under_cancellation() {
    // transpose kernel (in_c=3, out_c/groups=1, kw=1); input (in_c=3, length=1).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), W.to_vec()).unwrap();
    let layer = ConvTranspose1dLayer::new(kernel, Some(arr1(&[0.0_f32])), 1, 0).unwrap();
    let input = point_3ch(&[3, 1]);

    let plain = layer.propagate_ibp_with_engine(&input, None).unwrap();
    let sound = layer.propagate_ibp_sound_with_engine(&input, None).unwrap();
    assert_encloses(
        plain.lower().iter().copied().next().unwrap(),
        sound.lower().iter().copied().next().unwrap(),
        "ConvTranspose1d",
    );
}

#[test]
fn conv_transpose2d_sound_ibp_encloses_under_cancellation() {
    // transpose kernel (in_c=3, out_c=1, kh=1, kw=1); input (in_c=3, H=1, W=1).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), W.to_vec()).unwrap();
    let layer = ConvTranspose2dLayer::new(kernel, Some(arr1(&[0.0_f32])), (1, 1), (0, 0)).unwrap();
    let input = point_3ch(&[3, 1, 1]);

    let plain = layer.propagate_ibp_with_engine(&input, None).unwrap();
    let sound = layer.propagate_ibp_sound_with_engine(&input, None).unwrap();
    assert_encloses(
        plain.lower().iter().copied().next().unwrap(),
        sound.lower().iter().copied().next().unwrap(),
        "ConvTranspose2d",
    );
}
