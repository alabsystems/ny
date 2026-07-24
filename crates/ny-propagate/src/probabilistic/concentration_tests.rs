// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for concentration inequality certificates.
//! Part of #3921 Phase 2.

use super::*;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

#[derive(Clone, Default)]
struct SharedBuffer(Arc<Mutex<String>>);

impl SharedBuffer {
    fn snapshot(&self) -> String {
        self.0.lock().expect("buffer lock").clone()
    }

    fn push_line(&self, line: String) {
        let mut buffer = self.0.lock().expect("buffer lock");
        buffer.push_str(&line);
        buffer.push('\n');
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl MessageVisitor {
    fn render(self) -> String {
        match (self.message, self.fields.is_empty()) {
            (Some(message), true) => message,
            (Some(message), false) => format!("{message} {}", self.fields.join(" ")),
            (None, true) => String::new(),
            (None, false) => self.fields.join(" "),
        }
    }
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let rendered = format!("{value:?}");
        if field.name() == "message" {
            self.message = Some(rendered.trim_matches('"').to_string());
        } else {
            self.fields.push(format!("{}={rendered}", field.name()));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }
}

#[derive(Clone, Default)]
struct RecordingSubscriber {
    buffer: SharedBuffer,
    next_id: Arc<AtomicU64>,
}

impl RecordingSubscriber {
    fn output(&self) -> String {
        self.buffer.snapshot()
    }
}

impl Subscriber for RecordingSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        self.buffer
            .push_line(format!("{} {}", event.metadata().level(), visitor.render()));
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[test]
fn test_hoeffding_known_range() {
    // Known range [0, 1], n=1000 samples → epsilon < 0.1 at 95% confidence
    let mean = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 0.3, 0.7]).unwrap();
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap();
    let crown_bounds = BoundedTensor::new(lower, upper).unwrap();

    let bounds = hoeffding_bound(&mean, &crown_bounds, 1000, 0.95).unwrap();

    assert_eq!(bounds.len(), 3);
    for b in &bounds {
        assert!(b.epsilon < 0.1, "epsilon={} should be < 0.1", b.epsilon);
        assert!(b.failure_probability <= 0.05 + 1e-10);
        assert_eq!(b.num_samples, 1000);
        assert!((b.bound_range - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_hoeffding_tighter_crown_bounds_give_smaller_epsilon() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();

    // Wide range: [0, 10]
    let wide_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap(),
    )
    .unwrap();
    let wide = hoeffding_bound(&mean, &wide_bounds, 500, 0.95).unwrap();

    // Tight range: [0, 1]
    let tight_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let tight = hoeffding_bound(&mean, &tight_bounds, 500, 0.95).unwrap();

    // Tighter CROWN bounds → smaller epsilon (monotone in range)
    assert!(
        tight[0].epsilon < wide[0].epsilon,
        "tight eps={} should be < wide eps={}",
        tight[0].epsilon,
        wide[0].epsilon
    );
}

#[test]
fn test_hoeffding_more_samples_reduce_epsilon() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let few = hoeffding_bound(&mean, &bounds, 100, 0.95).unwrap();
    let many = hoeffding_bound(&mean, &bounds, 10000, 0.95).unwrap();

    assert!(many[0].epsilon < few[0].epsilon);
}

#[test]
fn test_hoeffding_zero_range_gives_zero_epsilon() {
    // Point bounds: lower == upper → range = 0, epsilon = 0
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap();
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let result = hoeffding_bound(&mean, &bounds, 100, 0.95).unwrap();
    for b in &result {
        assert!((b.epsilon).abs() < 1e-12);
    }
}

#[test]
fn test_mcdiarmid_basic() {
    let output = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.1, -0.1, -0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.1, 0.1]).unwrap(),
    )
    .unwrap();
    let lipschitz = LipschitzEstimate {
        value: 2.0,
        is_sound: true,
        unhandled_layers: Vec::new(),
    };

    let bounds = mcdiarmid_bound(&output, &input_bounds, &lipschitz, 1000, 0.95).unwrap();
    assert_eq!(bounds.len(), 2);
    for b in &bounds {
        assert!(b.epsilon > 0.0);
        assert!(b.epsilon.is_finite());
        assert!(b.failure_probability <= 0.05 + 1e-10);
    }
}

#[test]
fn test_mcdiarmid_zero_lipschitz() {
    // Constant network (Lipschitz = 0) → epsilon = 0
    let output = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();
    let lipschitz = LipschitzEstimate {
        value: 0.0,
        is_sound: true,
        unhandled_layers: Vec::new(),
    };

    let bounds = mcdiarmid_bound(&output, &input_bounds, &lipschitz, 100, 0.95).unwrap();
    assert!((bounds[0].epsilon).abs() < 1e-12);
}

/// #4145: Default mcdiarmid_bound rejects unsound Lipschitz estimates.
#[test]
fn test_mcdiarmid_bound_rejects_unsound_lipschitz_4145() {
    let output = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.2]).unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    )
    .unwrap();
    let unsound_lip = LipschitzEstimate {
        value: 1.0,
        is_sound: false,
        unhandled_layers: vec!["Exp".to_string()],
    };
    let err = mcdiarmid_bound(&output, &input_bounds, &unsound_lip, 128, 0.95)
        .expect_err("unsound Lipschitz must be rejected by default");
    let msg = format!("{err}");
    assert!(
        msg.contains("sound Lipschitz estimate"),
        "error message: {msg}"
    );
    assert!(
        msg.contains("Exp"),
        "error should name unhandled layers: {msg}"
    );
}

/// #4145: Optimistic variant warns but succeeds for unsound estimates.
#[test]
fn test_mcdiarmid_optimistic_warns_when_lipschitz_unsound() {
    let output = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.2]).unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    )
    .unwrap();
    let lipschitz = LipschitzEstimate {
        value: 1.0,
        is_sound: false,
        unhandled_layers: vec!["Exp".to_string()],
    };
    let subscriber = RecordingSubscriber::default();
    let output_buffer = subscriber.clone();

    let bounds = tracing::subscriber::with_default(subscriber, || {
        mcdiarmid_bound_optimistic(&output, &input_bounds, &lipschitz, 128, 0.95)
    })
    .expect("optimistic variant should warn, not fail");

    assert_eq!(bounds.len(), 1);
    assert!(
        !bounds[0].is_sound,
        "optimistic bound should be marked unsound"
    );

    let output = output_buffer.output();
    assert!(
        output.contains("WARN"),
        "expected warn-level output, got: {output}"
    );
    assert!(
        output.contains("optimistic Lipschitz estimate"),
        "expected unsound-estimate warning, got: {output}"
    );
}

/// #4145: McDiarmid per-bound `is_sound` reflects the Lipschitz estimate.
/// Sound path uses default API; unsound path requires explicit opt-in.
#[test]
fn test_mcdiarmid_bound_is_sound_reflects_lipschitz_4145() {
    let output = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.1, -0.1, -0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.1, 0.1]).unwrap(),
    )
    .unwrap();

    let sound_lip = LipschitzEstimate {
        value: 2.0,
        is_sound: true,
        unhandled_layers: Vec::new(),
    };
    for b in &mcdiarmid_bound(&output, &input_bounds, &sound_lip, 100, 0.95).unwrap() {
        assert!(b.is_sound, "sound Lipschitz → is_sound=true");
    }

    // Default API rejects unsound estimates.
    let unsound_lip = LipschitzEstimate {
        value: 2.0,
        is_sound: false,
        unhandled_layers: vec!["Exp".to_string()],
    };
    assert!(
        mcdiarmid_bound(&output, &input_bounds, &unsound_lip, 100, 0.95).is_err(),
        "default mcdiarmid_bound must reject unsound Lipschitz"
    );

    // Optimistic API accepts unsound estimates with is_sound=false on each bound.
    for b in &mcdiarmid_bound_optimistic(&output, &input_bounds, &unsound_lip, 100, 0.95).unwrap() {
        assert!(!b.is_sound, "unsound Lipschitz → is_sound=false");
    }
}

/// #4145: Certificate `is_sound` propagates from the Lipschitz estimate.
/// Default API rejects unsound; optimistic API marks certificate unsound.
#[test]
fn test_certificate_is_sound_propagates_lipschitz_4145() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
    let output = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
    let crown_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.1, -0.1, -0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.1, 0.1]).unwrap(),
    )
    .unwrap();

    // Sound Lipschitz → sound certificate via default API.
    let sound_lip = LipschitzEstimate {
        value: 2.0,
        is_sound: true,
        unhandled_layers: Vec::new(),
    };
    let sound_cert = ConcentrationCertificate::compute_with_mcdiarmid(
        &mean,
        &crown_bounds,
        &output,
        &input_bounds,
        &sound_lip,
        100,
        0.95,
        false,
    )
    .unwrap();
    assert!(sound_cert.is_sound, "sound Lipschitz → sound certificate");

    // Unsound Lipschitz → default API rejects.
    let unsound_lip = LipschitzEstimate {
        value: 2.0,
        is_sound: false,
        unhandled_layers: vec!["Exp".to_string()],
    };
    assert!(
        ConcentrationCertificate::compute_with_mcdiarmid(
            &mean,
            &crown_bounds,
            &output,
            &input_bounds,
            &unsound_lip,
            100,
            0.95,
            false,
        )
        .is_err(),
        "default compute_with_mcdiarmid must reject unsound Lipschitz"
    );

    // Unsound Lipschitz → optimistic API returns unsound certificate.
    let unsound_cert = ConcentrationCertificate::compute_with_mcdiarmid_optimistic(
        &mean,
        &crown_bounds,
        &output,
        &input_bounds,
        &unsound_lip,
        100,
        0.95,
        false,
    )
    .unwrap();
    assert!(
        !unsound_cert.is_sound,
        "unsound Lipschitz → unsound certificate via optimistic API"
    );

    // Hoeffding-only → always sound.
    let hoeffding_cert =
        ConcentrationCertificate::compute(&mean, &crown_bounds, 100, 0.95, false).unwrap();
    assert!(hoeffding_cert.is_sound, "Hoeffding-only → always sound");
}

#[test]
fn test_estimate_lipschitz_linear_network() {
    // Two-layer linear network: Lipschitz = sigma_1 * sigma_2
    let w1 = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();
    let w2 = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();

    let l1 = crate::layers::LinearLayer::new(w1, None).unwrap();
    let l2 = crate::layers::LinearLayer::new(w2, None).unwrap();

    let mut net = Network::new();
    net.add_layer(Layer::Linear(l1));
    net.add_layer(Layer::ReLU(crate::layers::ReLULayer));
    net.add_layer(Layer::Linear(l2));

    let lip = estimate_lipschitz_from_network(&net).unwrap();
    // sigma(w1) = 2.0 (diagonal matrix), sigma(w2) = sqrt(2) ≈ 1.414
    // Product ≈ 2.83, but spectral_norm is an upper bound so lip >= 2.83
    assert!(lip.value >= 2.0, "Lipschitz={} should be >= 2.0", lip.value);
    assert!(lip.value.is_finite());
    assert!(lip.is_sound);
    assert!(lip.unhandled_layers.is_empty());
}

#[test]
fn test_estimate_lipschitz_reports_unhandled_layers() {
    let mut net = Network::new();
    net.add_layer(Layer::Exp(crate::layers::ExpLayer::new()));

    let lip = estimate_lipschitz_from_network(&net).unwrap();

    assert_eq!(lip.value, 1.0);
    assert!(!lip.is_sound);
    assert_eq!(lip.unhandled_layers, vec!["Exp".to_string()]);
}

#[test]
fn test_estimate_lipschitz_zero_linear_network_is_sound() {
    let zero_weight = Array2::zeros((2, 2));
    let linear = crate::layers::LinearLayer::new(zero_weight, None).unwrap();

    let mut net = Network::new();
    net.add_layer(Layer::Linear(linear));

    let lip = estimate_lipschitz_from_network(&net).unwrap();

    assert_eq!(lip.value, 0.0);
    assert!(lip.is_sound);
    assert!(lip.unhandled_layers.is_empty());
}

#[test]
fn test_concentration_certificate_compute_with_mcdiarmid_populates_sections() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let output = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.45]).unwrap();
    let crown_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.1, -0.2]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
    )
    .unwrap();
    let lipschitz = LipschitzEstimate {
        value: 1.5,
        is_sound: true,
        unhandled_layers: Vec::new(),
    };

    let certificate = ConcentrationCertificate::compute_with_mcdiarmid(
        &mean,
        &crown_bounds,
        &output,
        &input_bounds,
        &lipschitz,
        256,
        0.95,
        false,
    )
    .unwrap();

    assert_eq!(certificate.hoeffding_bounds.len(), 1);
    assert_eq!(certificate.mcdiarmid_bounds.as_ref().map(Vec::len), Some(1));
    assert!((certificate.overall_confidence - 0.95).abs() < 1e-12);
}

#[test]
fn test_hoeffding_invalid_inputs() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Zero samples
    assert!(hoeffding_bound(&mean, &bounds, 0, 0.95).is_err());
    // Invalid confidence
    assert!(hoeffding_bound(&mean, &bounds, 100, 1.0).is_err());
    assert!(hoeffding_bound(&mean, &bounds, 100, -0.1).is_err());
}

/// #4331: Bonferroni correction divides per-dim failure probability by d.
#[test]
fn test_bonferroni_correction_10dim_95pct() {
    let dim = 10;
    let mean = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.5; dim]).unwrap();
    let lower = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0; dim]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0; dim]).unwrap();
    let crown_bounds = BoundedTensor::new(lower, upper).unwrap();

    // Without Bonferroni: per-dim delta = 0.05
    let cert_no_bonf =
        ConcentrationCertificate::compute(&mean, &crown_bounds, 1000, 0.95, false).unwrap();
    // With Bonferroni: per-dim delta = 0.05 / 10 = 0.005
    let cert_bonf =
        ConcentrationCertificate::compute(&mean, &crown_bounds, 1000, 0.95, true).unwrap();

    // Both should have overall_confidence = 0.95
    assert!((cert_no_bonf.overall_confidence - 0.95).abs() < 1e-12);
    assert!((cert_bonf.overall_confidence - 0.95).abs() < 1e-12);

    // Bonferroni should produce wider epsilon (more conservative per-dim)
    for i in 0..dim {
        assert!(
            cert_bonf.hoeffding_bounds[i].epsilon > cert_no_bonf.hoeffding_bounds[i].epsilon,
            "Bonferroni epsilon[{i}]={} should be > non-Bonferroni epsilon={}",
            cert_bonf.hoeffding_bounds[i].epsilon,
            cert_no_bonf.hoeffding_bounds[i].epsilon
        );
    }

    // Verify the per-dim failure probability with Bonferroni is ~0.005
    // Hoeffding failure_probability = 2*exp(-2*n*eps^2/R^2) = delta_per_dim
    // With Bonferroni: delta_per_dim = 0.005
    for b in &cert_bonf.hoeffding_bounds {
        assert!(
            (b.failure_probability - 0.005).abs() < 1e-6,
            "Bonferroni per-dim failure_prob={} should be ~0.005",
            b.failure_probability
        );
    }
}

/// #4331: Single-dim output → Bonferroni has no effect.
#[test]
fn test_bonferroni_no_effect_single_dim() {
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let crown_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let cert_no =
        ConcentrationCertificate::compute(&mean, &crown_bounds, 500, 0.95, false).unwrap();
    let cert_yes =
        ConcentrationCertificate::compute(&mean, &crown_bounds, 500, 0.95, true).unwrap();

    assert!(
        (cert_no.hoeffding_bounds[0].epsilon - cert_yes.hoeffding_bounds[0].epsilon).abs() < 1e-12,
        "Bonferroni should have no effect for single-dim output"
    );
}
