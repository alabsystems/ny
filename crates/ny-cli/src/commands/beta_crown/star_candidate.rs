// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verdict-neutral adapter from NY's real Sequential/VNN-LIB types to star search.
//!
//! This module exists to answer the next algorithm question on actual ACAS Xu models:
//! can the exact-split search close the hard property-2 rows inside budget? It is
//! reachable from dispatch ONLY through [`run_dark_star_probe`], which observes and
//! logs; nothing here can EMIT or ALTER a competition verdict. It CAN delay one: armed,
//! the probe spends its budget on the caller's thread, and a row that would have been
//! answered can time out instead (measured — an `unsat` row became `timeout` with the
//! probe armed). That is why it is default-off and why `# Cost` below says it is not
//! free. In particular, a
//! candidate-empty result still needs roundoff-enclosing star algebra or an independent
//! concrete-model certificate before it can become UNSAT authority.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use ny_mip::star_verify::{
    star_search_unsafe_conjunction, StarBudget, StarLayer, StarStats, StarUnsafeCandidateVerdict,
    StarUnsafeConjunction, StarUnsafeRow,
};
use ny_onnx::vnnlib::{
    CertifiedInputBox, CertifiedRelationalOutputAtom, CertifiedRelationalOutputConjunction,
};
use ny_propagate::{Layer, Network};
use ny_tensor::zonotope::{Star, ZonotopeTensor};

/// One structurally authenticated, verdict-neutral star-search problem.
pub(crate) struct SequentialStarCandidate {
    layers: Vec<StarLayer>,
    input: Star,
    unsafe_region: StarUnsafeConjunction,
    input_width: usize,
    output_width: usize,
}

impl SequentialStarCandidate {
    /// Run candidate search. No returned variant has competition-verdict authority.
    pub(crate) fn search(
        &self,
        budget: &StarBudget,
    ) -> ny_mip::Result<(StarUnsafeCandidateVerdict, StarStats)> {
        star_search_unsafe_conjunction(&self.layers, &self.input, &self.unsafe_region, budget)
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn input_width(&self) -> usize {
        self.input_width
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }

    pub(crate) fn unsafe_row_count(&self) -> usize {
        self.unsafe_region.rows.len()
    }
}

/// Adapt a real Sequential model and source-certified VNN-LIB surfaces.
///
/// Capability is deliberately narrow: dense Linear/ReLU chains plus shape-only
/// Flatten/Reshape/Squeeze/Unsqueeze nodes. Any other layer declines before search.
pub(crate) fn adapt_certified_sequential_candidate(
    network: &Network,
    input_box: &CertifiedInputBox,
    unsafe_outputs: &CertifiedRelationalOutputConjunction,
) -> Result<SequentialStarCandidate> {
    let input = star_from_outward_box(input_box.lower(), input_box.upper())?;
    let input_width = input_box.len();
    let (layers, output_width) = adapt_layers(network, input_width)?;
    let unsafe_region = adapt_unsafe_atoms(unsafe_outputs.atoms(), output_width)?;
    Ok(SequentialStarCandidate {
        layers,
        input,
        unsafe_region,
        input_width,
        output_width,
    })
}

fn adapt_layers(network: &Network, input_width: usize) -> Result<(Vec<StarLayer>, usize)> {
    let mut layers = Vec::with_capacity(network.num_layers());
    let mut width = input_width;
    for (index, layer) in network.layers().iter().enumerate() {
        match layer {
            Layer::Linear(linear) => {
                if linear.in_features() != width {
                    bail!(
                        "star candidate: Linear layer {index} expects {} inputs after width {width}",
                        linear.in_features()
                    );
                }
                if linear.weight().iter().any(|value| !value.is_finite())
                    || linear
                        .bias()
                        .is_some_and(|bias| bias.iter().any(|value| !value.is_finite()))
                {
                    bail!("star candidate: Linear layer {index} has non-finite parameters");
                }
                width = linear.out_features();
                layers.push(StarLayer::Gemm {
                    weight: linear.weight().clone(),
                    bias: linear.bias().cloned(),
                });
            }
            Layer::ReLU(_) => layers.push(StarLayer::Relu),
            // These preserve flattened row-major values; StarLayer is already flat.
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {}
            _ => bail!(
                "star candidate: unsupported Sequential layer at index {index}; only dense \
                 Linear/ReLU chains and shape-only nodes are admitted"
            ),
        }
    }
    if layers.is_empty() {
        bail!("star candidate: empty adapted network");
    }
    Ok((layers, width))
}

/// Build one independent alpha symbol per input coordinate while enclosing the certified
/// outward f64 box in the REAL set `center + radius*alpha` represented by f32 coefficients.
fn star_from_outward_box(lower: &[f64], upper: &[f64]) -> Result<Star> {
    if lower.is_empty() || lower.len() != upper.len() {
        bail!(
            "star candidate: input endpoint lengths differ or are empty ({} vs {})",
            lower.len(),
            upper.len()
        );
    }
    let width = lower.len();
    let mut coefficients = Array2::<f32>::zeros((1 + width, width));
    for (index, (&lo, &hi)) in lower.iter().zip(upper).enumerate() {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            bail!("star candidate: invalid input interval {index}: [{lo}, {hi}]");
        }
        // First enclose the certified f64 endpoints in binary32. Build the affine
        // form against these representable endpoints, then audit the stored f32
        // coefficients as exact f64 values below. Checking `center - radius` in f32
        // would be insufficient: a real endpoint just inside the box can round onto
        // the boundary and make an unsound enclosure appear valid.
        let lo32 = ny_core::f64_to_f32_down(lo);
        let hi32 = ny_core::f64_to_f32_up(hi);
        if !lo32.is_finite() || !hi32.is_finite() || lo32 > hi32 {
            bail!("star candidate: input interval {index} cannot be enclosed in f32");
        }
        let center = (f64::from(lo32).midpoint(f64::from(hi32))) as f32;
        if !center.is_finite() {
            bail!("star candidate: input center {index} is not finite");
        }
        let center64 = f64::from(center);
        let required_radius = (center64 - f64::from(lo32))
            .max(f64::from(hi32) - center64)
            .max(0.0);
        let radius = ny_core::f64_to_f32_up(required_radius);
        if !radius.is_finite() {
            bail!("star candidate: input radius {index} is not finite");
        }
        let represented_lo = center64 - f64::from(radius);
        let represented_hi = center64 + f64::from(radius);
        if represented_lo > f64::from(lo32) || represented_hi < f64::from(hi32) {
            bail!("star candidate: failed to enclose input interval {index}");
        }
        coefficients[[0, index]] = center;
        coefficients[[1 + index, index]] = radius;
    }
    let zonotope = ZonotopeTensor::new(coefficients.into_dyn())
        .context("star candidate: construct input zonotope")?;
    Ok(Star::from_zonotope(zonotope))
}

fn adapt_unsafe_atoms(
    atoms: &[CertifiedRelationalOutputAtom],
    output_width: usize,
) -> Result<StarUnsafeConjunction> {
    if atoms.is_empty() {
        bail!("star candidate: empty unsafe output conjunction");
    }
    let mut rows = Vec::with_capacity(atoms.len());
    for &atom in atoms {
        let (lhs, rhs, strict) = match atom {
            CertifiedRelationalOutputAtom::LessEq(lhs, rhs) => (lhs, rhs, false),
            CertifiedRelationalOutputAtom::GreaterEq(lhs, rhs) => (rhs, lhs, false),
            CertifiedRelationalOutputAtom::LessThan(lhs, rhs) => (lhs, rhs, true),
            CertifiedRelationalOutputAtom::GreaterThan(lhs, rhs) => (rhs, lhs, true),
        };
        if lhs >= output_width || rhs >= output_width {
            bail!(
                "star candidate: unsafe atom references output ({lhs}, {rhs}) at width \
                 {output_width}"
            );
        }
        // Y_lhs <= Y_rhs  ==>  Y_lhs - Y_rhs <= 0.
        let mut coefficients = vec![0.0; output_width];
        coefficients[lhs] += 1.0;
        coefficients[rhs] -= 1.0;
        rows.push(StarUnsafeRow {
            coefficients,
            threshold: 0.0,
            strict,
        });
    }
    Ok(StarUnsafeConjunction { rows })
}

/// Resolved knobs for the verdict-neutral star MEASUREMENT lane (#star-measure).
///
/// Absent `NY_STAR_DARK_SECONDS`, nothing here runs and dispatch is byte-for-byte the
/// pipeline that shipped. The lane exists to answer one question on the real scored
/// inputs — does the exact split search close acasxu prop_2 on 1_5/3_3/4_2 inside
/// budget — and it answers it by LOGGING, never by deciding.
#[derive(Debug, Clone, Copy)]
struct DarkStarProbe {
    budget: Duration,
    max_stars: usize,
    max_depth: usize,
    dual_iters: usize,
    prefer_input_split: bool,
    exact_below_unstable: usize,
}

fn governed_usize(decl: &'static ny_levers::LeverDecl) -> usize {
    let value = ny_levers::read(decl)
        .value
        .as_u64()
        .expect("star measurement integer declarations must resolve to U64");
    usize::try_from(value).expect("UsizeTrimmed rejects values wider than usize")
}

impl DarkStarProbe {
    /// `None` — the shipped default — unless an operator names a positive budget.
    fn from_env() -> Option<Self> {
        use ny_levers::decls::star;

        // Zero is a well-formed value of the declaration; it is this probe's
        // own rule that zero means "do not construct me".
        let seconds = ny_levers::read(&star::STAR_DARK_SECONDS)
            .value
            .as_u64()
            .filter(|seconds| *seconds > 0)?;
        Some(Self {
            budget: Duration::from_secs(seconds),
            // Time is the intended limit; the star cap only keeps a runaway bounded.
            max_stars: governed_usize(&star::STAR_DARK_MAX_STARS),
            max_depth: governed_usize(&star::STAR_DARK_MAX_DEPTH),
            dual_iters: governed_usize(&star::STAR_DARK_DUAL_ITERS),
            // ACAS Xu is 5 inputs against ~300 neurons, so the input tree is the
            // smaller one; `star_verify`'s own module docs argue for this default.
            prefer_input_split: governed_usize(&star::STAR_DARK_INPUT_SPLIT) != 0,
            exact_below_unstable: governed_usize(&star::STAR_DARK_EXACT_BELOW),
        })
    }

    fn star_budget(&self, deadline: Instant) -> StarBudget {
        let mut budget = StarBudget::new(self.max_stars, self.max_depth, deadline);
        budget.dual_iters = self.dual_iters;
        budget.prefer_input_split = self.prefer_input_split;
        budget.exact_below_unstable = self.exact_below_unstable;
        budget
    }
}

/// Compact one-line rendering of the search counters, safe to grep out of a run log.
fn render_stats(stats: &StarStats) -> String {
    let depth_reached = stats.depth_histogram.len().saturating_sub(1);
    let deepest_popped = stats
        .depth_histogram
        .iter()
        .enumerate()
        .rfind(|(_, (popped, _))| *popped > 0)
        .map_or(0, |(depth, _)| depth);
    format!(
        "popped={} splits={} bisections={} pruned_infeasible={} leaves_verified={} \
         discharged_by_overapprox={} lp_reclaimed_stable={} exact_lp_calls={} \
         verified_float_hits={} star_tail_hits={} depth_reached={depth_reached} \
         deepest_popped={deepest_popped} ns_tail={} ns_dual={} ns_exact_lp={} ns_star={}",
        stats.popped,
        stats.splits,
        stats.input_bisections,
        stats.pruned_infeasible,
        stats.leaves_verified,
        stats.discharged_by_overapprox,
        stats.lp_reclaimed_stable,
        stats.exact_lp_calls,
        stats.verified_float_hits,
        stats.star_tail_hits,
        stats.ns_tail,
        stats.ns_dual,
        stats.ns_exact_lp,
        stats.ns_star,
    )
}

/// What the star lane WOULD decide, and what a verdict lane may do with it.
///
/// The mapping is deliberately lossy in the safe direction. `CandidateUnsafeClosureEmpty`
/// is the only outcome that could ever become `unsat`, and it may not until the roundoff
/// enclosure over the split tree is written down; `CandidateUnsafeClosureNotProvedEmpty`
/// could never become `sat` at all without a concrete witness through
/// `gate_sat_with_trusted_oracle`. So both map to `unknown` here, and this function is
/// where a future authority would have to change — visibly — to say anything else.
fn emittable_verdict_today(_verdict: &StarUnsafeCandidateVerdict) -> &'static str {
    "unknown"
}

/// Run the exact split search on a real dispatch row and LOG what it would decide.
///
/// # Verdict neutrality
/// This returns `()`. It cannot influence the caller's result, and its outcome is
/// reported only on stderr under the `NY_STAR_DARK_V1` marker. The two open authority
/// obligations recorded in `ny_mip::star_verify` — `#star-witness` SAT extraction and the
/// UNSAT roundoff enclosure — are undischarged, so no outcome here is verdict evidence.
///
/// # Cost
/// The probe spends its own wall-clock budget from the caller's thread, so an operator
/// who arms it must widen the scored budget by the same amount or accept a timeout. That
/// is the honest shape of a measurement lane: it is not free, and it is off by default.
pub(crate) fn run_dark_star_probe(network: &Network, property: &Path) {
    let Some(probe) = DarkStarProbe::from_env() else {
        return;
    };
    let started = Instant::now();
    let line = dark_star_probe_line(&probe, network, property, started);
    eprintln!(
        "NY_STAR_DARK_V1 property={} elapsed_ms={} input_split={} exact_below={} \
         max_depth={} budget_s={} {line}",
        property.display(),
        started.elapsed().as_millis(),
        probe.prefer_input_split,
        probe.exact_below_unstable,
        probe.max_depth,
        probe.budget.as_secs(),
    );
}

fn dark_star_probe_line(
    probe: &DarkStarProbe,
    network: &Network,
    property: &Path,
    started: Instant,
) -> String {
    let (_ordinary, input_box, unsafe_outputs) =
        match ny_onnx::vnnlib::load_vnnlib_with_certified_affine_property(property) {
            Ok(loaded) => loaded,
            Err(error) => return format!("event=decline stage=property reason=\"{error}\""),
        };
    // Real VNN-COMP ACAS Xu ONNX carries a `SubConstant -> Linear -> AddConstant`
    // normalisation prelude and shape nodes, which the adapter's deliberately narrow
    // Linear/ReLU capability refuses. Reuse the SAME canonicaliser the exact affine
    // lane uses (`mip_preprocess`), whose fold is exact on the f32 parameters, rather
    // than widening the adapter to accept layer families it has no star transformer for.
    let stripped = super::mip_preprocess::strip_shape_layers(network);
    let folded = match super::mip_preprocess::fold_constant_layers(&stripped) {
        Ok(folded) => folded,
        Err(error) => return format!("event=decline stage=fold reason=\"{error}\""),
    };
    let candidate = match adapt_certified_sequential_candidate(&folded, &input_box, &unsafe_outputs)
    {
        Ok(candidate) => candidate,
        Err(error) => return format!("event=decline stage=adapt reason=\"{error}\""),
    };
    let shape = format!(
        "layers={} in={} out={} unsafe_rows={}",
        candidate.layer_count(),
        candidate.input_width(),
        candidate.output_width(),
        candidate.unsafe_row_count(),
    );
    let budget = probe.star_budget(started + probe.budget);
    match candidate.search(&budget) {
        Ok((verdict, stats)) => format!(
            "event=searched {shape} star_verdict={verdict:?} emittable_verdict={} {}",
            emittable_verdict_today(&verdict),
            render_stats(&stats),
        ),
        Err(error) => format!("event=error {shape} reason=\"{error}\""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The measurement lane may never mint verdict authority.
    ///
    /// Measured 2026-08-12: across three configurations on acasxu prop_2 1_5/3_3/4_2 at
    /// the 116 s budget the search produced no `CandidateUnsafeClosureEmpty` at all, so
    /// neither open obligation (`#star-witness`, the UNSAT roundoff enclosure) was even
    /// reachable, let alone discharged. This pins the mapping so that a later edit which
    /// tries to emit on a star outcome has to delete a test that says why it cannot.
    #[test]
    fn no_star_outcome_maps_to_an_emittable_verdict() {
        for verdict in [
            StarUnsafeCandidateVerdict::CandidateUnsafeClosureEmpty,
            StarUnsafeCandidateVerdict::CandidateUnsafeClosureNotProvedEmpty,
            StarUnsafeCandidateVerdict::Unknown("budget".into()),
        ] {
            assert_eq!(
                emittable_verdict_today(&verdict),
                "unknown",
                "{verdict:?} must stay verdict-neutral until its obligation is discharged"
            );
        }
    }

    #[test]
    fn outward_box_adapter_contains_every_certified_endpoint() {
        let lower = [-0.303_531_156, -0.009_549_297, 0.493_380_324];
        let upper = [-0.298_552_812, 0.009_549_297, 0.5];
        let star = star_from_outward_box(&lower, &upper).expect("input star");
        let bounded = star.interval_bounds().expect("bounds");
        for index in 0..lower.len() {
            assert!(f64::from(bounded.lower()[[index]]) <= lower[index]);
            assert!(f64::from(bounded.upper()[[index]]) >= upper[index]);
        }
    }

    #[test]
    fn certified_atom_adapter_preserves_acas_prop2_conjunction() {
        let atoms = [
            CertifiedRelationalOutputAtom::LessEq(1, 0),
            CertifiedRelationalOutputAtom::LessEq(2, 0),
            CertifiedRelationalOutputAtom::LessEq(3, 0),
            CertifiedRelationalOutputAtom::LessEq(4, 0),
        ];
        let spec = adapt_unsafe_atoms(&atoms, 5).expect("unsafe spec");
        assert_eq!(spec.rows.len(), 4);
        for (offset, row) in spec.rows.iter().enumerate() {
            assert_eq!(row.coefficients[0], -1.0);
            assert_eq!(row.coefficients[offset + 1], 1.0);
            assert_eq!(row.threshold, 0.0);
            assert!(!row.strict);
        }
    }

    #[test]
    fn tracked_real_acas_model_and_property_reach_the_candidate_search_surface() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let nnet = ny_onnx::nnet::load_nnet(root.join("tests/models/acasxu_1_1.nnet"))
            .expect("tracked ACAS NNet");
        let network = nnet.to_prop_network().expect("Sequential model");
        let (_ordinary, input, unsafe_outputs) =
            ny_onnx::vnnlib::load_vnnlib_with_certified_affine_property(
                root.join("tests/models/acasxu_prop2.vnnlib"),
            )
            .expect("source-certified property 2");
        let candidate = adapt_certified_sequential_candidate(&network, &input, &unsafe_outputs)
            .expect("real ACAS candidate adapter");
        assert_eq!(candidate.input_width(), 5);
        assert_eq!(candidate.output_width(), 5);
        assert_eq!(candidate.unsafe_row_count(), 4);
        assert_eq!(
            candidate.layer_count(),
            13,
            "6x Linear/ReLU plus final Linear"
        );
    }

    /// Bounded real-instance correctness smoke and optional measurement harness.
    ///
    /// Override `NY_STAR_ACAS_NNET` to point at hard net 3_3 or 4_2 and
    /// `NY_STAR_ACAS_VNNLIB` for its property. The default tracked 1_1/prop2 pair is a
    /// portable one-second smoke, not one of the open score rows. Operators can
    /// raise the environment budgets for an explicit measurement run.
    #[test]
    fn real_acas_candidate_search_smoke_and_optional_measurement() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let nnet_path = std::env::var_os("NY_STAR_ACAS_NNET")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("tests/models/acasxu_1_1.nnet"));
        let property_path = std::env::var_os("NY_STAR_ACAS_VNNLIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("tests/models/acasxu_prop2.vnnlib"));
        let seconds = std::env::var("NY_STAR_ACAS_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1);
        let max_stars = std::env::var("NY_STAR_ACAS_MAX_STARS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20_000);

        let nnet = ny_onnx::nnet::load_nnet(&nnet_path).expect("ACAS NNet");
        let network = nnet.to_prop_network().expect("Sequential model");
        let (_ordinary, input, unsafe_outputs) =
            ny_onnx::vnnlib::load_vnnlib_with_certified_affine_property(&property_path)
                .expect("source-certified property");
        let candidate = adapt_certified_sequential_candidate(&network, &input, &unsafe_outputs)
            .expect("candidate adapter");
        let budget = StarBudget::new(
            max_stars,
            512,
            Instant::now() + Duration::from_secs(seconds),
        );
        let started = Instant::now();
        let (verdict, stats) = candidate.search(&budget).expect("candidate search");
        assert!(
            stats.popped > 0,
            "the smoke must execute the search rather than only construct it"
        );
        eprintln!(
            "star-candidate model={} property={} elapsed={:?} verdict={verdict:?} stats={stats:?}",
            nnet_path.display(),
            property_path.display(),
            started.elapsed()
        );
    }
}
