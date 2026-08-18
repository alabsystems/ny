// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CLI branching heuristic parsing.
//!
//! Centralizes the string→[`BranchingHeuristic`] mapping that was previously
//! duplicated between `bench_acasxu` and `beta_crown verify`.  Each command
//! layers its own policy on top (GPU-BaB gate, relu-split shorthand) via
//! the helpers here.
//!
//! Part of #1893.

use anyhow::Result;
use ny_core::LayerType;
use ny_onnx::Network;
use ny_propagate::{BetaCrownConfig, BranchingHeuristic};

/// The eight base tokens accepted everywhere (including aliases).
///
/// Kept as a constant so contract tests can assert parity.
#[cfg(test)]
const BASE_TOKENS: &[&str] = &[
    "width",
    "impact",
    "babsr",
    "fsb",
    "kfsb",
    "kfsb-intercept-only",
    "sequential",
    "input",
];

/// Extended token only accepted by `beta_crown verify` (triggers ReLU splitting).
pub(crate) const RELU_TOKEN: &str = "relu";

/// Token requesting model-intrinsic auto-selection of the branching method.
///
/// This is the DEFAULT value of `--branching` for `beta_crown verify`. It is
/// resolved into a concrete [`BranchingHeuristic`] once the model + spec are
/// loaded (see [`auto_select_branching`]). `auto` defers to a preset's
/// `bab.branching.method` when one is declared (so benchmark presets keep
/// control), and otherwise self-selects from the model's structure.
pub(crate) const AUTO_TOKEN: &str = "auto";

/// Input element-count threshold below which input-splitting is *unconditionally*
/// preferred (the "low-dim input" class).
///
/// Networks whose input is low-dimensional (e.g. ACAS-Xu = 5, TLLVerifyBench = 2,
/// control-system state vectors of 4-6, nn4sys lindex = 1, sat_relu = 30, cgan
/// latent = 5) verify far better with INPUT splitting than with ReLU (kFSB)
/// splitting: the input box is tightened directly and ReLU-splitting exhausts its
/// domain budget without closing the bound. ReLU splitting only pays off once
/// there is a large unstable-neuron frontier, which low-dim nets don't have.
///
/// The in-repo VNN-COMP low-dim input-split categories all sit at <= 30 inputs
/// (sat_relu = 30 is the largest); 64 leaves a robust margin above them while
/// staying well below the smallest moderate/high-dim category considered by the
/// structural rules (collins_rul = 400). The low-dim ReLU-split category
/// safenlp (= 30, NLP) is disambiguated by the MIP complete-verifier signal, not
/// by input dimensionality.
pub(crate) const INPUT_SPLIT_MAX_INPUT_DIM: usize = 64;

/// Upper bound on input element count for the "moderate-dim small/shallow net"
/// input-split class.
///
/// Above [`INPUT_SPLIT_MAX_INPUT_DIM`] but at/below this value, input splitting is
/// still the right call *iff* the network is structurally small (few parameters
/// AND few ReLU nodes): the input box can be fanned out and the small,
/// few-unstable-neuron net converges before the domain budget is exhausted. This
/// is the "dist_shift class": dist_shift = 792 inputs over a small conv/FC MNIST
/// autoencoder wants INPUT splitting even though 792 > [`INPUT_SPLIT_MAX_INPUT_DIM`].
///
/// The band stops at 2048 so that genuinely high-dimensional image / transformer
/// inputs (CIFAR = 3072, ViT = 3072, traffic_signs = 12288, TinyImageNet = 9408,
/// VGGNet = 150528) are never eligible — those have a huge unstable-neuron
/// frontier and are hopeless for input splitting regardless of param count.
pub(crate) const MODERATE_INPUT_SPLIT_MAX_INPUT_DIM: usize = 2048;

/// Max parameter count for the "moderate-dim small/shallow net" input-split class.
///
/// dist_shift's small MNIST autoencoder is well under this; image/feature CNNs in
/// the same input band (e.g. collins_rul) carry many more parameters and a far
/// larger unstable-ReLU frontier, so they correctly fall through to ReLU (kFSB)
/// splitting. Used in conjunction with [`MODERATE_NET_MAX_RELU`] — BOTH must hold,
/// so a small-parameter-but-deep net or a shallow-but-huge net is not mistaken for
/// the dist_shift class.
pub(crate) const MODERATE_NET_MAX_PARAMS: usize = 2_000_000;

/// Max ReLU-family node count for the "moderate-dim small/shallow net" class.
///
/// Counts ReLU / LeakyReLU / PReLU activation *nodes* (not individual neurons) as a
/// cheap proxy for network depth / the size of the unstable-neuron frontier.
/// A handful of activation layers (autoencoder-scale) keeps input splitting
/// tractable; many activation layers (deep conv / transformer stacks) imply a
/// large unstable frontier where ReLU splitting wins.
pub(crate) const MODERATE_NET_MAX_RELU: usize = 64;

/// Max-domains budget applied when `auto` selects input splitting and the user
/// did not pass an explicit `--max-domains`.
///
/// Mirrors the input-split companion budget the VNN-COMP runner used
/// (`--max-domains 50000` for the input-split categories; see
/// `vnncomp_scripts/run_instance.sh`). Input splitting fans the input box into
/// many subdomains, so it needs a larger frontier than the ReLU-split default;
/// this keeps auto-selected input splitting behaving like the hand-tuned
/// configuration. SOUND: a domain cap only bounds search effort — hitting it
/// yields `unknown`/`timeout`, never a wrong verdict.
pub(crate) const AUTO_INPUT_SPLIT_MAX_DOMAINS: usize = 50_000;

/// Why [`auto_select_branching`] chose a particular method (for logging).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutoBranchingReason {
    /// MIP is the active complete verifier (SAT-encoded / NLP / malware-Conv):
    /// CROWN is too loose, MIP does the real work, and ReLU (kFSB) splitting is
    /// the correct BaB fallback. Input splitting is hopeless on these nets.
    MipComplete,
    /// The network input is low-dimensional, so input splitting tightens the
    /// input box directly and beats ReLU splitting.
    LowDimInput,
    /// The network input is moderate-dimensional but the net is structurally
    /// small/shallow (few params AND few ReLU nodes), so the unstable-neuron
    /// frontier is small and input splitting still converges. This is the
    /// "dist_shift class": a small conv/FC autoencoder over a ~hundreds-element
    /// input.
    SmallShallowNet,
    /// The network input is high-dimensional (image / FC / transformer), OR the
    /// net carries a large unstable-ReLU frontier (deep/wide conv or transformer
    /// stack), so ReLU (kFSB) splitting is the only tractable choice.
    HighDimOrManyRelu,
}

impl AutoBranchingReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::MipComplete => "MIP complete-verifier active (ReLU/kFSB BaB fallback)",
            Self::LowDimInput => "low-dimensional network input (input splitting)",
            Self::SmallShallowNet => "moderate input over a small/shallow net (input splitting)",
            Self::HighDimOrManyRelu => {
                "high-dimensional input or large unstable-ReLU frontier (ReLU/kFSB splitting)"
            }
        }
    }
}

/// Cheap, model-intrinsic signals available at/after model load, used to pick the
/// branching method from the model's STRUCTURE rather than from raw input size
/// alone.
///
/// All fields are derived during load (input shape, the ONNX/native layer list,
/// and `param_count`), so gathering them costs nothing extra. The structural
/// fields are bundled in an [`Option`] (`structure`) because the resolution can
/// also run *before* the model is built (e.g. epsilon-ball mode peeked only the
/// spec input count); in that pre-structure case the heuristic degrades to the
/// dimensionality-only decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelClassSignals {
    /// Flattened input element count (e.g. 5 for ACAS-Xu, 3072 for CIFAR).
    pub(crate) input_element_count: usize,
    /// MIP is the active complete verifier (SAT/NLP/malware categories).
    pub(crate) mip_complete_verifier: bool,
    /// Structural signals from the loaded network; `None` when the model has not
    /// been built yet (input count came only from the spec).
    pub(crate) structure: Option<ModelStructure>,
}

/// Structural signals extracted from a loaded network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelStructure {
    /// Total parameter count (capacity / width / depth proxy).
    pub(crate) param_count: usize,
    /// Whether the model contains Conv2d / ConvTranspose layers.
    pub(crate) has_conv: bool,
    /// Count of ReLU-family activation nodes (depth / unstable-frontier proxy).
    pub(crate) relu_node_count: usize,
    /// Whether the model is a non-sequential graph (DAG: residual/attention/concat).
    pub(crate) is_dag: bool,
}

impl ModelClassSignals {
    /// Convenience constructor for the pre-structure (dimensionality-only) case.
    #[cfg(test)]
    pub(crate) fn dim_only(input_element_count: usize, mip_complete_verifier: bool) -> Self {
        Self {
            input_element_count,
            mip_complete_verifier,
            structure: None,
        }
    }
}

impl ModelStructure {
    /// Extract the cheap structural signals from a loaded `Network`.
    ///
    /// All signals are read directly off the parsed layer list and the network's
    /// `param_count`, so this is O(layers) and allocates nothing — it costs nothing
    /// beyond a single pass over the already-loaded metadata.
    ///
    /// `is_dag` is supplied by the caller (it derives from the graph-build pass,
    /// which detects binary/multi-input ops the flat layer list does not surface).
    pub(crate) fn from_network(network: &Network, is_dag: bool) -> Self {
        let has_conv = network.layers.iter().any(|layer| {
            matches!(
                layer.layer_type,
                LayerType::Conv2d
                    | LayerType::Conv1d
                    | LayerType::ConvTranspose2d
                    | LayerType::ConvTranspose1d
            )
        });
        let relu_node_count = network
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.layer_type,
                    LayerType::ReLU | LayerType::LeakyRelu | LayerType::PRelu
                )
            })
            .count();
        Self {
            param_count: network.param_count,
            has_conv,
            relu_node_count,
            is_dag,
        }
    }
}

/// A request to resolve `--branching auto` once the model is loaded.
///
/// Carries everything the model-class-aware selection needs that is NOT derivable
/// from the loaded network itself: the MIP-complete-verifier flag and the spec's
/// input element count (used as a fallback when the model's own `input_dim`
/// could not be resolved). [`resolve_auto_branching`] fills in the structural
/// signals from the freshly-loaded model and returns a [`ResolvedAutoBranching`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct AutoBranchingRequest {
    /// MIP is the active complete verifier.
    pub(crate) mip_complete_verifier: bool,
    /// Spec-peeked input element count, if a VNN-LIB spec was present.
    pub(crate) spec_input_count: Option<usize>,
}

/// The resolved outcome of model-class-aware auto-branching.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedAutoBranching {
    /// The selected branching heuristic.
    pub(crate) heuristic: BranchingHeuristic,
    /// Whether ReLU splitting should be enabled (true iff `heuristic == Kfsb`).
    pub(crate) use_relu_split: bool,
    /// Whether input splitting was selected.
    pub(crate) is_input_split: bool,
    /// Why the method was chosen (for logging).
    pub(crate) reason: AutoBranchingReason,
    /// The input element count actually used for the decision.
    pub(crate) input_element_count: usize,
    /// Enable aggregation-critical full kFSB for the high-dimensional
    /// auto-selected lane. MIP fallback kFSB deliberately leaves this off.
    pub(crate) use_multi_objective_critical_kfsb: bool,
}

/// Resolve `--branching auto` from a request plus the loaded model's structure.
///
/// `model_input_dim` is the authoritative flattened input element count from the
/// loaded model; the spec-peeked count in `request` is used only when the model
/// dim is unavailable (it should always be available post-load, but we keep the
/// fallback for robustness).
///
/// SOUNDNESS: returns only a (sound) branching-method choice; never a verdict.
pub(crate) fn resolve_auto_branching(
    request: AutoBranchingRequest,
    structure: ModelStructure,
    model_input_dim: usize,
) -> ResolvedAutoBranching {
    let input_element_count = if model_input_dim > 0 {
        model_input_dim
    } else {
        request.spec_input_count.unwrap_or(model_input_dim)
    };
    let signals = ModelClassSignals {
        input_element_count,
        mip_complete_verifier: request.mip_complete_verifier,
        structure: Some(structure),
    };
    let (heuristic, reason) = auto_select_branching(signals);
    let is_input_split = matches!(heuristic, BranchingHeuristic::InputSplit);
    // A resolved Kfsb heuristic implies ReLU splitting, which routes conv/DAG
    // models to the GRAPH engine. Mirrors explicit-kfsb behavior.
    let use_relu_split = matches!(heuristic, BranchingHeuristic::Kfsb);
    let use_multi_objective_critical_kfsb = matches!(
        (&heuristic, reason),
        (
            BranchingHeuristic::Kfsb,
            AutoBranchingReason::HighDimOrManyRelu
        )
    );
    ResolvedAutoBranching {
        heuristic,
        use_relu_split,
        is_input_split,
        reason,
        input_element_count,
        use_multi_objective_critical_kfsb,
    }
}

/// Stamp policy that is valid only after model-aware auto branching resolves.
///
/// Keeping this separate from reusable preset application prevents explicit
/// kFSB, MIP-fallback, and unrelated presets from inheriting the costly full
/// multi-objective scorer.
pub(crate) fn apply_resolved_auto_branching_runtime_policy(
    config: &mut BetaCrownConfig,
    resolved: &ResolvedAutoBranching,
) {
    config.use_multi_objective_critical_kfsb = resolved.use_multi_objective_critical_kfsb;
}

/// Model-CLASS-aware auto-selection of the branching method.
///
/// Picks the best sound search strategy from the model's STRUCTURE, subsuming the
/// per-category preset routing that previously lived in `vnncomp_scripts/
/// run_instance.sh` and in per-benchmark preset `bab.branching.method`.
///
/// Decision order (first match wins):
///
/// 1. **MIP complete verifier active** => [`BranchingHeuristic::Kfsb`]
///    ([`AutoBranchingReason::MipComplete`]). On SAT-encoded / NLP / malware-Conv
///    categories, CROWN is too loose and MIP does the real work; the BaB pass is a
///    fallback where ReLU splitting is correct and input splitting is hopeless.
///
/// 2. **Low-dim input** (`input_element_count <= INPUT_SPLIT_MAX_INPUT_DIM`) =>
///    [`BranchingHeuristic::InputSplit`] ([`AutoBranchingReason::LowDimInput`]).
///    ACAS-Xu (5), cersyve (4), lsnc (6), nn4sys (1), TLLVerifyBench (2),
///    linearizenn (4), cgan latent (5), sat_relu (30) all have tiny inputs and a
///    small unstable frontier; tightening the input box directly wins.
///
/// 3. **Moderate-dim input over a small/shallow net** (`input_element_count <=
///    MODERATE_INPUT_SPLIT_MAX_INPUT_DIM` AND `param_count <= MODERATE_NET_MAX_PARAMS`
///    AND `relu_node_count <= MODERATE_NET_MAX_RELU`) =>
///    [`BranchingHeuristic::InputSplit`] ([`AutoBranchingReason::SmallShallowNet`]).
///    This is the dist_shift class: 792 inputs over a small conv/FC autoencoder. The
///    net is small enough that the input box can be fanned out and the
///    few-unstable-neuron net converges. Genuine CNNs in the same input band
///    (collins_rul) carry far more params / ReLU nodes and fall through to rule 4.
///    Requires the structural signals; when they are absent (pre-build) this rule is
///    skipped and a moderate-dim input falls through to rule 4 (conservative: ReLU
///    splitting is always sound, just possibly slower).
///
/// 4. **Otherwise** => [`BranchingHeuristic::Kfsb`]
///    ([`AutoBranchingReason::HighDimOrManyRelu`]). High-dimensional image /
///    transformer inputs (CIFAR 3072, ViT 3072, TinyImageNet 9408, traffic_signs
///    12288, VGGNet 150528, yolo 8112) and moderate-dim-but-large nets have a huge
///    unstable-ReLU frontier where input splitting is hopeless and ReLU (kFSB)
///    splitting is the only tractable choice.
///
/// SOUNDNESS: every branching method is sound — this only selects which sound
/// search strategy runs, never changing a verdict.
pub(crate) fn auto_select_branching(
    signals: ModelClassSignals,
) -> (BranchingHeuristic, AutoBranchingReason) {
    if signals.mip_complete_verifier {
        return (BranchingHeuristic::Kfsb, AutoBranchingReason::MipComplete);
    }
    if signals.input_element_count <= INPUT_SPLIT_MAX_INPUT_DIM {
        return (
            BranchingHeuristic::InputSplit,
            AutoBranchingReason::LowDimInput,
        );
    }
    // Moderate-dim band: input splitting only if the net is structurally small.
    if signals.input_element_count <= MODERATE_INPUT_SPLIT_MAX_INPUT_DIM {
        if let Some(structure) = signals.structure {
            if structure.param_count <= MODERATE_NET_MAX_PARAMS
                && structure.relu_node_count <= MODERATE_NET_MAX_RELU
            {
                return (
                    BranchingHeuristic::InputSplit,
                    AutoBranchingReason::SmallShallowNet,
                );
            }
        }
    }
    (
        BranchingHeuristic::Kfsb,
        AutoBranchingReason::HighDimOrManyRelu,
    )
}

/// Parse a base branching token (no relu).
///
/// Accepted tokens: width, impact, babsr, fsb, kfsb, kfsb-intercept-only, sequential, input.
pub(crate) fn parse_branching_heuristic(s: &str) -> Result<BranchingHeuristic> {
    match s {
        "width" => Ok(BranchingHeuristic::LargestBoundWidth),
        "impact" | "babsr" => Ok(BranchingHeuristic::BoundImpact),
        "fsb" => Ok(BranchingHeuristic::FilteredSmartBranching),
        "kfsb" => Ok(BranchingHeuristic::Kfsb),
        "kfsb-intercept-only" => Ok(BranchingHeuristic::KfsbInterceptOnly),
        "sequential" => Ok(BranchingHeuristic::Sequential),
        "input" => Ok(BranchingHeuristic::InputSplit),
        _ => anyhow::bail!(
            "Unknown branching heuristic: '{}'. Use: width, impact, babsr, fsb, kfsb, kfsb-intercept-only, sequential, or input",
            s
        ),
    }
}

/// Returns `true` when the CLI branching token requests auto-selection.
///
/// `auto` is the default value of `--branching`. It is treated like "no explicit
/// CLI heuristic" for preset precedence (so a preset's `bab.branching.method`
/// keeps control), then resolved via [`auto_select_branching`] once the model +
/// spec are loaded.
pub(crate) fn is_auto_branching(branching: Option<&str>) -> bool {
    branching == Some(AUTO_TOKEN)
}

/// Parse a branching token including the "relu" and "auto" shorthands.
///
/// "relu" maps to `LargestBoundWidth` with `use_relu_split = true`.
/// "auto" defers like `None` (resolved later by [`auto_select_branching`]).
/// All other tokens delegate to [`parse_branching_heuristic`].
///
/// Returns `(None, false)` when `branching` is `None` or `auto` (defer to preset
/// / auto-selection).
pub(crate) fn parse_branching_with_relu(
    branching: Option<&str>,
) -> Result<(Option<BranchingHeuristic>, bool)> {
    match branching {
        None => Ok((None, false)),
        Some(AUTO_TOKEN) => Ok((None, false)),
        Some(RELU_TOKEN) => Ok((Some(BranchingHeuristic::LargestBoundWidth), true)),
        Some(s) => {
            let h = parse_branching_heuristic(s)?;
            let relu = matches!(
                h,
                BranchingHeuristic::Kfsb
                    | BranchingHeuristic::KfsbInterceptOnly
                    | BranchingHeuristic::FilteredSmartBranching
                    | BranchingHeuristic::BoundImpact
            );
            Ok((Some(h), relu))
        }
    }
}

/// Validate that `heuristic` is compatible with GPU BaB.
///
/// GPU BaB supports BaBSR (`impact`/`babsr`) and input splitting (`input`).
/// Returns an error containing the rejected token for other heuristics.
pub(crate) fn validate_gpu_bab_branching(
    heuristic: &BranchingHeuristic,
    branching_str: &str,
) -> Result<()> {
    if !matches!(
        heuristic,
        BranchingHeuristic::BoundImpact | BranchingHeuristic::InputSplit
    ) {
        anyhow::bail!(
            "--gpu-bab supports --branching=impact (alias: babsr) or --branching=input; got --branching={branching_str}."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- base token parity ----

    #[test]
    fn base_tokens_all_parse_successfully() {
        for token in BASE_TOKENS {
            let result = parse_branching_heuristic(token);
            assert!(
                result.is_ok(),
                "BASE_TOKENS entry '{token}' should parse, got: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn base_token_mapping_is_correct() {
        assert!(matches!(
            parse_branching_heuristic("width").unwrap(),
            BranchingHeuristic::LargestBoundWidth
        ));
        assert!(matches!(
            parse_branching_heuristic("impact").unwrap(),
            BranchingHeuristic::BoundImpact
        ));
        assert!(matches!(
            parse_branching_heuristic("babsr").unwrap(),
            BranchingHeuristic::BoundImpact
        ));
        assert!(matches!(
            parse_branching_heuristic("fsb").unwrap(),
            BranchingHeuristic::FilteredSmartBranching
        ));
        assert!(matches!(
            parse_branching_heuristic("kfsb").unwrap(),
            BranchingHeuristic::Kfsb
        ));
        assert!(matches!(
            parse_branching_heuristic("kfsb-intercept-only").unwrap(),
            BranchingHeuristic::KfsbInterceptOnly
        ));
        assert!(matches!(
            parse_branching_heuristic("sequential").unwrap(),
            BranchingHeuristic::Sequential
        ));
        assert!(matches!(
            parse_branching_heuristic("input").unwrap(),
            BranchingHeuristic::InputSplit
        ));
    }

    #[test]
    fn unknown_token_is_rejected() {
        assert!(parse_branching_heuristic("unknown").is_err());
        assert!(parse_branching_heuristic("").is_err());
        // "relu" is NOT a base token — it is extended
        assert!(parse_branching_heuristic("relu").is_err());
    }

    #[test]
    fn fsb_and_kfsb_map_to_distinct_heuristics() {
        assert!(matches!(
            parse_branching_heuristic("fsb").unwrap(),
            BranchingHeuristic::FilteredSmartBranching
        ));
        assert!(matches!(
            parse_branching_heuristic("kfsb").unwrap(),
            BranchingHeuristic::Kfsb
        ));
    }

    // ---- relu extension ----

    #[test]
    fn relu_token_enables_relu_split() {
        let (heuristic, relu) = parse_branching_with_relu(Some("relu")).unwrap();
        assert!(matches!(
            heuristic.unwrap(),
            BranchingHeuristic::LargestBoundWidth
        ));
        assert!(relu, "relu token should set use_relu_split");
    }

    #[test]
    fn relu_splitting_heuristics_set_relu_flag() {
        let relu_tokens = ["kfsb", "kfsb-intercept-only", "fsb", "impact", "babsr"];
        for token in &relu_tokens {
            let (h, relu) = parse_branching_with_relu(Some(token)).unwrap();
            assert!(h.is_some(), "should parse '{token}'");
            assert!(relu, "ReLU-splitting token '{token}' should set relu flag");
        }
    }

    #[test]
    fn non_relu_splitting_tokens_do_not_set_relu_flag() {
        let non_relu_tokens = ["width", "sequential", "input"];
        for token in &non_relu_tokens {
            let (h, relu) = parse_branching_with_relu(Some(token)).unwrap();
            assert!(h.is_some(), "should parse '{token}'");
            assert!(!relu, "non-ReLU token '{token}' should not set relu flag");
        }
    }

    #[test]
    fn none_input_defers_to_preset() {
        let (h, relu) = parse_branching_with_relu(None).unwrap();
        assert!(h.is_none());
        assert!(!relu);
    }

    // ---- GPU BaB gate ----

    #[test]
    fn gpu_bab_accepts_bound_impact() {
        assert!(validate_gpu_bab_branching(&BranchingHeuristic::BoundImpact, "impact").is_ok());
        assert!(validate_gpu_bab_branching(&BranchingHeuristic::BoundImpact, "babsr").is_ok());
    }

    #[test]
    fn gpu_bab_accepts_input_split() {
        assert!(validate_gpu_bab_branching(&BranchingHeuristic::InputSplit, "input").is_ok());
    }

    #[test]
    fn gpu_bab_rejects_unsupported() {
        let unsupported = [
            (BranchingHeuristic::LargestBoundWidth, "width"),
            (BranchingHeuristic::FilteredSmartBranching, "fsb"),
            (BranchingHeuristic::Sequential, "sequential"),
        ];
        for (h, name) in &unsupported {
            let err = validate_gpu_bab_branching(h, name)
                .expect_err(&format!("gpu-bab should reject '{name}'"));
            let msg = err.to_string();
            assert!(
                msg.contains("--gpu-bab supports"),
                "Expected gpu-bab compatibility error, got: {msg}"
            );
            assert!(
                msg.contains(name),
                "Expected rejected token '{name}' in error, got: {msg}"
            );
        }
    }

    // ---- contract: bench_acasxu token parity ----

    /// Ensures every token accepted by `bench_acasxu` (the BASE_TOKENS set)
    /// is also accepted by `parse_branching_with_relu`.  If bench_acasxu adds
    /// a token, it must appear in BASE_TOKENS here or this test fails.
    #[test]
    fn bench_acasxu_tokens_are_subset_of_base_tokens() {
        // bench_acasxu previously accepted exactly these 7 tokens
        let bench_tokens = [
            "width",
            "impact",
            "babsr",
            "fsb",
            "kfsb",
            "sequential",
            "input",
        ];
        for token in &bench_tokens {
            assert!(
                BASE_TOKENS.contains(token),
                "bench_acasxu token '{token}' missing from BASE_TOKENS"
            );
        }
    }

    /// Ensures the beta_crown verify command accepts all BASE_TOKENS plus "relu".
    #[test]
    fn beta_crown_verify_accepts_all_base_plus_relu() {
        for token in BASE_TOKENS {
            assert!(
                parse_branching_with_relu(Some(token)).is_ok(),
                "beta_crown verify should accept base token '{token}'"
            );
        }
        assert!(
            parse_branching_with_relu(Some(RELU_TOKEN)).is_ok(),
            "beta_crown verify should accept 'relu'"
        );
    }

    // ---- auto token + auto-selection ----

    #[test]
    fn auto_token_is_recognized() {
        assert!(is_auto_branching(Some("auto")));
        assert!(!is_auto_branching(Some("input")));
        assert!(!is_auto_branching(Some("kfsb")));
        assert!(!is_auto_branching(None));
    }

    #[test]
    fn auto_token_defers_like_none() {
        // "auto" must parse without error and defer (None heuristic, no relu),
        // exactly like `None`, so preset branching keeps precedence.
        let (h, relu) = parse_branching_with_relu(Some("auto")).unwrap();
        assert!(
            h.is_none(),
            "auto should not pin a heuristic pre-resolution"
        );
        assert!(!relu, "auto should not set the relu flag");
        let (hn, relun) = parse_branching_with_relu(None).unwrap();
        assert_eq!(h, hn);
        assert_eq!(relu, relun);
    }

    #[test]
    fn auto_token_rejected_by_base_parser_accepted_by_relu_parser() {
        // The base parser does not know "auto" (it is an extended token, like relu).
        assert!(parse_branching_heuristic("auto").is_err());
        // The relu-aware parser accepts it (deferring).
        assert!(parse_branching_with_relu(Some("auto")).is_ok());
    }

    /// Low-dimensional inputs (ACAS-Xu 5, TLL 2, control state 4-6, nn4sys 1,
    /// sat_relu 30, cgan latent 5) select input splitting regardless of structure.
    #[test]
    fn auto_select_low_dim_input_picks_input_split() {
        for dim in [1usize, 2, 4, 5, 6, 30, 64] {
            // Both with and without structure: low-dim short-circuits before the
            // structural rule, so the result must not depend on it.
            let no_struct = auto_select_branching(ModelClassSignals::dim_only(dim, false));
            assert!(
                matches!(no_struct.0, BranchingHeuristic::InputSplit),
                "input_dim={dim} (non-MIP, no structure) should select InputSplit, got {:?}",
                no_struct.0
            );
            assert_eq!(no_struct.1, AutoBranchingReason::LowDimInput);

            let with_huge_net = auto_select_branching(ModelClassSignals {
                input_element_count: dim,
                mip_complete_verifier: false,
                structure: Some(ModelStructure {
                    param_count: 100_000_000,
                    has_conv: true,
                    relu_node_count: 10_000,
                    is_dag: true,
                }),
            });
            assert!(
                matches!(with_huge_net.0, BranchingHeuristic::InputSplit),
                "input_dim={dim} (low-dim) must pick InputSplit even with a huge net"
            );
            assert_eq!(with_huge_net.1, AutoBranchingReason::LowDimInput);
        }
    }

    /// High-dimensional image / transformer inputs (CIFAR 3072, TinyImageNet 9408,
    /// traffic_signs 12288, vggnet 150528, yolo 8112) select ReLU/kFSB splitting
    /// no matter how small the net's parameter count looks — they sit above the
    /// moderate-dim band entirely.
    #[test]
    fn auto_select_high_dim_input_picks_kfsb() {
        for dim in [2049usize, 3072, 4096, 8112, 9408, 12288, 150528] {
            // Even a (hypothetically) tiny net at this input dim must pick kFSB:
            // the input box is too high-dimensional to fan out.
            let (h, reason) = auto_select_branching(ModelClassSignals {
                input_element_count: dim,
                mip_complete_verifier: false,
                structure: Some(ModelStructure {
                    param_count: 1,
                    has_conv: false,
                    relu_node_count: 1,
                    is_dag: false,
                }),
            });
            assert!(
                matches!(h, BranchingHeuristic::Kfsb),
                "input_dim={dim} (non-MIP) should select Kfsb, got {h:?}"
            );
            assert_eq!(reason, AutoBranchingReason::HighDimOrManyRelu);
        }
    }

    /// The dist_shift class: moderate input (792) over a SMALL/shallow conv/FC
    /// autoencoder selects input splitting via the structural rule.
    #[test]
    fn auto_select_small_shallow_moderate_dim_picks_input_split() {
        let (h, reason) = auto_select_branching(ModelClassSignals {
            input_element_count: 792,
            mip_complete_verifier: false,
            structure: Some(ModelStructure {
                param_count: 60_000, // small MNIST autoencoder
                has_conv: true,
                relu_node_count: 6,
                is_dag: true, // Concat -> graph
            }),
        });
        assert!(
            matches!(h, BranchingHeuristic::InputSplit),
            "dist_shift class (792, small conv autoencoder) should pick InputSplit, got {h:?}"
        );
        assert_eq!(reason, AutoBranchingReason::SmallShallowNet);
    }

    /// A moderate-dim CNN that is NOT small/shallow (collins_rul class: 400 inputs
    /// but many params / ReLU nodes) must fall through to kFSB — the widened input
    /// regime must not misroute it.
    #[test]
    fn auto_select_moderate_dim_large_net_picks_kfsb() {
        // Exceed params only.
        let big_params = auto_select_branching(ModelClassSignals {
            input_element_count: 400,
            mip_complete_verifier: false,
            structure: Some(ModelStructure {
                param_count: MODERATE_NET_MAX_PARAMS + 1,
                has_conv: true,
                relu_node_count: 8,
                is_dag: false,
            }),
        });
        assert!(
            matches!(big_params.0, BranchingHeuristic::Kfsb),
            "moderate-dim net over param budget should pick Kfsb"
        );
        assert_eq!(big_params.1, AutoBranchingReason::HighDimOrManyRelu);

        // Exceed ReLU node count only (deep net).
        let deep = auto_select_branching(ModelClassSignals {
            input_element_count: 400,
            mip_complete_verifier: false,
            structure: Some(ModelStructure {
                param_count: 50_000,
                has_conv: true,
                relu_node_count: MODERATE_NET_MAX_RELU + 1,
                is_dag: false,
            }),
        });
        assert!(
            matches!(deep.0, BranchingHeuristic::Kfsb),
            "moderate-dim deep net (many ReLU nodes) should pick Kfsb"
        );
        assert_eq!(deep.1, AutoBranchingReason::HighDimOrManyRelu);
    }

    /// Without structural signals (pre-build, e.g. epsilon-ball mode), a moderate-dim
    /// input conservatively falls through to kFSB: the structural rule cannot fire,
    /// and ReLU splitting is always sound (just possibly slower). Low-dim still works.
    #[test]
    fn auto_select_moderate_dim_without_structure_falls_through_to_kfsb() {
        let (h, reason) = auto_select_branching(ModelClassSignals::dim_only(792, false));
        assert!(
            matches!(h, BranchingHeuristic::Kfsb),
            "moderate-dim with no structure should conservatively pick Kfsb, got {h:?}"
        );
        assert_eq!(reason, AutoBranchingReason::HighDimOrManyRelu);
    }

    /// MIP complete-verifier categories (safenlp 30, malbeware 4096) always select
    /// kFSB regardless of input dimensionality OR structure: input splitting is
    /// hopeless on NLP / malware nets and MIP does the real work.
    #[test]
    fn auto_select_mip_complete_always_picks_kfsb() {
        for dim in [30usize, 5, 792, 4096, 100_000] {
            // MIP short-circuits before any input/structure rule.
            let (h, reason) = auto_select_branching(ModelClassSignals {
                input_element_count: dim,
                mip_complete_verifier: true,
                structure: Some(ModelStructure {
                    param_count: 10,
                    has_conv: false,
                    relu_node_count: 2,
                    is_dag: false,
                }),
            });
            assert!(
                matches!(h, BranchingHeuristic::Kfsb),
                "input_dim={dim} (MIP) should select Kfsb, got {h:?}"
            );
            assert_eq!(reason, AutoBranchingReason::MipComplete);
        }
    }

    /// Exhaustive check that the model-class-aware heuristic reproduces the
    /// ground-truth per-category routing using each category's (input_count,
    /// has_conv, param_count, relu_node_count, mip) signature.
    ///
    /// Categories kept on their preset branching (genuine exceptions the heuristic
    /// cannot derive without the model) are noted but still asserted where the
    /// signature is unambiguous:
    ///   * sat_relu: ground-truth INPUT but MIP-routed; its preset pins
    ///     `branching: input` so auto never fires. Asserted here ONLY in the
    ///     non-MIP form to document the intended class; the live path keeps the
    ///     preset.
    ///   * collins_rul: kept on preset `kfsb` (moderate-dim CNN); asserted with a
    ///     representative large-CNN signature so the heuristic agrees.
    #[test]
    fn auto_select_matches_ground_truth_routing_table() {
        struct Row {
            name: &'static str,
            dim: usize,
            has_conv: bool,
            params: usize,
            relus: usize,
            mip: bool,
            expect_input: bool,
        }
        let table = [
            // ---- INPUT-SPLIT (low-dim) ----
            Row {
                name: "acasxu",
                dim: 5,
                has_conv: false,
                params: 13_000,
                relus: 6,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "cersyve",
                dim: 4,
                has_conv: false,
                params: 5_000,
                relus: 4,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "lsnc",
                dim: 6,
                has_conv: false,
                params: 8_000,
                relus: 4,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "nn4sys",
                dim: 1,
                has_conv: false,
                params: 2_000,
                relus: 3,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "tllverifybench",
                dim: 2,
                has_conv: false,
                params: 4_000,
                relus: 4,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "linearizenn",
                dim: 4,
                has_conv: false,
                params: 6_000,
                relus: 4,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "cgan",
                dim: 5,
                has_conv: true,
                params: 500_000,
                relus: 8,
                mip: false,
                expect_input: true,
            },
            Row {
                name: "ml4acopf",
                dim: 22,
                has_conv: false,
                params: 100_000,
                relus: 8,
                mip: false,
                expect_input: true,
            },
            // sat_relu ground-truth class (non-MIP form; live path keeps preset).
            Row {
                name: "sat_relu",
                dim: 30,
                has_conv: false,
                params: 3_000,
                relus: 4,
                mip: false,
                expect_input: true,
            },
            // ---- INPUT-SPLIT (moderate-dim, small/shallow): the dist_shift class ----
            Row {
                name: "dist_shift",
                dim: 792,
                has_conv: true,
                params: 60_000,
                relus: 6,
                mip: false,
                expect_input: true,
            },
            // ---- RELU-SPLIT (high-dim) ----
            Row {
                name: "cifar100",
                dim: 3072,
                has_conv: true,
                params: 2_000_000,
                relus: 20,
                mip: false,
                expect_input: false,
            },
            Row {
                name: "tinyimagenet",
                dim: 9408,
                has_conv: true,
                params: 10_000_000,
                relus: 30,
                mip: false,
                expect_input: false,
            },
            Row {
                name: "yolo",
                dim: 8112,
                has_conv: true,
                params: 5_000_000,
                relus: 40,
                mip: false,
                expect_input: false,
            },
            Row {
                name: "vggnet16",
                dim: 150528,
                has_conv: true,
                params: 138_000_000,
                relus: 16,
                mip: false,
                expect_input: false,
            },
            Row {
                name: "traffic_signs",
                dim: 12288,
                has_conv: true,
                params: 3_000_000,
                relus: 12,
                mip: false,
                expect_input: false,
            },
            Row {
                name: "vit",
                dim: 3072,
                has_conv: false,
                params: 20_000_000,
                relus: 12,
                mip: false,
                expect_input: false,
            },
            // ---- RELU-SPLIT (moderate-dim CNN, kept on preset; large signature) ----
            Row {
                name: "collins_rul",
                dim: 400,
                has_conv: true,
                params: 5_000_000,
                relus: 8,
                mip: false,
                expect_input: false,
            },
            // ---- RELU-SPLIT (MIP complete-verifier) ----
            Row {
                name: "safenlp",
                dim: 30,
                has_conv: false,
                params: 50_000,
                relus: 4,
                mip: true,
                expect_input: false,
            },
            Row {
                name: "malbeware",
                dim: 4096,
                has_conv: true,
                params: 1_000_000,
                relus: 4,
                mip: true,
                expect_input: false,
            },
        ];
        for row in table {
            let (h, _reason) = auto_select_branching(ModelClassSignals {
                input_element_count: row.dim,
                mip_complete_verifier: row.mip,
                structure: Some(ModelStructure {
                    param_count: row.params,
                    has_conv: row.has_conv,
                    relu_node_count: row.relus,
                    is_dag: false,
                }),
            });
            let got_input = matches!(h, BranchingHeuristic::InputSplit);
            assert_eq!(
                got_input, row.expect_input,
                "category '{}' (dim={}, conv={}, params={}, relus={}, mip={}): expected input_split={}, got {h:?}",
                row.name, row.dim, row.has_conv, row.params, row.relus, row.mip, row.expect_input
            );
        }
    }
}
