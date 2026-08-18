// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize};

/// Soundness interpretation for a verification run.
///
/// This is a *label* surfaced to downstream consumers. It does not prove soundness; it records
/// whether the run used any known heuristics/approximations that weaken proof semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationSoundnessMode {
    /// No known heuristic/unsound switches were enabled.
    Sound,
    /// At least one heuristic/approximation that weakens proof semantics was used.
    Heuristic,
}

/// Heuristics/approximations used by a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeuristicUsed {
    /// LayerNorm IBP "forward mode" uses the center point for mean/std computation.
    LayerNormForwardMode { num_nodes: usize },
    /// RMSNorm IBP "forward mode" uses the center point for RMS computation.
    RmsNormForwardMode { num_nodes: usize },
    /// GroupNorm IBP "forward mode" uses the center point for grouped mean/std computation.
    GroupNormForwardMode { num_nodes: usize },
    /// InstanceNorm IBP "forward mode" uses the center point for per-channel mean/std computation.
    InstanceNormForwardMode { num_nodes: usize },
    /// AdaIN IBP "forward mode" uses the center point through its inner InstanceNorm.
    AdaInForwardMode { num_nodes: usize },
    /// LayerNorm CROWN sampling mode uses heuristic sampling-based linearization.
    LayerNormCrownSampling { num_nodes: usize },
    /// Softmax CROWN used sampling-based relaxations.
    SoftmaxCrownSampling { num_nodes: usize },
    /// CausalSoftmax CROWN used sampling-based relaxations.
    CausalSoftmaxCrownSampling { num_nodes: usize },
    /// LogSoftmax CROWN used sampling-based relaxations.
    LogSoftmaxCrownSampling { num_nodes: usize },
    /// One or more nonlinear CROWN relaxations are sampling-inflated (coarse-grained marker).
    SamplingBasedNonlinearRelaxations,
    /// ReduceMax/ReduceMin CROWN uses the center-point argmax/argmin as a fixed index.
    /// alpha-beta-CROWN only implements this path when the extremum index is assumed stable;
    /// perturbed extrema are not handled exactly.
    ReduceExtremumFixedIndex { num_nodes: usize },
    /// Sqrt encountered negative-domain inputs (x < 0) during propagation.
    SqrtNegativeDomain { num_nodes: usize },
    /// Compare node uses continuous approximation (subtraction/abs) instead of discrete boolean.
    /// Supports graph translation of Gt/Ge/Lt/Le/Eq/Ne comparison nodes.
    ContinuousComparisonApproximation { num_nodes: usize },
}

/// Machine-readable provenance for verification soundness semantics.
#[derive(Debug, Clone, Serialize)]
pub struct SoundnessProvenance {
    pub(crate) mode: VerificationSoundnessMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) heuristics_used: Vec<HeuristicUsed>,
}

/// Deserialize provenance fail-closed: an explicit heuristic entry can never be
/// under-labeled as [`VerificationSoundnessMode::Sound`] by inconsistent input.
impl<'de> Deserialize<'de> for SoundnessProvenance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct SoundnessProvenanceRaw {
            mode: VerificationSoundnessMode,
            #[serde(default)]
            heuristics_used: Vec<HeuristicUsed>,
        }

        let raw = SoundnessProvenanceRaw::deserialize(deserializer)?;
        let mode = if raw.heuristics_used.is_empty() {
            raw.mode
        } else {
            VerificationSoundnessMode::Heuristic
        };
        Ok(Self {
            mode,
            heuristics_used: raw.heuristics_used,
        })
    }
}

impl SoundnessProvenance {
    /// Create a provenance indicating a sound run (no heuristics).
    pub fn sound() -> Self {
        Self {
            mode: VerificationSoundnessMode::Sound,
            heuristics_used: vec![],
        }
    }

    /// Create a provenance whose mode is [`VerificationSoundnessMode::Heuristic`]
    /// but which carries no enumerated [`HeuristicUsed`] entry.
    ///
    /// This is for an approximation that legitimately weakens *exactness* (so the
    /// run should be labeled `Heuristic`) yet has no dedicated `HeuristicUsed`
    /// variant to record — for example the P8 precision-widening post-pass, whose
    /// dedicated variant is a documented follow-on. Combining this with another
    /// provenance via [`combine`](Self::combine) propagates the `Heuristic` label
    /// (see `combine`'s explicit-mode rule).
    ///
    /// NOTE: precision widening is a SOUND over-approximation; the `Heuristic`
    /// label here flags that the verdict was produced under a modeling
    /// approximation, not that soundness was lost.
    pub fn heuristic() -> Self {
        Self {
            mode: VerificationSoundnessMode::Heuristic,
            heuristics_used: vec![],
        }
    }

    /// Create provenance from heuristics list.
    /// Mode is automatically determined: Sound if empty, Heuristic otherwise.
    pub fn from_heuristics(heuristics_used: Vec<HeuristicUsed>) -> Self {
        let mode = if heuristics_used.is_empty() {
            VerificationSoundnessMode::Sound
        } else {
            VerificationSoundnessMode::Heuristic
        };
        Self {
            mode,
            heuristics_used,
        }
    }

    /// Get the soundness mode.
    pub fn mode(&self) -> VerificationSoundnessMode {
        self.mode
    }

    /// Get the recorded heuristics.
    pub fn heuristics_used(&self) -> &[HeuristicUsed] {
        &self.heuristics_used
    }

    /// Merge two provenances into one.
    ///
    /// Heuristics from `self` are followed by those from `other` (order preserved, both kept).
    /// The merged mode is `Heuristic` if either input is `Heuristic` or the merged heuristic
    /// list is non-empty, else `Sound`. Routed through [`from_heuristics`] so the mode invariant
    /// holds; the `self`/`other` mode check additionally honors an explicit `Heuristic` label even
    /// when no heuristics are listed.
    ///
    /// [`from_heuristics`]: Self::from_heuristics
    pub fn combine(&self, other: &Self) -> Self {
        let mut heuristics_used =
            Vec::with_capacity(self.heuristics_used.len() + other.heuristics_used.len());
        heuristics_used.extend(self.heuristics_used.iter().cloned());
        heuristics_used.extend(other.heuristics_used.iter().cloned());
        let mut combined = Self::from_heuristics(heuristics_used);
        if self.mode == VerificationSoundnessMode::Heuristic
            || other.mode == VerificationSoundnessMode::Heuristic
        {
            combined.mode = VerificationSoundnessMode::Heuristic;
        }
        combined
    }

    /// Fold [`combine`] over an iterator of provenances, starting from [`sound`].
    ///
    /// [`combine`]: Self::combine
    /// [`sound`]: Self::sound
    pub fn combine_all<'a>(items: impl IntoIterator<Item = &'a SoundnessProvenance>) -> Self {
        items
            .into_iter()
            .fold(Self::sound(), |acc, item| acc.combine(item))
    }
}

impl Default for SoundnessProvenance {
    fn default() -> Self {
        Self::sound()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heuristic(num_nodes: usize) -> SoundnessProvenance {
        SoundnessProvenance::from_heuristics(vec![HeuristicUsed::LayerNormForwardMode {
            num_nodes,
        }])
    }

    #[test]
    fn combine_sound_with_sound_is_sound() {
        let combined = SoundnessProvenance::sound().combine(&SoundnessProvenance::sound());
        assert_eq!(combined.mode(), VerificationSoundnessMode::Sound);
        assert!(combined.heuristics_used().is_empty());
    }

    #[test]
    fn combine_sound_with_heuristic_is_heuristic() {
        let combined = SoundnessProvenance::sound().combine(&heuristic(1));
        assert_eq!(combined.mode(), VerificationSoundnessMode::Heuristic);
        assert_eq!(combined.heuristics_used().len(), 1);
    }

    #[test]
    fn combine_concatenates_heuristics_in_order() {
        let combined = heuristic(1).combine(&heuristic(2));
        assert_eq!(combined.mode(), VerificationSoundnessMode::Heuristic);
        assert_eq!(
            combined.heuristics_used(),
            &[
                HeuristicUsed::LayerNormForwardMode { num_nodes: 1 },
                HeuristicUsed::LayerNormForwardMode { num_nodes: 2 },
            ]
        );
    }

    #[test]
    fn combine_all_over_mixed_is_heuristic_with_all() {
        let s0 = SoundnessProvenance::sound();
        let h1 = heuristic(1);
        let s2 = SoundnessProvenance::sound();
        let h3 = heuristic(3);
        let combined = SoundnessProvenance::combine_all([&s0, &h1, &s2, &h3]);
        assert_eq!(combined.mode(), VerificationSoundnessMode::Heuristic);
        assert_eq!(
            combined.heuristics_used(),
            &[
                HeuristicUsed::LayerNormForwardMode { num_nodes: 1 },
                HeuristicUsed::LayerNormForwardMode { num_nodes: 3 },
            ]
        );
    }

    #[test]
    fn heuristic_constructor_sets_mode_without_listing() {
        let p = SoundnessProvenance::heuristic();
        assert_eq!(p.mode(), VerificationSoundnessMode::Heuristic);
        assert!(p.heuristics_used().is_empty());
    }

    #[test]
    fn combine_propagates_heuristic_label_from_empty_marker() {
        // A sound run combined with the no-detail heuristic marker becomes
        // Heuristic, even though no HeuristicUsed entries are listed.
        let combined = SoundnessProvenance::sound().combine(&SoundnessProvenance::heuristic());
        assert_eq!(combined.mode(), VerificationSoundnessMode::Heuristic);
        assert!(combined.heuristics_used().is_empty());
        // Symmetric.
        let combined2 = SoundnessProvenance::heuristic().combine(&SoundnessProvenance::sound());
        assert_eq!(combined2.mode(), VerificationSoundnessMode::Heuristic);
    }

    #[test]
    fn combine_all_empty_is_sound() {
        let combined = SoundnessProvenance::combine_all(std::iter::empty());
        assert_eq!(combined.mode(), VerificationSoundnessMode::Sound);
        assert!(combined.heuristics_used().is_empty());
    }

    #[test]
    fn deserialize_cannot_underlabel_explicit_heuristics_as_sound() {
        let json = r#"{
            "mode": "sound",
            "heuristics_used": [
                {"type": "layer_norm_forward_mode", "num_nodes": 1}
            ]
        }"#;
        let provenance: SoundnessProvenance =
            serde_json::from_str(json).expect("valid provenance JSON");
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert_eq!(provenance.heuristics_used().len(), 1);
    }

    #[test]
    fn deserialize_preserves_explicit_empty_heuristic_marker() {
        let json = r#"{"mode":"heuristic"}"#;
        let provenance: SoundnessProvenance =
            serde_json::from_str(json).expect("valid provenance JSON");
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(provenance.heuristics_used().is_empty());
    }
}
