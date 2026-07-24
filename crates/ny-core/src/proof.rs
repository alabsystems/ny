// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Proof certificate for verified (UNSAT) results.
///
/// When verification succeeds, the solver can optionally produce a proof
/// that the property holds. This proof can be exported in various formats
/// for independent verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProof {
    /// Proof format identifier.
    format: ProofFormat,
    /// Raw proof bytes (format-specific encoding).
    data: Vec<u8>,
    /// Number of proof steps (if available).
    num_steps: Option<usize>,
    /// Summary of proof statistics.
    stats: Option<ProofStats>,
}

/// Proof format for UNSAT certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofFormat {
    /// Alethe format (SMT-LIB extension, checkable by carcara).
    Alethe,
    /// LFSC (Logical Framework with Side Conditions).
    Lfsc,
    /// DRAT (Delete Resolution Asymmetric Tautology) - for SAT.
    Drat,
    /// Custom format for bound propagation proofs.
    BoundTrace,
}

/// Statistics about the proof.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofStats {
    /// Number of assumptions (input assertions).
    num_assumptions: usize,
    /// Number of resolution steps.
    num_resolutions: usize,
    /// Number of theory lemmas.
    num_theory_lemmas: usize,
    /// Total proof size in bytes.
    pub(crate) size_bytes: usize,
}

impl ProofStats {
    /// Create proof statistics from individual counts.
    pub fn new(
        num_assumptions: usize,
        num_resolutions: usize,
        num_theory_lemmas: usize,
        size_bytes: usize,
    ) -> Self {
        Self {
            num_assumptions,
            num_resolutions,
            num_theory_lemmas,
            size_bytes,
        }
    }

    /// Number of assumptions (input assertions).
    #[inline]
    pub fn num_assumptions(&self) -> usize {
        self.num_assumptions
    }

    /// Number of resolution steps.
    #[inline]
    pub fn num_resolutions(&self) -> usize {
        self.num_resolutions
    }

    /// Number of theory lemmas.
    #[inline]
    pub fn num_theory_lemmas(&self) -> usize {
        self.num_theory_lemmas
    }

    /// Total proof size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl VerificationProof {
    /// Create a proof from raw parts.
    ///
    /// If stats are provided, `size_bytes` is normalized to `data.len()`.
    pub fn from_parts(
        format: ProofFormat,
        data: Vec<u8>,
        num_steps: Option<usize>,
        stats: Option<ProofStats>,
    ) -> Self {
        let stats = stats.map(|proof_stats| ProofStats {
            size_bytes: data.len(),
            ..proof_stats
        });
        Self {
            format,
            data,
            num_steps,
            stats,
        }
    }

    /// Create a new Alethe format proof.
    ///
    /// # REQUIRES
    /// - `proof_text` is valid UTF-8 (guaranteed by `String`)
    ///
    /// # ENSURES
    /// - `result.format == ProofFormat::Alethe`
    /// - `result.as_text() == Some(proof_text.as_str())`
    /// - `result.stats.as_ref().map(|s| s.size_bytes) == Some(result.data.len())`
    pub fn alethe(proof_text: String) -> Self {
        let data = proof_text.into_bytes();
        Self::from_parts(ProofFormat::Alethe, data, None, Some(ProofStats::default()))
    }

    /// Create an Alethe proof with statistics.
    ///
    /// # REQUIRES
    /// - `proof_text` is valid UTF-8 (guaranteed by `String`)
    ///
    /// # ENSURES
    /// - `result.format == ProofFormat::Alethe`
    /// - `result.num_steps == Some(num_steps)`
    /// - `result.as_text().is_some()`
    /// - `result.stats.as_ref().map(|s| s.size_bytes) == Some(result.data.len())`
    pub fn alethe_with_stats(proof_text: String, num_steps: usize, stats: ProofStats) -> Self {
        let data = proof_text.into_bytes();
        Self::from_parts(ProofFormat::Alethe, data, Some(num_steps), Some(stats))
    }

    /// Get the proof format.
    pub fn format(&self) -> ProofFormat {
        self.format
    }

    /// Get the number of proof steps.
    pub fn num_steps(&self) -> Option<usize> {
        self.num_steps
    }

    /// Get proof statistics.
    pub fn stats(&self) -> Option<&ProofStats> {
        self.stats.as_ref()
    }

    /// Get the proof as a UTF-8 string (for text-based formats like Alethe).
    ///
    /// # ENSURES
    /// - Returns `Some(&str)` iff `self.format` is text-based and `self.data` is valid UTF-8
    pub fn as_text(&self) -> Option<&str> {
        match self.format() {
            ProofFormat::Alethe | ProofFormat::Lfsc => std::str::from_utf8(self.as_bytes()).ok(),
            _ => None,
        }
    }

    /// Get the raw proof bytes.
    ///
    /// # ENSURES
    /// - Returns a slice of `self.data`
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Export proof to a file.
    ///
    /// # REQUIRES
    /// - `path` points to a writable location
    ///
    /// # ENSURES
    /// - On `Ok(())`, file contents equal `self.as_bytes()`
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, &self.data)
    }
}
