// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Source-authenticated, CONDITIONAL paired ReLU phase certificates.
//!
//! A paired phase has four closed alternatives, including both signs at zero.
//! A conditional band or empty-node certificate cannot establish the whole
//! root band. The cover API rebuilds the source model itself, checks all four
//! alternatives, and exposes root evidence only when its premises are empty.
//! Candidate text supplies multipliers only; rows, variables, phase bindings,
//! boxes, objective coefficients and claimed bounds are source-owned.

use super::lp_certificate::{certified_lower, DualMultipliers};
use super::lp_model::{self, Model};
use super::*;

const MAX_PHASE_PAIRS: usize = 10_000;
const MAX_CERTIFICATE_BYTES: usize = 64 * 1024 * 1024;
const MAX_MULTIPLIER: f64 = 1e100;
const MAX_PHASE_TREE_NODES: usize = 65_536;
const MAX_PHASE_TREE_DEPTH: usize = 128;

/// Exact source ReLU pair and its two closed preactivation signs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PairedPhase {
    pub stage: usize,
    pub neuron: usize,
    pub f_nonnegative: bool,
    pub g_nonnegative: bool,
}

impl PairedPhase {
    pub fn new(stage: usize, neuron: usize, f_nonnegative: bool, g_nonnegative: bool) -> Self {
        Self {
            stage,
            neuron,
            f_nonnegative,
            g_nonnegative,
        }
    }

    /// Cover order is (++), (--), (+-), (-+). Zero belongs to both signs.
    pub fn closed_children(stage: usize, neuron: usize) -> [Self; 4] {
        [
            Self::new(stage, neuron, true, true),
            Self::new(stage, neuron, false, false),
            Self::new(stage, neuron, true, false),
            Self::new(stage, neuron, false, true),
        ]
    }
}

/// Mathematical evidence about the listed phase region, not the whole root.
#[derive(Clone, Debug)]
pub struct PhaseLpReport {
    assumptions: Vec<PairedPhase>,
    band: LpProofReport,
}

impl PhaseLpReport {
    pub fn assumptions(&self) -> &[PairedPhase] {
        &self.assumptions
    }

    pub fn band_certified(&self) -> bool {
        self.band.proved
    }

    pub fn lower(&self) -> &[f64] {
        &self.band.lower
    }

    pub fn upper(&self) -> &[f64] {
        &self.band.upper
    }
}

/// A checked positive lower bound for the zero objective proves this phase
/// region empty. Private fields prevent authored evidence tokens.
#[derive(Clone, Debug)]
pub struct PhaseInfeasibilityReport {
    assumptions: Vec<PairedPhase>,
    lower: f64,
}

impl PhaseInfeasibilityReport {
    pub fn assumptions(&self) -> &[PairedPhase] {
        &self.assumptions
    }

    pub fn contradiction_lower(&self) -> f64 {
        self.lower
    }
}

/// Text kind for one child in PairedPhase::closed_children order.
#[derive(Clone, Copy, Debug)]
pub enum PhaseChildCertificate<'a> {
    /// Existing PAIRED_LP_DUAL_GEQ_V1, checking both output signs.
    Band(&'a str),
    /// PAIRED_LP_INFEAS_GEQ_V1, checking only the zero objective.
    Infeasible(&'a str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseChildStatus {
    BandCertified,
    Infeasible,
}

/// All four children have source-replayed evidence. The conjunction of the
/// premises is safe; it is a complete root proof only with EMPTY premises.
#[derive(Clone, Debug)]
pub struct PhaseCoverReport {
    premises: Vec<PairedPhase>,
    stage: usize,
    neuron: usize,
    children: [PhaseChildStatus; 4],
}

impl PhaseCoverReport {
    pub fn premises(&self) -> &[PairedPhase] {
        &self.premises
    }

    pub fn split_pair(&self) -> (usize, usize) {
        (self.stage, self.neuron)
    }

    pub fn children(&self) -> &[PhaseChildStatus; 4] {
        &self.children
    }

    pub fn conditioned_band_certified(&self) -> bool {
        true
    }

    /// Mathematical root evidence still requires the CLI formula/receipt gate.
    pub fn root_band_certified(&self) -> bool {
        self.premises.is_empty()
    }
}

/// One node of a flat, source-authenticated proof transcript. Child order is
/// (++), (--), (+-), (-+), exactly as in PairedPhase::closed_children.
/// Indices provide a bounded iterative traversal and avoid recursive drops.
#[derive(Clone, Copy, Debug)]
pub enum PhaseProofNode<'a> {
    Band(&'a str),
    Infeasible(&'a str),
    Split {
        stage: usize,
        neuron: usize,
        children: [usize; 4],
    },
}

/// A finite four-child tree. Construction checks structure and resource caps,
/// but only source replay can authenticate its bindings and close its leaves.
#[derive(Clone, Debug)]
pub struct PhaseProofTree<'a> {
    nodes: Vec<PhaseProofNode<'a>>,
    root: usize,
}

impl<'a> PhaseProofTree<'a> {
    /// Reject aliases, cycles, missing/unreachable nodes, depth over 128,
    /// over 65,536 nodes, and over 64 MiB of aggregate certificate text.
    /// Repeated references to the same text count separately towards the cap.
    pub fn new(nodes: Vec<PhaseProofNode<'a>>, root: usize) -> Option<Self> {
        let result = Self { nodes, root };
        result.validate_structure(None)?;
        Some(result)
    }

    pub fn nodes(&self) -> &[PhaseProofNode<'a>] {
        &self.nodes
    }

    pub fn root(&self) -> usize {
        self.root
    }

    fn validate_structure(&self, deadline: Option<Instant>) -> Option<()> {
        if self.nodes.is_empty()
            || self.nodes.len() > MAX_PHASE_TREE_NODES
            || self.root >= self.nodes.len()
            || expired(deadline)
        {
            return None;
        }
        let mut bytes = 0usize;
        let mut parents = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            if expired(deadline) {
                return None;
            }
            match node {
                PhaseProofNode::Band(text) | PhaseProofNode::Infeasible(text) => {
                    bytes = bytes.checked_add(text.len())?;
                    if bytes > MAX_CERTIFICATE_BYTES {
                        return None;
                    }
                }
                PhaseProofNode::Split { children, .. } => {
                    for &child in children {
                        let count = parents.get_mut(child)?;
                        *count = count.checked_add(1)?;
                        if *count > 1 {
                            return None;
                        }
                    }
                }
            }
        }
        if parents[self.root] != 0
            || parents
                .iter()
                .enumerate()
                .any(|(j, &n)| j != self.root && n != 1)
        {
            return None;
        }
        let mut visited = vec![false; self.nodes.len()];
        let mut count = 0usize;
        let mut stack = vec![(self.root, 0usize)];
        while let Some((index, depth)) = stack.pop() {
            if expired(deadline) || depth > MAX_PHASE_TREE_DEPTH || visited[index] {
                return None;
            }
            visited[index] = true;
            count = count.checked_add(1)?;
            if let PhaseProofNode::Split { children, .. } = self.nodes[index] {
                let child_depth = depth.checked_add(1)?;
                for child in children.into_iter().rev() {
                    stack.push((child, child_depth));
                }
            }
        }
        (count == self.nodes.len() && !expired(deadline)).then_some(())
    }
}

/// Every reachable leaf was replayed against the same source/root and all
/// closed alternatives were retained. Fields cannot be authored by callers.
#[derive(Clone, Debug)]
pub struct PhaseTreeReport {
    premises: Vec<PairedPhase>,
    nodes: usize,
    band_leaves: usize,
    infeasible_leaves: usize,
    maximum_depth: usize,
}

impl PhaseTreeReport {
    pub fn premises(&self) -> &[PairedPhase] {
        &self.premises
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn band_leaves(&self) -> usize {
        self.band_leaves
    }

    pub fn infeasible_leaves(&self) -> usize {
        self.infeasible_leaves
    }

    pub fn maximum_depth(&self) -> usize {
        self.maximum_depth
    }

    pub fn conditioned_band_certified(&self) -> bool {
        true
    }

    /// Mathematical root evidence still needs the CLI formula/receipt gate.
    pub fn root_band_certified(&self) -> bool {
        self.premises.is_empty()
    }
}

fn expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

fn canonical_phases(phases: &[PairedPhase]) -> Option<Vec<PairedPhase>> {
    if phases.len() > MAX_PHASE_PAIRS {
        return None;
    }
    let mut result = phases.to_vec();
    result.sort_unstable();
    // Each pair is assigned once. Opposite closed signs can overlap at zero:
    // duplicate descriptors are a malformed transcript, not an empty region.
    if result
        .windows(2)
        .any(|p| (p[0].stage, p[0].neuron) == (p[1].stage, p[1].neuron))
    {
        return None;
    }
    Some(result)
}

fn binding(model: &Model, phase: PairedPhase) -> Option<(usize, usize, usize, usize)> {
    model
        .relu_pairs
        .iter()
        .find(|(stage, _)| *stage == phase.stage)?
        .1
        .get(phase.neuron)
        .copied()
}

fn append_tower_phase(model: &mut Model, z: usize, r: usize, nonnegative: bool) -> Option<()> {
    if nonnegative {
        model.row(vec![(z, 1.0)], 0.0, None)?;
        model.row(vec![(r, 1.0), (z, -1.0)], 0.0, None)?;
        model.row(vec![(z, 1.0), (r, -1.0)], 0.0, None)?;
    } else {
        model.row(vec![(z, -1.0)], 0.0, None)?;
        model.row(vec![(r, 1.0)], 0.0, None)?;
        model.row(vec![(r, -1.0)], 0.0, None)?;
    }
    Some(())
}

fn append_phases(model: &mut Model, phases: &[PairedPhase]) -> Option<()> {
    append_phases_until(model, phases, None)
}

fn append_phases_until(
    model: &mut Model,
    phases: &[PairedPhase],
    deadline: Option<Instant>,
) -> Option<()> {
    for &phase in phases {
        if expired(deadline) {
            return None;
        }
        let (zf, zg, rf, rg) = binding(model, phase)?;
        // Preserve the authored unique equality pivots. Phase equalities are
        // encoded by BOTH GEQ directions, including stable source neurons.
        append_tower_phase(model, zf, rf, phase.f_nonnegative)?;
        append_tower_phase(model, zg, rg, phase.g_nonnegative)?;
    }
    (!expired(deadline)).then_some(())
}

fn conditional_model(
    chain: &PairedChain,
    domain: &BoxDomain,
    phases: &[PairedPhase],
) -> Option<(Model, Vec<PairedPhase>)> {
    let phases = canonical_phases(phases)?;
    let mut model = lp_model::build(chain, domain)?;
    append_phases(&mut model, &phases)?;
    Some((model, phases))
}

fn checked_band(model: &Model, epsilon: f64, text: &str) -> Option<LpProofReport> {
    checked_band_until(model, epsilon, text, None)
}

fn checked_band_until(
    model: &Model,
    epsilon: f64,
    text: &str,
    deadline: Option<Instant>,
) -> Option<LpProofReport> {
    if !valid_epsilon(epsilon) || text.len() > MAX_CERTIFICATE_BYTES || expired(deadline) {
        return None;
    }
    let candidates = lp_model::parse_dual_text(model, text)?;
    if expired(deadline) {
        return None;
    }
    let report = lp_model::check_candidates(model, epsilon, &candidates, deadline)?;
    (!expired(deadline)).then_some(report)
}

fn valid_epsilon(epsilon: f64) -> bool {
    epsilon.is_finite() && epsilon > 0.0 && epsilon <= 1e100
}

fn number(tokens: &mut std::str::SplitWhitespace<'_>) -> Option<f64> {
    let value = tokens.next()?.parse::<f64>().ok()?;
    (value.is_finite() && value.abs() <= MAX_MULTIPLIER).then_some(value)
}

fn checked_infeasibility(model: &Model, text: &str) -> Option<f64> {
    checked_infeasibility_until(model, text, None)
}

fn checked_infeasibility_until(
    model: &Model,
    text: &str,
    deadline: Option<Instant>,
) -> Option<f64> {
    if text.len() > MAX_CERTIFICATE_BYTES || expired(deadline) {
        return None;
    }
    let mut tokens = text.split_whitespace();
    if tokens.next()? != "PAIRED_LP_INFEAS_GEQ_V1"
        || tokens.next()?.parse::<usize>().ok()? != model.equalities.len()
        || tokens.next()?.parse::<usize>().ok()? != model.inequalities.len()
    {
        return None;
    }
    let equality_multipliers = (0..model.equalities.len())
        .map(|_| number(&mut tokens))
        .collect::<Option<Vec<_>>>()?;
    let inequality_multipliers = (0..model.inequalities.len())
        .map(|_| number(&mut tokens))
        .collect::<Option<Vec<_>>>()?;
    if inequality_multipliers.iter().any(|&v| v < 0.0)
        || tokens.next().is_some()
        || expired(deadline)
    {
        return None;
    }
    let lower = certified_lower(
        &model.bounds,
        &vec![0.0; model.bounds.len()],
        &model.equalities,
        &model.inequalities,
        &DualMultipliers {
            equality_multipliers,
            inequality_multipliers,
        },
    )?;
    // For any feasible v, c*v is exactly zero. A rigorous POSITIVE lower
    // certificate contradicts feasibility. Equality to zero proves nothing.
    (lower > 0.0 && !expired(deadline)).then_some(lower)
}

/// Only one immutable source model and one path of assumptions are retained.
/// Model clones exist only while replaying a leaf. No failed leaf, malformed
/// subtree or expiry can produce a partial cover report.
fn check_tree_with_model(
    base: &Model,
    epsilon: f64,
    premises: Vec<PairedPhase>,
    tree: &PhaseProofTree<'_>,
    deadline: Option<Instant>,
) -> Option<PhaseTreeReport> {
    if !valid_epsilon(epsilon) || expired(deadline) {
        return None;
    }
    tree.validate_structure(deadline)?;
    for &premise in &premises {
        binding(base, premise)?;
    }
    // A frame truncates the shared path to its parent and appends its one
    // source-owned child descriptor. This is iterative DFS, not recursion.
    let mut stack = vec![(tree.root, 0usize, premises.len(), None)];
    let mut assumptions = premises.clone();
    let mut report = PhaseTreeReport {
        premises,
        nodes: 0,
        band_leaves: 0,
        infeasible_leaves: 0,
        maximum_depth: 0,
    };
    while let Some((index, depth, parent_len, phase)) = stack.pop() {
        if expired(deadline) || parent_len > assumptions.len() {
            return None;
        }
        assumptions.truncate(parent_len);
        if let Some(phase) = phase {
            assumptions.push(phase);
        }
        if assumptions.len() > MAX_PHASE_PAIRS {
            return None;
        }
        report.nodes = report.nodes.checked_add(1)?;
        report.maximum_depth = report.maximum_depth.max(depth);
        match tree.nodes[index] {
            PhaseProofNode::Band(text) | PhaseProofNode::Infeasible(text) => {
                let phases = canonical_phases(&assumptions)?;
                let mut model = base.clone();
                append_phases_until(&mut model, &phases, deadline)?;
                match tree.nodes[index] {
                    PhaseProofNode::Band(_) => {
                        if !checked_band_until(&model, epsilon, text, deadline)?.proved {
                            return None;
                        }
                        report.band_leaves = report.band_leaves.checked_add(1)?;
                    }
                    PhaseProofNode::Infeasible(_) => {
                        checked_infeasibility_until(&model, text, deadline)?;
                        report.infeasible_leaves = report.infeasible_leaves.checked_add(1)?;
                    }
                    PhaseProofNode::Split { .. } => unreachable!(),
                }
            }
            PhaseProofNode::Split {
                stage,
                neuron,
                children,
            } => {
                if assumptions.len() >= MAX_PHASE_PAIRS
                    || assumptions
                        .iter()
                        .any(|p| (p.stage, p.neuron) == (stage, neuron))
                {
                    return None;
                }
                let child_phases = PairedPhase::closed_children(stage, neuron);
                binding(base, child_phases[0])?;
                let child_depth = depth.checked_add(1)?;
                for child in (0..4).rev() {
                    stack.push((
                        children[child],
                        child_depth,
                        assumptions.len(),
                        Some(child_phases[child]),
                    ));
                }
            }
        }
    }
    // The entire source-bound transcript must finish before its original
    // absolute deadline; untimed/offline replay makes no timing claim.
    (report.nodes == tree.nodes.len() && !expired(deadline)).then_some(report)
}

impl PairedChain {
    /// Offline replay of a finite nested phase transcript. Every split checks
    /// all four closed alternatives, including both mixed-sign children.
    /// This bounded checker may decline; it makes no search-completeness claim.
    pub fn check_lp_phase_tree(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        premises: &[PairedPhase],
        tree: &PhaseProofTree<'_>,
    ) -> Option<PhaseTreeReport> {
        self.check_lp_phase_tree_inner(domain, epsilon, premises, tree, None)
    }

    /// Replay uses one original absolute deadline for source reconstruction,
    /// every node, certificate repair and the final report. Expiry yields no
    /// proof. A single bounded parse/arithmetic loop may finish after expiry;
    /// its result is then rejected, rather than claiming hard interruption.
    pub fn check_lp_phase_tree_until(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        premises: &[PairedPhase],
        tree: &PhaseProofTree<'_>,
        deadline: Instant,
    ) -> Option<PhaseTreeReport> {
        self.check_lp_phase_tree_inner(domain, epsilon, premises, tree, Some(deadline))
    }

    fn check_lp_phase_tree_inner(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        premises: &[PairedPhase],
        tree: &PhaseProofTree<'_>,
        deadline: Option<Instant>,
    ) -> Option<PhaseTreeReport> {
        if !valid_epsilon(epsilon) || expired(deadline) {
            return None;
        }
        tree.validate_structure(deadline)?;
        let premises = canonical_phases(premises)?;
        let base = lp_model::build_until(self, domain, deadline)?;
        check_tree_with_model(&base, epsilon, premises, tree, deadline)
    }

    /// Dump source rows conditioned on the supplied validated phase premises.
    /// This is an LP proposal scaffold, never a whole-root proof by itself.
    pub fn lp_phase_dump(&self, domain: &BoxDomain, phases: &[PairedPhase]) -> Option<String> {
        let (model, _) = conditional_model(self, domain, phases)?;
        lp_model::dump_model(&model)
    }

    /// Rebuild conditional source rows and check both signs of every output.
    /// A successful band is conditional on ALL returned assumptions.
    pub fn check_lp_phase_text(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        phases: &[PairedPhase],
        text: &str,
    ) -> Option<PhaseLpReport> {
        if !valid_epsilon(epsilon) || text.len() > MAX_CERTIFICATE_BYTES {
            return None;
        }
        let (model, assumptions) = conditional_model(self, domain, phases)?;
        let band = checked_band(&model, epsilon, text)?;
        Some(PhaseLpReport { assumptions, band })
    }

    /// Check an infeasibility certificate for exactly the listed premises.
    /// Grammar: PAIRED_LP_INFEAS_GEQ_V1 ne ni, followed by ne free equality
    /// multipliers and ni nonnegative inequality multipliers; exact EOF.
    pub fn check_lp_phase_infeasibility_text(
        &self,
        domain: &BoxDomain,
        phases: &[PairedPhase],
        text: &str,
    ) -> Option<PhaseInfeasibilityReport> {
        if text.len() > MAX_CERTIFICATE_BYTES {
            return None;
        }
        let (model, assumptions) = conditional_model(self, domain, phases)?;
        let lower = checked_infeasibility(&model, text)?;
        Some(PhaseInfeasibilityReport { assumptions, lower })
    }

    /// Source-owned full four-child replay. Caller text cannot omit, relabel
    /// or replace a mixed child. Existing premises are included in EVERY
    /// child, and a conditional cover is not exposed as root evidence.
    pub fn check_lp_phase_cover_text(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        premises: &[PairedPhase],
        stage: usize,
        neuron: usize,
        certificates: [PhaseChildCertificate<'_>; 4],
    ) -> Option<PhaseCoverReport> {
        if !valid_epsilon(epsilon)
            || certificates.iter().any(|certificate| match certificate {
                PhaseChildCertificate::Band(text) | PhaseChildCertificate::Infeasible(text) => {
                    text.len() > MAX_CERTIFICATE_BYTES
                }
            })
        {
            return None;
        }
        let premises = canonical_phases(premises)?;
        if premises
            .iter()
            .any(|p| (p.stage, p.neuron) == (stage, neuron))
            || premises.len() >= MAX_PHASE_PAIRS
        {
            return None;
        }
        let base = lp_model::build(self, domain)?;
        let mut children = [PhaseChildStatus::BandCertified; 4];
        for (index, phase) in PairedPhase::closed_children(stage, neuron)
            .into_iter()
            .enumerate()
        {
            let mut assumptions = premises.clone();
            assumptions.push(phase);
            let assumptions = canonical_phases(&assumptions)?;
            let mut model = base.clone();
            append_phases(&mut model, &assumptions)?;
            children[index] = match certificates[index] {
                PhaseChildCertificate::Band(text) => {
                    if !checked_band(&model, epsilon, text)?.proved {
                        return None;
                    }
                    PhaseChildStatus::BandCertified
                }
                PhaseChildCertificate::Infeasible(text) => {
                    checked_infeasibility(&model, text)?;
                    PhaseChildStatus::Infeasible
                }
            };
        }
        Some(PhaseCoverReport {
            premises,
            stage,
            neuron,
            children,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    fn zero_chain(width: usize, bias: f64) -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: width,
            stages: vec![
                Stage::Affine {
                    rows: width,
                    cols: 1,
                    wf: vec![0.0; width],
                    wg: vec![0.0; width],
                    bf: vec![bias; width],
                    bg: vec![bias; width],
                },
                Stage::Relu { width },
            ],
        }
    }

    fn domain() -> BoxDomain {
        BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        }
    }

    fn band_text(model: &Model, duals: &[DualMultipliers]) -> String {
        let mut text = format!(
            "PAIRED_LP_DUAL_GEQ_V1 {} {} {}\n",
            model.equalities.len(),
            model.inequalities.len(),
            model.outputs.len()
        );
        assert_eq!(duals.len(), 2 * model.outputs.len());
        for dual in duals {
            for value in dual
                .equality_multipliers
                .iter()
                .chain(&dual.inequality_multipliers)
            {
                write!(text, "{value:.17e} ").unwrap();
            }
        }
        text
    }

    fn empty_dual(model: &Model) -> DualMultipliers {
        DualMultipliers {
            equality_multipliers: vec![0.0; model.equalities.len()],
            inequality_multipliers: vec![0.0; model.inequalities.len()],
        }
    }

    fn zero_band(model: &Model) -> String {
        band_text(model, &vec![empty_dual(model); 2 * model.outputs.len()])
    }

    fn infeasibility_text(model: &Model, dual: &DualMultipliers) -> String {
        let mut text = format!(
            "PAIRED_LP_INFEAS_GEQ_V1 {} {}\n",
            model.equalities.len(),
            model.inequalities.len()
        );
        for value in dual
            .equality_multipliers
            .iter()
            .chain(&dual.inequality_multipliers)
        {
            write!(text, "{value:.17e} ").unwrap();
        }
        text
    }

    fn nested_nodes(texts: &[String]) -> Vec<PhaseProofNode<'_>> {
        assert_eq!(texts.len(), 16);
        let mut nodes = vec![PhaseProofNode::Split {
            stage: 1,
            neuron: 0,
            children: [1, 2, 3, 4],
        }];
        for i in 0..4 {
            nodes.push(PhaseProofNode::Split {
                stage: 1,
                neuron: 1,
                children: std::array::from_fn(|j| 5 + 4 * i + j),
            });
        }
        nodes.extend(texts.iter().map(|text| PhaseProofNode::Band(text)));
        nodes
    }

    fn nested_zero_texts(chain: &PairedChain, premises: &[PairedPhase]) -> Vec<String> {
        let mut texts = Vec::new();
        for first in PairedPhase::closed_children(1, 0) {
            for second in PairedPhase::closed_children(1, 1) {
                let mut phases = premises.to_vec();
                phases.extend([first, second]);
                let model = conditional_model(chain, &domain(), &phases).unwrap().0;
                texts.push(zero_band(&model));
            }
        }
        texts
    }

    #[test]
    fn nested_closed_phase_tree_checks_all_sixteen_zero_boundary_leaves() {
        let chain = zero_chain(2, 0.0);
        let texts = nested_zero_texts(&chain, &[]);
        let tree = PhaseProofTree::new(nested_nodes(&texts), 0).unwrap();
        let report = chain
            .check_lp_phase_tree(&domain(), 1.0, &[], &tree)
            .unwrap();
        assert!(report.root_band_certified());
        assert!(report.conditioned_band_certified());
        assert_eq!(report.nodes(), 21);
        assert_eq!(report.band_leaves(), 16);
        assert_eq!(report.infeasible_leaves(), 0);
        assert_eq!(report.maximum_depth(), 2);
        // A fixed four-child type prevents omission by arity. An unproved
        // mixed grandchild still invalidates the entire parent transcript.
        let mut incomplete = nested_nodes(&texts);
        incomplete[7] = PhaseProofNode::Band("missing mixed-child evidence");
        let incomplete = PhaseProofTree::new(incomplete, 0).unwrap();
        assert!(chain
            .check_lp_phase_tree(&domain(), 1.0, &[], &incomplete)
            .is_none());
    }

    #[test]
    fn nested_tree_keeps_premises_and_rejects_repeated_or_foreign_pairs() {
        let chain = zero_chain(3, 0.0);
        let premise = PairedPhase::new(1, 2, true, false);
        let texts = nested_zero_texts(&chain, &[premise]);
        let tree = PhaseProofTree::new(nested_nodes(&texts), 0).unwrap();
        let report = chain
            .check_lp_phase_tree(&domain(), 1.0, &[premise], &tree)
            .unwrap();
        assert!(report.conditioned_band_certified());
        assert!(!report.root_band_certified());
        assert_eq!(report.premises(), &[premise]);
        for (stage, neuron) in [(1, 0), (1, 2), (0, 0), (1, 3)] {
            let mut nodes = nested_nodes(&texts);
            nodes[1] = PhaseProofNode::Split {
                stage,
                neuron,
                children: [5, 6, 7, 8],
            };
            let tree = PhaseProofTree::new(nodes, 0).unwrap();
            assert!(chain
                .check_lp_phase_tree(&domain(), 1.0, &[premise], &tree)
                .is_none());
        }
    }

    #[test]
    fn nested_tree_accepts_repaired_positive_empty_leaf_evidence() {
        let chain = zero_chain(2, 1.0);
        let base = lp_model::build(&chain, &domain()).unwrap();
        let mut texts = Vec::new();
        let mut empty = Vec::new();
        for first in PairedPhase::closed_children(1, 0) {
            for second in PairedPhase::closed_children(1, 1) {
                let model = conditional_model(&chain, &domain(), &[first, second])
                    .unwrap()
                    .0;
                if first.f_nonnegative && first.g_nonnegative {
                    texts.push(zero_band(&model));
                    empty.push(false);
                } else {
                    let (zf, zg, _, _) = binding(&model, first).unwrap();
                    let (z, offset) = if !first.f_nonnegative {
                        (zf, 0)
                    } else {
                        (zg, 3)
                    };
                    let pivot = model.pivots.iter().position(|&p| p == z).unwrap();
                    let mut dual = empty_dual(&model);
                    dual.equality_multipliers[pivot] = 1.0;
                    dual.inequality_multipliers[base.inequalities.len() + offset] = 1.0;
                    texts.push(infeasibility_text(&model, &dual));
                    empty.push(true);
                }
            }
        }
        let mut nodes = nested_nodes(&texts);
        for i in 0..16 {
            if empty[i] {
                nodes[5 + i] = PhaseProofNode::Infeasible(&texts[i]);
            }
        }
        let tree = PhaseProofTree::new(nodes, 0).unwrap();
        let report = chain
            .check_lp_phase_tree(&domain(), 1.0, &[], &tree)
            .unwrap();
        assert!(report.root_band_certified());
        assert_eq!(report.band_leaves(), 4);
        assert_eq!(report.infeasible_leaves(), 12);
    }

    #[test]
    fn nested_safe_conditional_leaf_texts_cannot_cover_an_actual_violation() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 2,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 1,
                    wf: vec![1.0, 0.0],
                    wg: vec![1.0, 0.0],
                    bf: vec![0.5, 0.0],
                    bg: vec![0.0, 0.0],
                },
                Stage::Relu { width: 2 },
            ],
        };
        let first = PairedPhase::new(1, 0, false, false);
        let second = PairedPhase::new(1, 1, false, false);
        let base = lp_model::build(&chain, &domain()).unwrap();
        let offset = base.inequalities.len();
        let model = conditional_model(&chain, &domain(), &[first, second])
            .unwrap()
            .0;
        let mut positive = empty_dual(&model);
        let mut negative = empty_dual(&model);
        positive.inequality_multipliers[offset + 1] = 1.0;
        positive.inequality_multipliers[offset + 5] = 1.0;
        negative.inequality_multipliers[offset + 2] = 1.0;
        negative.inequality_multipliers[offset + 4] = 1.0;
        let text = band_text(
            &model,
            &[positive, negative, empty_dual(&model), empty_dual(&model)],
        );
        assert!(chain
            .check_lp_phase_text(&domain(), 0.1, &[first, second], &text)
            .unwrap()
            .band_certified());
        let texts = vec![text; 16];
        let tree = PhaseProofTree::new(nested_nodes(&texts), 0).unwrap();
        assert!(chain
            .check_lp_phase_tree(&domain(), 0.1, &[], &tree)
            .is_none());
        let (f, g) = chain.forward(&[1.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.5);
    }

    #[test]
    fn phase_tree_structure_and_aggregate_resource_caps_fail_closed() {
        assert!(PhaseProofTree::new(vec![], 0).is_none());
        assert!(PhaseProofTree::new(vec![PhaseProofNode::Band("")], 1).is_none());
        assert!(PhaseProofTree::new(
            vec![
                PhaseProofNode::Split {
                    stage: 1,
                    neuron: 0,
                    children: [1, 1, 2, 3]
                },
                PhaseProofNode::Band(""),
                PhaseProofNode::Band(""),
                PhaseProofNode::Band(""),
            ],
            0,
        )
        .is_none());
        // A disconnected cycle has valid indegrees but must be rejected.
        assert!(PhaseProofTree::new(
            vec![
                PhaseProofNode::Band(""),
                PhaseProofNode::Split {
                    stage: 1,
                    neuron: 0,
                    children: [1, 2, 3, 4]
                },
                PhaseProofNode::Band(""),
                PhaseProofNode::Band(""),
                PhaseProofNode::Band(""),
            ],
            0,
        )
        .is_none());
        assert!(
            PhaseProofTree::new(vec![PhaseProofNode::Band(""); MAX_PHASE_TREE_NODES + 1], 0)
                .is_none()
        );
        let mut nodes = vec![PhaseProofNode::Band("")];
        let mut continuation = 0;
        for neuron in 0..=MAX_PHASE_TREE_DEPTH {
            let first = nodes.len();
            nodes[continuation] = PhaseProofNode::Split {
                stage: 1,
                neuron,
                children: std::array::from_fn(|j| first + j),
            };
            nodes.extend([PhaseProofNode::Band(""); 4]);
            continuation = first;
        }
        assert!(PhaseProofTree::new(nodes, 0).is_none());
        let text = "0".repeat(MAX_CERTIFICATE_BYTES / 4 + 1);
        assert!(PhaseProofTree::new(
            vec![
                PhaseProofNode::Split {
                    stage: 1,
                    neuron: 0,
                    children: [1, 2, 3, 4]
                },
                PhaseProofNode::Band(&text),
                PhaseProofNode::Band(&text),
                PhaseProofNode::Band(&text),
                PhaseProofNode::Band(&text),
            ],
            0,
        )
        .is_none());
    }

    #[test]
    fn phase_tree_shared_absolute_deadline_rejects_late_replay() {
        let chain = zero_chain(2, 0.0);
        let texts = nested_zero_texts(&chain, &[]);
        let tree = PhaseProofTree::new(nested_nodes(&texts), 0).unwrap();
        let late = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert!(chain
            .check_lp_phase_tree_until(&domain(), 1.0, &[], &tree, late)
            .is_none());
        // Even after valid source reconstruction, an already expired replay
        // tail cannot return the otherwise valid mathematical report.
        let base = lp_model::build(&chain, &domain()).unwrap();
        assert!(check_tree_with_model(&base, 1.0, vec![], &tree, Some(late)).is_none());
        let future = Instant::now() + Duration::from_secs(10);
        assert!(chain
            .check_lp_phase_tree_until(&domain(), 1.0, &[], &tree, future)
            .unwrap()
            .root_band_certified());
    }

    #[test]
    fn zero_boundary_requires_all_four_children_including_both_mixed_regions() {
        let chain = zero_chain(1, 0.0);
        let phases = PairedPhase::closed_children(1, 0);
        let texts: Vec<_> = phases
            .iter()
            .map(|phase| zero_band(&conditional_model(&chain, &domain(), &[*phase]).unwrap().0))
            .collect();
        let cover = chain
            .check_lp_phase_cover_text(
                &domain(),
                1.0,
                &[],
                1,
                0,
                std::array::from_fn(|i| PhaseChildCertificate::Band(&texts[i])),
            )
            .unwrap();
        assert!(cover.root_band_certified());
        assert!(cover.conditioned_band_certified());
        assert_eq!(cover.children(), &[PhaseChildStatus::BandCertified; 4]);
        // All four source regions are nonempty: zf=zg=0 for every input.
        // A zero lower bound must never prune either mixed child.
        let mixed = conditional_model(&chain, &domain(), &[phases[2]])
            .unwrap()
            .0;
        let fake_empty = infeasibility_text(&mixed, &empty_dual(&mixed));
        assert!(chain
            .check_lp_phase_infeasibility_text(&domain(), &[phases[2]], &fake_empty)
            .is_none());
        assert!(chain
            .check_lp_phase_cover_text(
                &domain(),
                1.0,
                &[],
                1,
                0,
                [
                    PhaseChildCertificate::Band(&texts[0]),
                    PhaseChildCertificate::Band(&texts[1]),
                    PhaseChildCertificate::Infeasible(&fake_empty),
                    PhaseChildCertificate::Band(&texts[3]),
                ],
            )
            .is_none());
        assert!(chain
            .check_lp_phase_cover_text(
                &domain(),
                1.0,
                &[],
                1,
                0,
                [
                    PhaseChildCertificate::Band(&texts[0]),
                    PhaseChildCertificate::Band(&texts[1]),
                    PhaseChildCertificate::Band(&texts[2]),
                    PhaseChildCertificate::Band("incomplete"),
                ],
            )
            .is_none());
    }

    #[test]
    fn cover_under_nonempty_premises_cannot_be_promoted_to_a_root_band() {
        let chain = zero_chain(2, 0.0);
        let premise = PairedPhase::new(1, 0, true, true);
        let texts: Vec<_> = PairedPhase::closed_children(1, 1)
            .iter()
            .map(|&phase| {
                zero_band(
                    &conditional_model(&chain, &domain(), &[premise, phase])
                        .unwrap()
                        .0,
                )
            })
            .collect();
        let cover = chain
            .check_lp_phase_cover_text(
                &domain(),
                1.0,
                &[premise],
                1,
                1,
                std::array::from_fn(|i| PhaseChildCertificate::Band(&texts[i])),
            )
            .unwrap();
        assert!(cover.conditioned_band_certified());
        assert!(!cover.root_band_certified());
        assert_eq!(cover.premises(), &[premise]);
        assert!(chain
            .check_lp_phase_cover_text(
                &domain(),
                1.0,
                &[premise],
                1,
                0,
                std::array::from_fn(|i| PhaseChildCertificate::Band(&texts[i])),
            )
            .is_none());
    }

    #[test]
    fn strictly_positive_zero_objective_certificate_proves_a_source_phase_empty() {
        let chain = zero_chain(1, 1.0);
        let phase = PairedPhase::new(1, 0, false, true);
        let base = lp_model::build(&chain, &domain()).unwrap();
        let offset = base.inequalities.len();
        let mut model = base;
        append_phases(&mut model, &[phase]).unwrap();
        let (zf, _, _, _) = binding(&model, phase).unwrap();
        let pivot = model.pivots.iter().position(|&p| p == zf).unwrap();
        let mut dual = empty_dual(&model);
        // Authored zf=1 plus the source phase row -zf>=0 gives 0>=1.
        dual.equality_multipliers[pivot] = 1.0;
        dual.inequality_multipliers[offset] = 1.0;
        let text = infeasibility_text(&model, &dual);
        let report = chain
            .check_lp_phase_infeasibility_text(&domain(), &[phase], &text)
            .unwrap();
        assert!(report.contradiction_lower() > 0.0);
        assert_eq!(report.assumptions(), &[phase]);
        dual.inequality_multipliers[offset] = -1.0;
        assert!(chain
            .check_lp_phase_infeasibility_text(
                &domain(),
                &[phase],
                &infeasibility_text(&model, &dual),
            )
            .is_none());
        assert!(chain
            .check_lp_phase_infeasibility_text(&domain(), &[phase], &(text + " 0"))
            .is_none());
    }

    #[test]
    fn an_inactive_safe_region_does_not_close_a_real_root_violation() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![0.5],
                    bg: vec![0.0],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let phase = PairedPhase::new(1, 0, false, false);
        let base = lp_model::build(&chain, &domain()).unwrap();
        let offset = base.inequalities.len();
        let mut model = base;
        append_phases(&mut model, &[phase]).unwrap();
        let mut positive = empty_dual(&model);
        let mut negative = empty_dual(&model);
        // rf>=0 and -rg>=0 bound +delta; their opposites bound -delta.
        positive.inequality_multipliers[offset + 1] = 1.0;
        positive.inequality_multipliers[offset + 5] = 1.0;
        negative.inequality_multipliers[offset + 2] = 1.0;
        negative.inequality_multipliers[offset + 4] = 1.0;
        let text = band_text(&model, &[positive, negative]);
        let report = chain
            .check_lp_phase_text(&domain(), 0.1, &[phase], &text)
            .unwrap();
        assert!(report.band_certified());
        assert_eq!(report.assumptions(), &[phase]);
        assert!(chain.check_lp_text(&domain(), 0.1, &text).is_none());
        assert!(chain
            .check_lp_phase_cover_text(
                &domain(),
                0.1,
                &[],
                1,
                0,
                [PhaseChildCertificate::Band(&text); 4],
            )
            .is_none());
        // An EXACT dyadic root trace outside this phase region violates .1.
        let (f, g) = chain.forward(&[1.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.5);
    }

    #[test]
    fn source_phase_admission_rejects_duplicates_bad_stages_and_bad_neurons() {
        let chain = zero_chain(1, 0.0);
        let phase = PairedPhase::new(1, 0, true, false);
        assert!(chain.lp_phase_dump(&domain(), &[phase]).is_some());
        for invalid in [
            vec![phase, phase],
            vec![phase, PairedPhase::new(1, 0, false, true)],
            vec![PairedPhase::new(0, 0, true, true)],
            vec![PairedPhase::new(1, 1, true, true)],
        ] {
            assert!(chain.lp_phase_dump(&domain(), &invalid).is_none());
        }
        let (model, _) = conditional_model(&chain, &domain(), &[phase]).unwrap();
        let mut text = zero_band(&model);
        text.push_str(" 0");
        assert!(chain
            .check_lp_phase_text(&domain(), 1.0, &[phase], &text)
            .is_none());
    }
}
