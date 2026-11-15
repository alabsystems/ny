// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::certify::scope::{NumericalSemantics, SCOPE_SCHEMA_VERSION};

use super::*;

/// Validated ordered original owners. The chain must start/end with an affine
/// node; ReLU, exact GELU-erf or SiLU may only follow an affine node. No branching,
/// tanh GELU, deployed arithmetic, or other op is given proof eligibility.
/// Reports keep the actual immutable owners alive.
pub struct PreparedOriginalAffineChain<'a> {
    pub(super) stages: Vec<Stage<'a>>,
    pub(super) shared: Arc<ChainOwner>,
    input_features: usize,
    output_features: usize,
}

impl<'a> PreparedOriginalAffineChain<'a> {
    /// Metadata/topology admission, separate from the original P1 validation
    /// requests for each affine. All phases consume the one supplied ledger.
    pub fn preparation_request(nodes: &[OriginalChainNode<'_>]) -> Result<WorkRequest> {
        if nodes.is_empty() || nodes.len() > MAX_NODES {
            return Err(NyError::UnsupportedConfiguration(
                "original-affine chain requires 1..=128 nodes".into(),
            ));
        }
        let metadata = add(
            META,
            mul(
                nodes.len(),
                add(size_of::<Stage<'_>>(), size_of::<PointLinearLayer>())?,
            )?,
        )?;
        request(
            add(
                mul(mul(nodes.len(), nodes.len())?, 8)?,
                mul(nodes.len(), 128)?,
            )?,
            metadata,
        )
    }

    pub fn prepare(
        nodes: &[OriginalChainNode<'a>],
        options: OriginalAffineOptions<'_>,
    ) -> Result<Self> {
        let admission = Self::preparation_request(nodes)?;
        let mut deadline = earlier(options.deadline, ny_core::instance_budget::deadline());
        // Bounded static owner inspection only; no coefficient read/allocation.
        for node in nodes {
            if let Layer::PointLinear(owner) = node.layer {
                if !owner.original_budget().shares_ledger(options.budget) {
                    return Err(NyError::UnsupportedConfiguration(
                        "original-affine chain owner/caller ledger mismatch".into(),
                    ));
                }
                deadline = earlier(deadline, owner.original_deadline());
            }
        }
        let mut phase =
            OriginalAffinePhase::admit(options.budget, options.class, admission, deadline)?;
        let mut model = Sha256::new();
        let mut coordinates = Sha256::new();
        model.update(b"ny-original-affine-chain-actual-owners/v1");
        coordinates.update(b"ny-original-affine-chain-ordered-edges/v1");
        let mut features = None;
        let mut input_features = None;
        let mut previous_output = None;
        let mut previous_affine = false;
        let mut has_gelu = false;
        let mut has_silu = false;
        for (index, node) in nodes.iter().enumerate() {
            phase.checkpoint()?;
            phase.charge_work(128)?;
            if node.ids.lowering != nodes[0].ids.lowering
                || previous_output.is_some_and(|p| p != node.ids.input)
                || node.ids.input == node.ids.output
                || node.ids.output == nodes[0].ids.input
            {
                return Err(invalid("disconnected edge or inconsistent lowering"));
            }
            for prior in &nodes[..index] {
                phase.charge_work(8)?;
                if prior.ids.node == node.ids.node || prior.ids.output == node.ids.output {
                    return Err(invalid("repeated node/output or a non-chain edge"));
                }
            }
            for digest in [node.ids.node, node.ids.input, node.ids.output] {
                coordinates.update(digest.0);
            }
            match node.layer {
                Layer::PointLinear(owner) => {
                    if features.is_some_and(|n| n != owner.in_features()) {
                        return Err(invalid("adjacent affine feature mismatch"));
                    }
                    input_features.get_or_insert(owner.in_features());
                    features = Some(owner.out_features());
                    coordinates.update([0]);
                    coordinates.update(as_work(owner.in_features())?.to_le_bytes());
                    coordinates.update(as_work(owner.out_features())?.to_le_bytes());
                    model.update(owner.original_digest().0);
                    previous_affine = true;
                }
                Layer::ReLU(_) if index > 0 && index + 1 < nodes.len() && previous_affine => {
                    coordinates.update([1]);
                    previous_affine = false;
                }
                Layer::GELU(layer)
                    if layer.approximation == crate::layers::GeluApproximation::Erf
                        && index > 0
                        && index + 1 < nodes.len()
                        && previous_affine =>
                {
                    coordinates.update([2]);
                    coordinates.update(b"gelu-erf-exact-real/v1");
                    has_gelu = true;
                    previous_affine = false;
                }
                Layer::SiLU(_) if index > 0 && index + 1 < nodes.len() && previous_affine => {
                    coordinates.update([3]);
                    coordinates.update(b"silu-exact-real/v1");
                    has_silu = true;
                    previous_affine = false;
                }
                _ => {
                    return Err(NyError::UnsupportedOp(
                        "original-affine chain requires original affine/(ReLU, exact GELU-erf or SiLU,affine) order"
                            .into(),
                    ));
                }
            }
            previous_output = Some(node.ids.output);
        }
        if !previous_affine {
            return Err(invalid("terminal node must be original affine"));
        }
        let input_features =
            input_features.ok_or_else(|| invalid("missing original affine input"))?;
        let output_features = features.ok_or_else(|| invalid("missing original affine output"))?;
        let mut stages = Vec::new();
        let mut owners: Vec<PointLinearLayer> = Vec::new();
        stages
            .try_reserve_exact(nodes.len())
            .map_err(|_| invalid("stage allocation"))?;
        owners
            .try_reserve_exact(nodes.len())
            .map_err(|_| invalid("owner allocation"))?;
        let actual_metadata = add(
            META,
            add(
                mul(stages.capacity(), size_of::<Stage<'_>>())?,
                mul(owners.capacity(), size_of::<PointLinearLayer>())?,
            )?,
        )?;
        if actual_metadata > admission.persistent_bytes {
            return Err(invalid("stage/owner capacity exceeded admission"));
        }
        phase.retain_metadata(actual_metadata)?;
        let mut preparation_metadata_bytes = 0usize;
        for node in nodes {
            phase.checkpoint()?;
            match node.layer {
                Layer::PointLinear(owner) => {
                    let context = PreparedOriginalAffine::prepare(
                        owner,
                        node.ids,
                        OriginalAffineOptions {
                            deadline,
                            ..options
                        },
                    )?;
                    preparation_metadata_bytes = add(
                        preparation_metadata_bytes,
                        context.preparation_reserved_bytes(),
                    )?;
                    stages.push(Stage::Affine(context));
                    // A tied original owner may appear at several nodes. Its
                    // resident capacity is retained once. Distinct owners with
                    // identical coefficient bits are still separate owners.
                    let mut already_owned = false;
                    for prior in &owners {
                        phase.charge_work(4)?;
                        if std::ptr::eq(prior.weight(), owner.weight()) {
                            already_owned = true;
                            break;
                        }
                    }
                    if !already_owned {
                        owners.push(owner.clone());
                    }
                }
                Layer::ReLU(_) => stages.push(Stage::Relu),
                Layer::GELU(_) => stages.push(Stage::GeluErf),
                Layer::SiLU(_) => stages.push(Stage::SiLu),
                _ => return Err(invalid("validated stage changed")),
            }
        }
        let scope = ScopeIdentity {
            schema_version: SCOPE_SCHEMA_VERSION,
            model: Digest(model.finalize().into()),
            lowering: nodes[0].ids.lowering,
            coordinates: Digest(coordinates.finalize().into()),
            assumptions: Digest::of_bytes(
                b"original-affine-chain-profile",
                match (has_gelu, has_silu) {
                    (false, false) => ORIGINAL_CHAIN_EXACT_REAL_PROFILE,
                    (true, false) => ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE,
                    (false, true) => ORIGINAL_CHAIN_SILU_EXACT_REAL_PROFILE,
                    (true, true) => ORIGINAL_CHAIN_GELU_SILU_EXACT_REAL_PROFILE,
                }
                .as_bytes(),
            ),
            numerical: NumericalSemantics::ExactReal,
        };
        let reservation = phase.commit()?;
        let shared = Arc::new(ChainOwner {
            owners,
            scope,
            deadline,
            preparation_metadata_bytes,
            reservation,
        });
        Ok(Self {
            stages,
            shared,
            input_features,
            output_features,
        })
    }

    pub fn scope(&self) -> &ScopeIdentity {
        &self.shared.scope
    }
    pub fn input_features(&self) -> usize {
        self.input_features
    }
    pub fn output_features(&self) -> usize {
        self.output_features
    }
    pub fn num_nodes(&self) -> usize {
        self.stages.len()
    }
    /// Shared chain and per-node affine preparation metadata. Original
    /// coefficient owner capacities are reported separately by output reports.
    pub fn preparation_reserved_bytes(&self) -> usize {
        self.shared
            .reservation
            .bytes()
            .saturating_add(self.shared.preparation_metadata_bytes)
    }

    fn options<'b>(&self, options: OriginalAffineOptions<'b>) -> Result<OriginalAffineOptions<'b>> {
        if self
            .shared
            .owners
            .iter()
            .any(|owner| !owner.original_budget().shares_ledger(options.budget))
        {
            return Err(NyError::UnsupportedConfiguration(
                "original-affine chain caller ledger mismatch".into(),
            ));
        }
        Ok(OriginalAffineOptions {
            deadline: earlier(self.shared.deadline, options.deadline),
            ..options
        })
    }
    fn phase(
        &self,
        admission: WorkRequest,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffinePhase> {
        let options = self.options(options)?;
        OriginalAffinePhase::admit(options.budget, options.class, admission, options.deadline)
    }
    fn positions(&self, input: &BoundedTensor) -> Result<usize> {
        if input.shape().is_empty()
            || input.shape().len() > 16
            || input.shape().last() != Some(&self.input_features)
            || input.lower().shape() != input.upper().shape()
        {
            return Err(invalid("input endpoint/feature shape"));
        }
        let mut positions = 1usize;
        for &n in &input.shape()[..input.shape().len() - 1] {
            positions = mul(positions, n)?;
        }
        if positions == 0 || cap(mul(positions, self.input_features)?)? != input.len() {
            return Err(invalid("nonempty input geometry required"));
        }
        Ok(positions)
    }

    /// Forward facts are retained as immutable original P1 reports and priced
    /// activation facts, suitable for repeated demands under this same ledger.
    pub fn forward(
        &self,
        input: &BoundedTensor,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalChainForwardReport> {
        let options = self.options(options)?;
        let positions = self.positions(input)?;
        let metadata = add(META, mul(self.stages.len(), size_of::<Fact>())?)?;
        let mut phase = self.phase(request(interval_terms(input)?, metadata)?, options)?;
        let input_digest = hash_interval(input, &mut phase)?;
        let mut facts: Vec<Fact> = Vec::new();
        facts
            .try_reserve_exact(self.stages.len())
            .map_err(|_| invalid("forward fact allocation"))?;
        let actual = add(META, mul(facts.capacity(), size_of::<Fact>())?)?;
        if actual > metadata {
            return Err(invalid("forward fact capacity exceeded admission"));
        }
        phase.retain_metadata(actual)?;
        for stage in &self.stages {
            phase.checkpoint()?;
            let source = facts.last().map_or(input, Fact::bounds);
            let fact = match stage {
                Stage::Affine(context) => Fact::Affine(context.interval(source, options)?),
                Stage::Relu => Fact::Relu(relu::forward(source, options)?),
                Stage::GeluErf => Fact::GeluErf(activation_stage::forward(source, options)?),
                Stage::SiLu => Fact::SiLu(activation_stage::forward_profile(
                    source,
                    activation::MathematicalActivation::SiLu,
                    options,
                )?),
            };
            facts.push(fact);
        }
        let output = match facts.last() {
            Some(Fact::Affine(report)) => report.binding().output,
            _ => return Err(invalid("terminal affine fact missing")),
        };
        let binding = OriginalChainBinding {
            kernel: OriginalChainKernel::Interval,
            query: query_digest(self.scope(), OriginalChainKernel::Interval, &[input_digest]),
            output,
        };
        OriginalChainForwardReport::finish(
            facts,
            self.shared.clone(),
            positions,
            input_digest,
            binding,
            phase,
        )
    }

    fn validate_facts(&self, facts: &OriginalChainForwardReport) -> Result<()> {
        if !Arc::ptr_eq(&self.shared, facts.owner()) || facts.facts().len() != self.stages.len() {
            return Err(invalid("forward facts belong to another sealed chain"));
        }
        Ok(())
    }
    fn backward(
        &self,
        facts: &OriginalChainForwardReport,
        incoming: &LinearBounds,
        stages: usize,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffineLinearReport> {
        let mut carrier: Option<Carrier> = None;
        for index in (0..stages).rev() {
            let source = carrier.as_ref().map_or(incoming, Carrier::bounds);
            let next = match &self.stages[index] {
                Stage::Affine(context) => Carrier::Affine(context.signed_linear(source, options)?),
                Stage::Relu => {
                    let pre = facts
                        .facts()
                        .get(index - 1)
                        .ok_or_else(|| invalid("ReLU preactivation fact"))?;
                    Carrier::Relu(relu::backward(source, pre.bounds(), options)?)
                }
                Stage::GeluErf => {
                    let Some(Fact::GeluErf(activation)) = facts.facts().get(index) else {
                        return Err(invalid("GELU forward fact missing"));
                    };
                    let pre = facts
                        .facts()
                        .get(index - 1)
                        .ok_or_else(|| invalid("GELU preactivation fact"))?;
                    Carrier::GeluErf(activation_stage::backward(
                        source,
                        pre.bounds(),
                        activation,
                        options,
                    )?)
                }
                Stage::SiLu => {
                    let Some(Fact::SiLu(activation)) = facts.facts().get(index) else {
                        return Err(invalid("SiLU forward fact missing"));
                    };
                    let pre = facts
                        .facts()
                        .get(index - 1)
                        .ok_or_else(|| invalid("SiLU preactivation fact"))?;
                    Carrier::SiLu(activation_stage::backward_profile(
                        source,
                        pre.bounds(),
                        activation,
                        activation::MathematicalActivation::SiLu,
                        options,
                    )?)
                }
            };
            carrier = Some(next);
        }
        match carrier {
            Some(Carrier::Affine(report)) => Ok(report),
            _ => Err(invalid("input affine carrier missing")),
        }
    }

    /// Compose every signed requested output row to the original chain INPUT.
    /// The input box and exact demand are included in the local query binding.
    pub fn signed_linear(
        &self,
        facts: &OriginalChainForwardReport,
        incoming: &LinearBounds,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalChainLinearReport> {
        self.validate_facts(facts)?;
        let options = self.options(options)?;
        let (_, columns) = linear_shape(incoming)?;
        if columns != mul(facts.positions(), self.output_features)? {
            return Err(invalid("demand position/output shape"));
        }
        let mut phase = self.phase(request(linear_terms(incoming)?, META)?, options)?;
        let source = hash_linear(incoming, &mut phase)?;
        let output = self.backward(facts, incoming, self.stages.len(), options)?;
        let binding = OriginalChainBinding {
            kernel: OriginalChainKernel::SignedLinear,
            query: query_digest(
                self.scope(),
                OriginalChainKernel::SignedLinear,
                &[facts.input_digest(), source],
            ),
            output: output.binding().output,
        };
        OriginalChainLinearReport::finish(output, facts.clone(), binding, phase)
    }

    /// Explicit target-minus-competitor rows at exact original positions/IDs.
    /// Other vocabulary rows are not covered. The terminal selector is streamed
    /// by P1 directly over the original terminal-head INPUT, not materialized.
    pub fn sparse_margins(
        &self,
        facts: &OriginalChainForwardReport,
        differences: &[OriginalLogitDifference],
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalChainLinearReport> {
        self.validate_facts(facts)?;
        let options = self.options(options)?;
        let phase = self.phase(request(512, META)?, options)?;
        let Some(Stage::Affine(terminal)) = self.stages.last() else {
            return Err(invalid("terminal original head"));
        };
        let seed = terminal.sparse_margins(facts.positions(), differences, options)?;
        let source = seed.binding().query;
        let output = if self.stages.len() == 1 {
            seed
        } else {
            self.backward(facts, seed.bounds(), self.stages.len() - 1, options)?
        };
        let binding = OriginalChainBinding {
            kernel: OriginalChainKernel::SparseLogitDifferences,
            query: query_digest(
                self.scope(),
                OriginalChainKernel::SparseLogitDifferences,
                &[facts.input_digest(), source],
            ),
            output: output.binding().output,
        };
        OriginalChainLinearReport::finish(output, facts.clone(), binding, phase)
    }
}
