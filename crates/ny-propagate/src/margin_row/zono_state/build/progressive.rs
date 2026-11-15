// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A progressively materialized ORIGINAL class chart. The complete adaptive
//! objective and every global relaxation symbol are fixed first. Advice ranks
//! which original rows to construct; it never supplies a coefficient or bound.
//! Only complete reverse passes/assemblies can append rows. Every published
//! prefix is independently usable by the existing certified node evaluator.

use super::*;

#[derive(Clone, Copy)]
struct Candidate {
    layer: usize,
    position: usize,
    original: usize,
    influence: f64,
}

/// Conservative peak including the old pass/assembly allowance, retained
/// original rows, a snapshot and its problem, growing-vector overlap, source
/// relaxation/advice, and optional complete proof-table copies. No allowance
/// is increased; admission can decline before starting a class objective.
pub(crate) fn estimate_bytes(
    frame: &FrozenDualFrame,
    opts: &FrozenDualOptions,
    max_rows: usize,
) -> u64 {
    let retain = max_rows.min(FROZEN_DUAL_MAX_DEPTH).min(frame.n_unstable) as u64;
    let n = frame.n_gens as u64;
    let rows = retain.saturating_mul(n).saturating_mul(8);
    estimate_ranked_state_bytes(frame, opts.chunk_cols, max_rows)
        .saturating_add(rows.saturating_mul(4))
        // Selected-suffix advice may score retained rows while the frontier lives.
        .saturating_add(
            n.saturating_mul(80)
                .saturating_mul(retain.min(rayon::current_num_threads().max(1) as u64)),
        )
        .saturating_add(n.saturating_mul(16))
        .saturating_add((frame.n_unstable as u64).saturating_mul(
            (size_of::<Candidate>()
                + size_of::<bool>()
                + 96
                + size_of::<crate::certify::FdLambdaRecord>()) as u64,
        ))
        .saturating_add((frame.layers.len() as u64).saturating_mul(128))
        .saturating_add(4096)
        // Previous open-stage receipt and the next level ledger coexist.
        .saturating_add(
            (FROZEN_DUAL_MAX_DEPTH as u64)
                .saturating_mul(4 * size_of::<crate::margin_row::zono_bab::LevelTrace>() as u64),
        )
}

pub(crate) struct ProgressiveClass<'a> {
    net: &'a TwinNet,
    frame: &'a FrozenDualFrame,
    relax: ClassRelax,
    state: FrozenDualState,
    candidates: Vec<Candidate>,
    built: Vec<bool>,
    original_rows: Vec<usize>,
    max_rows: usize,
    chunk: usize,
    next_batch: usize,
    max_bytes: u64,
    deadline: Option<Instant>,
    build_ms: f64,
    attempted_columns: usize,
    columns_complete: bool,
    ranking_done: bool,
}

impl<'a> ProgressiveClass<'a> {
    pub(crate) fn new(
        net: &'a TwinNet,
        frame: &'a FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
        max_rows: usize,
    ) -> Result<Self, ZonoRefusal> {
        let t0 = Instant::now();
        check_class(net, frame, class)?;
        check_deadline(opts.deadline)?;
        let estimate = estimate_bytes(frame, opts, max_rows);
        let budget = admitted_bytes(opts.max_state_bytes);
        if estimate > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            });
        }
        let mut relax = ClassRelax::chord(frame);
        let mut seed = Array2::<f64>::zeros((frame.n_out, 1));
        seed[[frame.t, 0]] = 1.0;
        seed[[class, 0]] = -1.0;
        let out = reverse_pass(
            net,
            frame,
            &mut relax,
            Choose::Every,
            frame.out_tensor,
            seed,
            opts.deadline,
        )?;
        let mut d = vec![0.0; frame.n_gens];
        let (c_obj, r_obj) = assemble(frame, &out, 0, &mut d)?;
        drop(out);
        check_deadline(opts.deadline)?;
        if !relax.matches(frame)
            || relax
                .mu
                .iter()
                .flatten()
                .any(|m| !m.is_finite() || *m <= 0.0)
            || relax
                .lam
                .iter()
                .flatten()
                .any(|l| !l.is_finite() || !(0.0..=1.0).contains(l))
        {
            return Err(ZonoRefusal::NonFinite("progressive class relaxation"));
        }
        let mut candidates = Vec::with_capacity(frame.n_unstable);
        for (li, layer) in frame.layers.iter().enumerate() {
            check_deadline(opts.deadline)?;
            for p in 0..layer.unst.len() {
                let enc = layer.col0 + p;
                candidates.push(Candidate {
                    layer: li,
                    position: p,
                    original: enc - frame.n_in_gens,
                    influence: d[enc].abs(),
                });
            }
        }
        // Advice only. Unique original IDs make ties deterministic without a
        // stable-sort allocation or constructing/scoring every original row.
        candidates.sort_unstable_by(|a, b| {
            b.influence
                .total_cmp(&a.influence)
                .then(a.original.cmp(&b.original))
        });
        check_deadline(opts.deadline)?;
        let state = FrozenDualState {
            t: frame.t,
            class,
            n_gens: frame.n_gens,
            n_in_gens: frame.n_in_gens,
            c_obj,
            d,
            r_obj,
            a: Vec::new(),
            c_in: Vec::new(),
            r: Vec::new(),
            lam: Vec::new(),
            mu: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
            enc: Vec::new(),
            layer: Vec::new(),
            site: Vec::new(),
            neuron: Vec::new(),
            stats: FrozenDualBuildStats {
                build_ms: t0.elapsed().as_secs_f64() * 1e3,
                passes: 1,
                columns: 1,
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
        };
        let build_ms = state.stats.build_ms;
        Ok(Self {
            net,
            frame,
            relax,
            state,
            candidates,
            built: vec![false; frame.n_unstable],
            original_rows: Vec::new(),
            max_rows: max_rows.min(FROZEN_DUAL_MAX_DEPTH).min(frame.n_unstable),
            chunk: opts.chunk_cols.clamp(1, MAX_DIM),
            next_batch: 1,
            max_bytes: opts.max_state_bytes,
            deadline: opts.deadline,
            build_ms,
            attempted_columns: 1,
            columns_complete: true,
            ranking_done: false,
        })
    }

    pub(crate) fn state(&self) -> &FrozenDualState {
        &self.state
    }
    #[cfg(test)]
    pub(crate) fn original_rows(&self) -> &[usize] {
        &self.original_rows
    }
    pub(crate) fn build_ms(&self) -> f64 {
        self.build_ms
    }
    pub(crate) fn attempted_columns(&self) -> usize {
        self.attempted_columns
    }

    pub(crate) fn columns_complete(&self) -> bool {
        self.columns_complete
    }

    /// Lazily materialize the original globally ranked state only after the
    /// cheap source certificate stays open. The consumer must check numeric
    /// prefix identity or DISCARD its old cover/frontier before searching this
    /// different row order. The absolute deadline and memory cap never reset.
    pub(crate) fn canonical_ranked(&mut self) -> Result<bool, ZonoRefusal> {
        self.canonical_ranked_checked(|| Ok(()))
    }

    fn canonical_ranked_checked(
        &mut self,
        before_commit: impl FnOnce() -> Result<(), ZonoRefusal>,
    ) -> Result<bool, ZonoRefusal> {
        let began = Instant::now();
        let result = self.canonical_ranked_inner(before_commit);
        self.build_ms += began.elapsed().as_secs_f64() * 1e3;
        if result.is_ok() {
            self.state.stats.build_ms = self.build_ms;
        }
        result
    }

    fn canonical_ranked_inner(
        &mut self,
        before_commit: impl FnOnce() -> Result<(), ZonoRefusal>,
    ) -> Result<bool, ZonoRefusal> {
        if self.ranking_done {
            return Ok(false);
        }
        check_deadline(self.deadline)?;
        let budget = admitted_bytes(self.max_bytes);
        if self.state.stats.estimate_bytes > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: self.state.stats.estimate_bytes,
                budget_bytes: budget,
            });
        }
        let opts = FrozenDualOptions {
            deadline: self.deadline,
            max_state_bytes: self.max_bytes,
            chunk_cols: self.chunk,
        };
        let ranked = FrozenDualState::build_ranked_with_relax(
            self.net,
            self.frame,
            self.state.class,
            &opts,
            self.max_rows,
        );
        let (ranked, table) = match ranked {
            Ok(value) => value,
            Err(error) => {
                self.columns_complete = false;
                return Err(error);
            }
        };
        self.attempted_columns = self
            .attempted_columns
            .saturating_add(ranked.state.stats.columns);
        let same = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if ranked.state.class != self.state.class
            || ranked.state.t != self.state.t
            || ranked.state.n_gens != self.state.n_gens
            || ranked.state.n_in_gens != self.state.n_in_gens
            || ranked.total_unstable != self.frame.n_unstable
            || !same(&ranked.state.d, &self.state.d)
            || !same(
                &[ranked.state.c_obj, ranked.state.r_obj],
                &[self.state.c_obj, self.state.r_obj],
            )
            || !table.relax.matches(self.frame)
            || !same(
                &table
                    .relax
                    .lam
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                &self.relax.lam.iter().flatten().copied().collect::<Vec<_>>(),
            )
            || !same(
                &table.relax.mu.iter().flatten().copied().collect::<Vec<_>>(),
                &self.relax.mu.iter().flatten().copied().collect::<Vec<_>>(),
            )
        {
            return Err(ZonoRefusal::Internal(
                "canonical continuation changed fixed original objective/relaxation".into(),
            ));
        }
        check_deadline(self.deadline)?;
        before_commit()?;
        check_deadline(self.deadline)?;
        let mut next = ranked.state;
        next.stats.passes = next.stats.passes.saturating_add(self.state.stats.passes);
        next.stats.columns = next.stats.columns.saturating_add(self.state.stats.columns);
        next.stats.estimate_bytes = self.state.stats.estimate_bytes;
        next.stats.budget_bytes = self.state.stats.budget_bytes;
        // A complete independent original state is published atomically. Its
        // row order may differ; only the search owner can decide continuation.
        self.state = next;
        self.original_rows = ranked.original_rows;
        self.built.fill(false);
        for &original in &self.original_rows {
            self.built[original] = true;
        }
        self.ranking_done = true;
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn cancel_canonical_before_commit_for_test(&mut self) -> Result<bool, ZonoRefusal> {
        self.canonical_ranked_checked(|| Err(ZonoRefusal::Deadline))
    }

    /// The builder stays alive while a snapshot is searched. Charge its
    /// actual capacities, complete relaxation and selection storage as resident.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.state
            .bytes()
            .saturating_add(self.relax.heap_bytes())
            .saturating_add(vector_bytes(&self.candidates))
            .saturating_add(vector_bytes(&self.built))
            .saturating_add(vector_bytes(&self.original_rows))
            .saturating_add(size_of::<Self>() as u64)
    }

    /// Caller verifies this complete table through the original verifier
    /// before using it in proof capture. It is not a source of search bounds.
    pub(crate) fn compact_relax(&self) -> CompactRelax {
        CompactRelax {
            class: self.state.class,
            t: self.frame.t,
            layout: layout_digest(self.frame),
            relax: self.relax.clone(),
        }
    }

    pub(crate) fn advance(&mut self) -> Result<bool, ZonoRefusal> {
        self.advance_checked(|| Ok(()))
    }

    fn advance_checked(
        &mut self,
        before_commit: impl FnOnce() -> Result<(), ZonoRefusal>,
    ) -> Result<bool, ZonoRefusal> {
        let t0 = Instant::now();
        let result = self.advance_inner(before_commit);
        self.build_ms += t0.elapsed().as_secs_f64() * 1e3;
        if result.is_ok() {
            self.state.stats.build_ms = self.build_ms;
        }
        result
    }

    fn advance_inner(
        &mut self,
        before_commit: impl FnOnce() -> Result<(), ZonoRefusal>,
    ) -> Result<bool, ZonoRefusal> {
        check_deadline(self.deadline)?;
        let budget = admitted_bytes(self.max_bytes);
        if self.state.stats.estimate_bytes > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: self.state.stats.estimate_bytes,
                budget_bytes: budget,
            });
        }
        let old_rows = self.state.num_unstable();
        if old_rows >= self.max_rows {
            return Ok(false);
        }
        let Some(first) = self
            .candidates
            .iter()
            .find(|c| !self.built[c.original])
            .copied()
        else {
            return Ok(false);
        };
        let count = self
            .next_batch
            .min(self.chunk)
            .min(self.max_rows - old_rows);
        // Amortize one original reverse pass over candidates at the same
        // layer. The next stage again starts from the highest unbuilt advice.
        let selected: Vec<Candidate> = self
            .candidates
            .iter()
            .filter(|c| c.layer == first.layer && !self.built[c.original])
            .take(count)
            .copied()
            .collect();
        let layer = &self.frame.layers[first.layer];
        let mut seed = Array2::<f64>::zeros((self.net.tsize[layer.input], selected.len()));
        for (col, c) in selected.iter().enumerate() {
            seed[[layer.unst[c.position], col]] = 1.0;
        }
        self.attempted_columns += selected.len();
        let out = reverse_pass(
            self.net,
            self.frame,
            &mut self.relax,
            Choose::Never,
            layer.input,
            seed,
            self.deadline,
        )?;
        check_deadline(self.deadline)?;
        let n = self.state.n_gens;
        let old_len = self.state.a.len();
        let extra = selected
            .len()
            .checked_mul(n)
            .ok_or_else(|| ZonoRefusal::Shape("progressive row size overflows".into()))?;
        self.state
            .a
            .try_reserve_exact(extra)
            .map_err(|_| ZonoRefusal::Memory {
                estimate_bytes: self.state.stats.estimate_bytes,
                budget_bytes: admitted_bytes(self.max_bytes),
            })?;
        self.state.a.resize(old_len + extra, 0.0);
        let mut centres = Vec::with_capacity(selected.len());
        let assembled = (|| {
            for (col, c) in selected.iter().enumerate() {
                check_deadline(self.deadline)?;
                let row = &mut self.state.a[old_len + col * n..old_len + (col + 1) * n];
                let value = assemble(self.frame, &out, col, row)?;
                if row[layer.col0 + c.position] != 0.0 {
                    return Err(ZonoRefusal::Internal(
                        "progressive row reads its own symbol".into(),
                    ));
                }
                centres.push(value);
            }
            check_deadline(self.deadline)?;
            before_commit()
        })();
        if let Err(error) = assembled {
            self.state.a.truncate(old_len);
            return Err(error);
        }
        // Commit only a wholly assembled batch under the unchanged relaxation.
        for (c, (centre, remainder)) in selected.iter().zip(centres) {
            let p = c.position;
            let j = layer.unst[p];
            self.state.c_in.push(centre);
            self.state.r.push(remainder);
            self.state.lam.push(self.relax.lam[c.layer][p]);
            self.state.mu.push(self.relax.mu[c.layer][p]);
            self.state.lo.push(layer.l[j]);
            self.state.hi.push(layer.u[j]);
            self.state.enc.push(layer.col0 + p);
            self.state.layer.push(c.layer);
            self.state.site.push(layer.site);
            self.state.neuron.push(j);
            self.original_rows.push(c.original);
            self.built[c.original] = true;
        }
        if let Err(error) = check_deadline(self.deadline) {
            self.state.a.truncate(old_len);
            self.state.c_in.truncate(old_rows);
            self.state.r.truncate(old_rows);
            self.state.lam.truncate(old_rows);
            self.state.mu.truncate(old_rows);
            self.state.lo.truncate(old_rows);
            self.state.hi.truncate(old_rows);
            self.state.enc.truncate(old_rows);
            self.state.layer.truncate(old_rows);
            self.state.site.truncate(old_rows);
            self.state.neuron.truncate(old_rows);
            self.original_rows.truncate(old_rows);
            for c in selected {
                self.built[c.original] = false;
            }
            return Err(error);
        }
        self.state.stats.passes += 1;
        self.state.stats.columns += selected.len();
        self.next_batch = self.next_batch.saturating_mul(2).min(self.chunk);
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn cancel_before_commit_for_test(&mut self) -> Result<bool, ZonoRefusal> {
        self.advance_checked(|| Err(ZonoRefusal::Deadline))
    }
}
