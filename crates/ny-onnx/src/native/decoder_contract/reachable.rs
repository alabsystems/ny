// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whole original GPTNeoX reachable-state computation. This interpreter never
//! accepts a hidden-state box, KV cache, logits, or an externally supplied
//! bound. Its only perturbed values are the original fixed-token embeddings.
//! All serialized coefficient bits denote exact real numbers; mathematical
//! normalization, GELU-erf, analytic RoPE and complete causal softmax are
//! enclosed by the paid scalar checker. This is not deployed Torch arithmetic.

use super::pretrained::PretrainedDecoder;
use super::{add, mul, DecoderContractRefusal as Refusal, DecoderOptions, Result};
use ny_propagate::{
    certify::work::{LeaseClass, PersistentReservation, WorkRequest},
    network::original_affine_chain::{CheckedDecoderMath, DecoderInterval as I, DecoderMathCounts},
};
use std::{sync::Arc, time::Instant};

fn numerical(error: ny_core::NyError) -> Refusal {
    Refusal::Build(error.to_string())
}
fn wide(value: f32) -> I {
    I::point(ny_core::f32_to_f64_exact(value))
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Refusal::Resource("whole decoder allocation"))?;
    if v.capacity() > n {
        return Err(Refusal::Resource("whole decoder actual capacity"));
    }
    Ok(v)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachableDecoderReceipt {
    pub complete_blocks: usize,
    pub layer_normalizations: usize,
    pub causal_attention_heads: usize,
    pub gelu_values: usize,
    pub serialized_affine_terms: u64,
    pub vocabulary_rows: usize,
    pub horizon: usize,
}

struct ReachableInner {
    tokens: Vec<usize>,
    radius: f64,
    target: usize,
    hidden: Vec<I>,
    logits: Vec<I>,
    margin: I,
    receipt: ReachableDecoderReceipt,
    deadline: Instant,
    // Derived payloads drop before their original source and paid claims.
    model: PretrainedDecoder,
    _resident: PersistentReservation,
    _arithmetic: PersistentReservation,
}

/// A sealed H=1 full-vocabulary result. Open results retain the real upstream
/// computation as well. Clone holds all original owners until the last alias.
#[derive(Clone)]
pub struct ReachableNextToken(Arc<ReachableInner>);
impl ReachableNextToken {
    pub fn model(&self) -> &PretrainedDecoder {
        &self.0.model
    }
    pub fn token_ids(&self) -> &[usize] {
        &self.0.tokens
    }
    pub fn positions(&self) -> std::ops::Range<usize> {
        0..self.0.tokens.len()
    }
    pub fn embedding_radius(&self) -> f64 {
        self.0.radius
    }
    pub fn target(&self) -> usize {
        self.0.target
    }
    pub fn receipt(&self) -> ReachableDecoderReceipt {
        self.0.receipt
    }
    pub fn deadline(&self) -> Instant {
        self.0.deadline
    }
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.0.model.checkpoint(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))?;
        Ok(())
    }
    /// Strict greedy invariance excludes every tie, including EOS.
    pub fn proven(&self) -> Result<bool> {
        self.checkpoint(None)?;
        Ok(self.0.margin.lo > 0.0)
    }
    pub fn margin(&self) -> Result<I> {
        self.checkpoint(None)?;
        Ok(self.0.margin)
    }
    pub fn logits(&self) -> Result<&[I]> {
        self.checkpoint(None)?;
        Ok(&self.0.logits)
    }
    pub fn final_hidden(&self) -> Result<&[I]> {
        self.checkpoint(None)?;
        Ok(&self.0.hidden)
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0._resident.bytes() + self.0._arithmetic.bytes()
    }
}

/// Aggregate geometry uses the actual source dimensions, not a caller-supplied
/// operation count. Every term is executed, including every vocabulary row.
fn quote(model: &PretrainedDecoder, s: usize) -> Result<(WorkRequest, DecoderMathCounts, u64)> {
    let p = model.parameters();
    let h = p.hidden_size;
    let i = p.intermediate_size;
    let l = p.num_layers;
    let affine_per = add(mul(4, mul(h, h)?)?, mul(2, mul(h, i)?)?)?;
    let affine = add(mul(mul(l, s)?, affine_per)?, mul(p.vocab_size, h)?)?;
    let keys = mul(s, add(s, 1)?)? / 2;
    let attention = mul(mul(mul(2, l)?, p.query_heads)?, mul(keys, p.head_dim)?)?;
    let terms = add(affine, attention)?;
    let nonaffine = mul(mul(32, mul(l, s)?)?, add(h, i)?)?;
    let terminal = add(mul(32, mul(s, h)?)?, mul(8, p.vocab_size)?)?;
    let elementary = add(mul(4, terms)?, add(nonaffine, terminal)?)?;
    let counts = DecoderMathCounts {
        elementary: elementary as u64,
        gelu: mul(mul(l, s)?, i)? as u64,
        roots: add(mul(2, mul(l, s)?)?, add(s, 1)?)? as u64,
        trig: mul(s, p.rotary_dim / 2)? as u64,
        exponentials: mul(mul(l, p.query_heads)?, keys)? as u64,
    };
    let scratch_cells = add(
        mul(16, mul(s, h)?)?,
        add(mul(4, mul(s, i)?)?, add(mul(4, s)?, mul(s, p.rotary_dim)?)?)?,
    )?;
    let scratch_bytes = add(mul(scratch_cells, size_of::<I>())?, 16384)?;
    let persistent_bytes = add(
        mul(add(mul(s, h)?, p.vocab_size)?, size_of::<I>())?,
        add(mul(s, size_of::<usize>())?, 4096)?,
    )?;
    Ok((
        WorkRequest {
            scratch_bytes,
            persistent_bytes,
            work: 4096,
            proposals: 0,
        },
        counts,
        affine as u64,
    ))
}

fn linear(
    model: &PretrainedDecoder,
    math: &mut CheckedDecoderMath,
    input: &[I],
    s: usize,
    weight: &str,
    bias: Option<&str>,
    out: usize,
    inner: usize,
) -> Result<Vec<I>> {
    if input.len() != mul(s, inner)? {
        return Err(Refusal::Tensor("whole decoder affine input shape".into()));
    }
    let w = model.tensor(weight, &[out, inner])?;
    let b = bias.map(|name| model.tensor(name, &[out])).transpose()?;
    let mut result = exact_vec(mul(s, out)?)?;
    for t in 0..s {
        for row in 0..out {
            math.bookkeeping(32).map_err(numerical)?;
            let mut value = b.map_or(I::point(0.0), |b| wide(b[row]));
            for k in 0..inner {
                math.bookkeeping(16).map_err(numerical)?;
                let term = math
                    .mul(wide(w[row * inner + k]), input[t * inner + k])
                    .map_err(numerical)?;
                value = math.add(value, term).map_err(numerical)?;
            }
            result.push(value);
            #[cfg(test)]
            tests::after_affine_row();
        }
    }
    Ok(result)
}

fn norm(
    model: &PretrainedDecoder,
    math: &mut CheckedDecoderMath,
    input: &[I],
    s: usize,
    prefix: &str,
) -> Result<Vec<I>> {
    let h = model.parameters().hidden_size;
    if input.len() != mul(s, h)? {
        return Err(Refusal::Tensor("whole decoder norm input shape".into()));
    }
    let w = model.tensor(&format!("{prefix}.weight"), &[h])?;
    let b = model.tensor(&format!("{prefix}.bias"), &[h])?;
    // Exact original decimal epsilon, not a rounded binary32 replacement.
    let epsilon = math
        .div_positive(I::point(1.0), I::point(100000.0))
        .map_err(numerical)?;
    let mut result = exact_vec(input.len())?;
    for t in 0..s {
        let row = &input[t * h..(t + 1) * h];
        let mut sum = I::point(0.0);
        for &x in row {
            math.bookkeeping(16).map_err(numerical)?;
            sum = math.add(sum, x).map_err(numerical)?;
        }
        let mean = math
            .div_positive(sum, I::point(h as f64))
            .map_err(numerical)?;
        let mut squares = I::point(0.0);
        for &x in row {
            let d = math.sub(x, mean).map_err(numerical)?;
            // A square of a crossing interval has exact lower zero.
            let mut square = math.mul(d, d).map_err(numerical)?;
            square.lo = if d.lo <= 0.0 && d.hi >= 0.0 {
                0.0
            } else {
                square.lo.max(0.0)
            };
            squares = math.add(squares, square).map_err(numerical)?;
        }
        squares.lo = squares.lo.max(0.0);
        let variance = math
            .div_positive(squares, I::point(h as f64))
            .map_err(numerical)?;
        let v = math.add(variance, epsilon).map_err(numerical)?;
        let denominator = math.sqrt(v).map_err(numerical)?;
        for k in 0..h {
            let d = math.sub(row[k], mean).map_err(numerical)?;
            let normalized = math.div_positive(d, denominator).map_err(numerical)?;
            let scaled = math.mul(normalized, wide(w[k])).map_err(numerical)?;
            result.push(math.add(scaled, wide(b[k])).map_err(numerical)?);
        }
    }
    Ok(result)
}

fn rotary(
    math: &mut CheckedDecoderMath,
    qkv: &mut [I],
    s: usize,
    h: usize,
    heads: usize,
    d: usize,
    r: usize,
    tables: &[(I, I)],
) -> Result<()> {
    // Original NeoX packing is [head, Q/K/V, head_dim], not three whole-head slabs.
    for t in 0..s {
        for head in 0..heads {
            for which in 0..2 {
                for k in 0..r / 2 {
                    math.bookkeeping(32).map_err(numerical)?;
                    let base = t * 3 * h + head * 3 * d + which * d;
                    let a = qkv[base + k];
                    let b = qkv[base + r / 2 + k];
                    let (sine, cosine) = tables[t * r / 2 + k];
                    let ac = math.mul(a, cosine).map_err(numerical)?;
                    let bs = math.mul(b, sine).map_err(numerical)?;
                    let bc = math.mul(b, cosine).map_err(numerical)?;
                    let ass = math.mul(a, sine).map_err(numerical)?;
                    qkv[base + k] = math.sub(ac, bs).map_err(numerical)?;
                    qkv[base + r / 2 + k] = math.add(bc, ass).map_err(numerical)?;
                }
            }
        }
    }
    Ok(())
}

fn attention(
    model: &PretrainedDecoder,
    math: &mut CheckedDecoderMath,
    qkv: &[I],
    s: usize,
    scale: I,
) -> Result<Vec<I>> {
    let p = model.parameters();
    let h = p.hidden_size;
    let d = p.head_dim;
    let mut output = exact_vec(mul(s, h)?)?;
    for i in 0..mul(s, h)? {
        if i % 64 == 0 {
            math.bookkeeping(64).map_err(numerical)?;
        }
        output.push(I::point(0.0));
    }
    for t in 0..s {
        for head in 0..p.query_heads {
            let mut scores = exact_vec(t + 1)?;
            for key in 0..=t {
                let mut score = I::point(0.0);
                for k in 0..d {
                    math.bookkeeping(16).map_err(numerical)?;
                    let product = math
                        .mul(
                            qkv[t * 3 * h + head * 3 * d + k],
                            qkv[key * 3 * h + head * 3 * d + d + k],
                        )
                        .map_err(numerical)?;
                    score = math.add(score, product).map_err(numerical)?;
                }
                scores.push(math.mul(score, scale).map_err(numerical)?);
            }
            // Any finite exact-real shift leaves softmax unchanged. The upper max
            // makes every exponent nonpositive; it is not an argmax proof.
            let mut shift = f64::NEG_INFINITY;
            for score in &scores {
                math.bookkeeping(16).map_err(numerical)?;
                shift = shift.max(score.hi);
            }
            let mut denominator = I::point(0.0);
            for score in &mut scores {
                math.bookkeeping(32).map_err(numerical)?;
                let delta = math.sub(*score, I::point(shift)).map_err(numerical)?;
                let delta = I {
                    lo: delta.lo,
                    hi: delta.hi.min(0.0),
                };
                *score = math.exponential_nonpositive(delta).map_err(numerical)?;
                denominator = math.add(denominator, *score).map_err(numerical)?;
            }
            for score in &mut scores {
                *score = math.div_positive(*score, denominator).map_err(numerical)?;
                score.lo = score.lo.max(0.0);
                score.hi = score.hi.min(1.0);
            }
            for k in 0..d {
                let mut value = I::point(0.0);
                for key in 0..=t {
                    math.bookkeeping(16).map_err(numerical)?;
                    let product = math
                        .mul(scores[key], qkv[key * 3 * h + head * 3 * d + 2 * d + k])
                        .map_err(numerical)?;
                    value = math.add(value, product).map_err(numerical)?;
                }
                output[t * h + head * d + k] = value;
            }
        }
    }
    Ok(output)
}

/// Fixed IDs select the actual pinned embedding table. Every coordinate of
/// every selected token may independently vary by the exact binary64 radius.
/// Positions are exactly 0..S-1 and masks include every and only causal key.
pub fn certify_reachable_next_token(
    model: &PretrainedDecoder,
    tokens: &[usize],
    radius: f64,
    target: usize,
    options: &DecoderOptions,
) -> Result<ReachableNextToken> {
    let at = model.checkpoint(options.deadline)?;
    let p = model.parameters();
    let s = tokens.len();
    let h = p.hidden_size;
    if s == 0
        || s > options.max_sequence
        || s > p.max_positions
        || target >= p.vocab_size
        || !radius.is_finite()
        || radius < 0.0
    {
        return Err(Refusal::Unsupported(
            "fixed-token whole decoder request shape/radius/target".into(),
        ));
    }
    let caller = options.budget.as_ref().ok_or(Refusal::Resource(
        "whole decoder request requires original ledger",
    ))?;
    if !caller.shares_ledger(model.budget()) {
        return Err(Refusal::Resource("whole decoder foreign ledger"));
    }
    let (request, counts, affine_terms) = quote(model, s)?;
    let arithmetic_request = counts.request().map_err(numerical)?;
    super::admit(options, &request)?;
    super::admit(options, &arithmetic_request)?;
    if add(request.scratch_bytes, arithmetic_request.scratch_bytes)? > options.max_scratch_bytes
        || add(
            model.reserved_bytes(),
            add(
                request.persistent_bytes,
                arithmetic_request.persistent_bytes,
            )?,
        )? > options.max_resident_bytes
    {
        return Err(Refusal::Resource("whole source and computation live cap"));
    }
    // Combined original ledger admissions precede ALL proportional allocation.
    let mut lease = caller
        .lease_for(LeaseClass::Fallback, request)
        .map_err(Refusal::Budget)?;
    let at = lease.deadline().map_or(at, |d| at.min(d));
    let mut math = CheckedDecoderMath::new(caller, Some(at), counts).map_err(numerical)?;
    let at = math.deadline().map_or(at, |d| at.min(d));
    let embed = model.tensor("gpt_neox.embed_in.weight", &[p.vocab_size, h])?;
    let mut owned_tokens = exact_vec(s)?;
    let mut state = exact_vec(mul(s, h)?)?;
    for &token in tokens {
        math.bookkeeping(32).map_err(numerical)?;
        if token >= p.vocab_size {
            return Err(Refusal::Tensor(
                "original token ID outside vocabulary".into(),
            ));
        }
        owned_tokens.push(token);
        for k in 0..h {
            let center = wide(embed[token * h + k]);
            state.push(
                math.add(
                    center,
                    I {
                        lo: -radius,
                        hi: radius,
                    },
                )
                .map_err(numerical)?,
            );
        }
    }
    let mut tables = exact_vec(mul(s, p.rotary_dim / 2)?)?;
    for position in 0..s {
        for k in 0..p.rotary_dim / 2 {
            // base10000^(2k/r) = 10^(8k/r), an INTEGER power for r=2,4,8.
            let exponent = 8 * k / p.rotary_dim;
            let denominator = 10usize
                .checked_pow(exponent as u32)
                .ok_or(Refusal::Resource("analytic rope denominator"))?;
            let angle = math
                .div_positive(I::point(position as f64), I::point(denominator as f64))
                .map_err(numerical)?;
            tables.push(math.sin_cos(angle).map_err(numerical)?);
        }
    }
    let root = math.sqrt(I::point(p.head_dim as f64)).map_err(numerical)?;
    let scale = math.div_positive(I::point(1.0), root).map_err(numerical)?;
    for layer in 0..p.num_layers {
        math.bookkeeping(128).map_err(numerical)?;
        let prefix = format!("gpt_neox.layers.{layer}");
        let norm1 = norm(
            model,
            &mut math,
            &state,
            s,
            &format!("{prefix}.input_layernorm"),
        )?;
        let mut qkv = linear(
            model,
            &mut math,
            &norm1,
            s,
            &format!("{prefix}.attention.query_key_value.weight"),
            Some(&format!("{prefix}.attention.query_key_value.bias")),
            3 * h,
            h,
        )?;
        drop(norm1);
        rotary(
            &mut math,
            &mut qkv,
            s,
            h,
            p.query_heads,
            p.head_dim,
            p.rotary_dim,
            &tables,
        )?;
        let context = attention(model, &mut math, &qkv, s, scale)?;
        drop(qkv);
        let attn = linear(
            model,
            &mut math,
            &context,
            s,
            &format!("{prefix}.attention.dense.weight"),
            Some(&format!("{prefix}.attention.dense.bias")),
            h,
            h,
        )?;
        drop(context);
        // Parallel residual MUST normalize original previous state again.
        let norm2 = norm(
            model,
            &mut math,
            &state,
            s,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
        let mut up = linear(
            model,
            &mut math,
            &norm2,
            s,
            &format!("{prefix}.mlp.dense_h_to_4h.weight"),
            Some(&format!("{prefix}.mlp.dense_h_to_4h.bias")),
            p.intermediate_size,
            h,
        )?;
        drop(norm2);
        for value in &mut up {
            *value = math.gelu(*value).map_err(numerical)?;
        }
        let down = linear(
            model,
            &mut math,
            &up,
            s,
            &format!("{prefix}.mlp.dense_4h_to_h.weight"),
            Some(&format!("{prefix}.mlp.dense_4h_to_h.bias")),
            h,
            p.intermediate_size,
        )?;
        drop(up);
        for k in 0..state.len() {
            let residual = math.add(attn[k], down[k]).map_err(numerical)?;
            state[k] = math.add(state[k], residual).map_err(numerical)?;
        }
    }
    let hidden = norm(model, &mut math, &state, s, "gpt_neox.final_layer_norm")?;
    drop(state);
    drop(tables);
    let logits = linear(
        model,
        &mut math,
        &hidden[(s - 1) * h..s * h],
        1,
        "embed_out.weight",
        None,
        p.vocab_size,
        h,
    )?;
    let mut margin = I {
        lo: f64::INFINITY,
        hi: f64::INFINITY,
    };
    for (class, &logit) in logits.iter().enumerate() {
        math.bookkeeping(32).map_err(numerical)?;
        if class == target {
            continue;
        }
        let difference = math.sub(logits[target], logit).map_err(numerical)?;
        margin.lo = margin.lo.min(difference.lo);
        margin.hi = margin.hi.min(difference.hi);
    }
    if !margin.lo.is_finite() || !margin.hi.is_finite() || margin.lo > margin.hi {
        return Err(Refusal::Build("whole vocabulary margin invalid".into()));
    }
    math.checkpoint().map_err(numerical)?;
    model.checkpoint(Some(at))?;
    let arithmetic = math.finish().map_err(numerical)?;
    let actual = add(
        mul(add(hidden.capacity(), logits.capacity())?, size_of::<I>())?,
        add(mul(owned_tokens.capacity(), size_of::<usize>())?, 4096)?,
    )?;
    if actual > request.persistent_bytes {
        return Err(Refusal::Resource("whole decoder result capacity"));
    }
    lease.mark_materialized().map_err(Refusal::Budget)?;
    lease.complete().map_err(Refusal::Budget)?;
    let resident = lease.commit_persistent().map_err(Refusal::Budget)?;
    let receipt = ReachableDecoderReceipt {
        complete_blocks: p.num_layers,
        layer_normalizations: 2 * p.num_layers + 1,
        causal_attention_heads: p.num_layers * p.query_heads,
        gelu_values: p.num_layers * s * p.intermediate_size,
        serialized_affine_terms: affine_terms,
        vocabulary_rows: p.vocab_size,
        horizon: 1,
    };
    let result = ReachableNextToken(Arc::new(ReachableInner {
        model: model.clone(),
        tokens: owned_tokens,
        radius,
        target,
        hidden,
        logits,
        margin,
        receipt,
        deadline: at,
        _resident: resident,
        _arithmetic: arithmetic,
    }));
    result.checkpoint(None)?;
    Ok(result)
}

#[cfg(test)]
#[path = "reachable_tests.rs"]
mod tests;
