// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn dyadic_chain() -> PairedChain {
    let delta = 1.0 / 1024.0;
    PairedChain {
        input_dim: 2,
        output_dim: 2,
        stages: vec![
            Stage::Affine {
                rows: 3,
                cols: 2,
                wf: vec![1.0, 0.5, -0.5, 1.0, 0.25, -0.75],
                wg: vec![1.0 + delta, 0.5, -0.5, 1.0 - delta, 0.25, -0.75 + delta],
                bf: vec![0.25, -0.5, 0.0],
                bg: vec![0.25 + delta, -0.5, delta],
            },
            Stage::Relu { width: 3 },
            Stage::Affine {
                rows: 3,
                cols: 3,
                wf: vec![0.5, 0.25, -0.5, -0.25, 0.75, 0.5, 1.0, -0.5, 0.25],
                wg: vec![
                    0.5,
                    0.25 + delta,
                    -0.5,
                    -0.25,
                    0.75,
                    0.5,
                    1.0 - delta,
                    -0.5,
                    0.25,
                ],
                bf: vec![0.0, 0.125, -0.25],
                bg: vec![delta, 0.125, -0.25],
            },
            Stage::Relu { width: 3 },
            Stage::Affine {
                rows: 2,
                cols: 3,
                wf: vec![1.0, -1.0, 0.5, -0.5, 0.25, 1.0],
                wg: vec![1.0, -1.0, 0.5 + delta, -0.5, 0.25 - delta, 1.0],
                bf: vec![0.0, 0.25],
                bg: vec![0.0, 0.25 + delta],
            },
        ],
    }
}

fn trace(chain: &PairedChain, input: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut f = input.to_vec();
    let mut g = f.clone();
    let mut trace = vec![(f.clone(), g.clone())];
    for stage in &chain.stages {
        match stage {
            Stage::Affine {
                rows,
                cols,
                wf,
                wg,
                bf,
                bg,
            } => {
                let mut nf = bf.clone();
                let mut ng = bg.clone();
                for i in 0..*rows {
                    for j in 0..*cols {
                        nf[i] += wf[i * cols + j] * f[j];
                        ng[i] += wg[i * cols + j] * g[j];
                    }
                }
                f = nf;
                g = ng;
            }
            Stage::Relu { .. } => {
                for v in f.iter_mut().chain(&mut g) {
                    *v = v.max(0.0);
                }
            }
        }
        trace.push((f.clone(), g.clone()));
    }
    trace
}

#[test]
fn progressive_faces_are_subsets_and_preserve_exact_dyadic_traces() {
    let chain = dyadic_chain();
    let domain = BoxDomain {
        lo: vec![-1.0, -0.5],
        hi: vec![1.0, 0.5],
    };
    let original = forward::states(&chain, &domain, None, true).unwrap();
    let refined = states(&chain, &domain, None, true).unwrap();
    assert_eq!(refined.len(), original.len());
    for (old, new) in original.iter().zip(&refined) {
        for j in 0..old.d_lo.len() {
            assert!(new.f_lo[j] >= old.f_lo[j] && new.f_hi[j] <= old.f_hi[j]);
            assert!(new.g_lo[j] >= old.g_lo[j] && new.g_hi[j] <= old.g_hi[j]);
            assert!(new.d_lo[j] >= old.d_lo[j] && new.d_hi[j] <= old.d_hi[j]);
        }
    }
    // Inputs, weights and biases are small dyadics; every operation in this
    // trace has fewer than 53 significant bits. These forward values equal
    // their exact stored-real mathematical values, without a float tolerance.
    for a in 0..=32 {
        for b in 0..=16 {
            let values = trace(&chain, &[-1.0 + a as f64 / 16.0, -0.5 + b as f64 / 16.0]);
            for (s, (f, g)) in values.iter().enumerate() {
                for j in 0..f.len() {
                    let delta = f[j] - g[j];
                    assert!(refined[s].f_lo[j] <= f[j] && f[j] <= refined[s].f_hi[j]);
                    assert!(refined[s].g_lo[j] <= g[j] && g[j] <= refined[s].g_hi[j]);
                    assert!(
                        refined[s].d_lo[j] <= delta && delta <= refined[s].d_hi[j],
                        "state {s}, neuron {j}: {} <= {delta} <= {}",
                        refined[s].d_lo[j],
                        refined[s].d_hi[j]
                    );
                }
            }
        }
    }
}

#[test]
fn certified_root_references_preserve_every_child_trace_and_backward_bound() {
    let chain = dyadic_chain();
    let root = BoxDomain {
        lo: vec![-1.0, -0.5],
        hi: vec![1.0, 0.5],
    };
    // References originate from the exact same private source/root certifier.
    // No external bounds or candidate-authored reference faces are accepted.
    let reference = states(&chain, &root, None, true).unwrap();
    for (xl, xh) in [(-1.0, 0.0), (0.0, 1.0)] {
        for (yl, yh) in [(-0.5, 0.0), (0.0, 0.5)] {
            let child = BoxDomain {
                lo: vec![xl, yl],
                hi: vec![xh, yh],
            };
            let child_states =
                forward::states_with_reference(&chain, &child, None, true, Some(&reference))
                    .unwrap();
            let backward = chain
                .bound_with_reference_until(&child, true, None, None, Some(&reference))
                .unwrap();
            assert_eq!(child_states.len(), reference.len());
            assert_eq!(backward.lower.len(), chain.output_dim);
            assert_eq!(backward.upper.len(), chain.output_dim);
            for (state, root_state) in child_states.iter().zip(&reference) {
                for j in 0..state.f_lo.len() {
                    assert!(state.f_lo[j] >= root_state.f_lo[j]);
                    assert!(state.f_hi[j] <= root_state.f_hi[j]);
                    assert!(state.g_lo[j] >= root_state.g_lo[j]);
                    assert!(state.g_hi[j] <= root_state.g_hi[j]);
                    assert!(state.d_lo[j] >= root_state.d_lo[j]);
                    assert!(state.d_hi[j] <= root_state.d_hi[j]);
                }
            }
            let output_state = child_states.last().unwrap();
            for j in 0..chain.output_dim {
                assert!(backward.lower[j] >= output_state.d_lo[j]);
                assert!(backward.upper[j] <= output_state.d_hi[j]);
            }
            // The child grids, authored weights, and biases are small dyadics.
            // Every operation remains below 53 significant bits, so these
            // traces are exact real values and require no floating tolerance.
            for a in 0..=16 {
                for b in 0..=16 {
                    let input = [xl + a as f64 / 16.0, yl + b as f64 / 32.0];
                    let values = trace(&chain, &input);
                    for (s, (f, g)) in values.iter().enumerate() {
                        for j in 0..f.len() {
                            let delta = f[j] - g[j];
                            assert!(child_states[s].f_lo[j] <= f[j]);
                            assert!(f[j] <= child_states[s].f_hi[j]);
                            assert!(child_states[s].g_lo[j] <= g[j]);
                            assert!(g[j] <= child_states[s].g_hi[j]);
                            assert!(child_states[s].d_lo[j] <= delta);
                            assert!(delta <= child_states[s].d_hi[j]);
                        }
                    }
                    let (f, g) = values.last().unwrap();
                    for j in 0..chain.output_dim {
                        let delta = f[j] - g[j];
                        assert!(backward.lower[j] <= delta);
                        assert!(delta <= backward.upper[j]);
                    }
                }
            }
        }
    }
}

#[test]
fn correlated_hidden_copies_tighten_each_pre_relu_tower_box() {
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![0.0],
                bg: vec![0.0],
            },
            Stage::Relu { width: 1 },
            // These authored hidden variables are exactly equal, but a
            // forward interval sum cannot correlate their uncertain faces.
            Stage::Affine {
                rows: 2,
                cols: 1,
                wf: vec![1.0, 1.0],
                wg: vec![1.0, 1.0],
                bf: vec![0.0, 0.0],
                bg: vec![0.0, 0.0],
            },
            Stage::Affine {
                rows: 1,
                cols: 2,
                wf: vec![1.0, -1.0],
                wg: vec![1.0, -1.0],
                bf: vec![0.0],
                bg: vec![0.0],
            },
            Stage::Relu { width: 1 },
        ],
    };
    let domain = BoxDomain {
        lo: vec![-1.0],
        hi: vec![1.0],
    };
    let original = forward::states(&chain, &domain, None, true).unwrap();
    let refined = states(&chain, &domain, None, true).unwrap();
    for (old_lo, old_hi, new_lo, new_hi) in [
        (
            original[4].f_lo[0],
            original[4].f_hi[0],
            refined[4].f_lo[0],
            refined[4].f_hi[0],
        ),
        (
            original[4].g_lo[0],
            original[4].g_hi[0],
            refined[4].g_lo[0],
            refined[4].g_hi[0],
        ),
    ] {
        assert!(old_hi - old_lo > 0.9);
        assert!(new_lo <= 0.0 && new_hi >= 0.0);
        assert!(new_hi - new_lo < 1e-10);
        assert!(new_lo >= old_lo && new_hi <= old_hi);
    }
    // Every trace operation and cancellation is exact for these dyadics.
    for k in 0..=128 {
        for (s, (f, g)) in trace(&chain, &[-1.0 + k as f64 / 64.0]).iter().enumerate() {
            for j in 0..f.len() {
                assert!(refined[s].f_lo[j] <= f[j] && f[j] <= refined[s].f_hi[j]);
                assert!(refined[s].g_lo[j] <= g[j] && g[j] <= refined[s].g_hi[j]);
                assert!(refined[s].d_lo[j] <= f[j] - g[j] && f[j] - g[j] <= refined[s].d_hi[j]);
            }
        }
    }
    // Hulls are rebuilt from the newly tightened state, rather than left
    // prepared from the old wide box. Here 0 >= f + k requires k <= -uf;
    // the original hull requires k near -1 and cannot pass this tight check.
    let k = refined[4].hulls.as_ref().unwrap()[0]
        .certified_intercept(0.0, 0.0, 1.0, 0.0)
        .unwrap();
    assert!(k <= 0.0 && k > -1e-10);
}

#[test]
fn identical_twins_keep_zero_inside_every_refined_strip() {
    let mut chain = dyadic_chain();
    for stage in &mut chain.stages {
        if let Stage::Affine { wf, wg, bf, bg, .. } = stage {
            *wg = wf.clone();
            *bg = bf.clone();
        }
    }
    let domain = BoxDomain {
        lo: vec![-1.0, -0.5],
        hi: vec![1.0, 0.5],
    };
    let refined = states(&chain, &domain, None, true).unwrap();
    for state in refined {
        for j in 0..state.d_lo.len() {
            assert!(state.d_lo[j] <= 0.0 && state.d_hi[j] >= 0.0);
            assert!(magnitude(state.d_lo[j], state.d_hi[j]) < 1e-9);
        }
    }
}

#[test]
fn refinement_declines_expiry_and_malformed_domains_without_partial_states() {
    let chain = dyadic_chain();
    let domain = BoxDomain {
        lo: vec![-1.0, -0.5],
        hi: vec![1.0, 0.5],
    };
    assert!(states(&chain, &domain, Some(Instant::now()), true).is_none());
    let bad = BoxDomain {
        lo: vec![f64::NAN, 0.0],
        hi: vec![1.0, 0.5],
    };
    assert!(states(&chain, &bad, None, true).is_none());
    let bad = BoxDomain {
        lo: vec![1.0, 0.0],
        hi: vec![-1.0, 0.5],
    };
    assert!(states(&chain, &bad, None, true).is_none());
    let mut bad_chain = chain;
    bad_chain.stages[1] = Stage::Relu { width: 4 };
    assert!(states(&bad_chain, &domain, None, true).is_none());
}
