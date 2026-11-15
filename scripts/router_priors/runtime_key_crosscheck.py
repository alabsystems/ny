#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Cross-check the prior's TRAINING key against its RUNTIME key on every training model (#router-r7 review).

The pipeline keys a training unit by `priorslib.key_bits` over the static features table
(`instance_features.tsv`: LIVE-node op flags). The router keys a row by `CostKey::from_facts` over the plan
resolver's protobuf scan (`plan_resolver::scan_onnx_graph` -> `router::facts::op_kinds`: every node, operand
constness from INITIALIZERS only). This tool re-implements that scan and mapping in pure Python (a SECOND
implementation: it can drift from the Rust source; the authoritative check needs a binary) and compares, per
training model, the (conv, nonpwl) key and the parameter count.

    python3 scripts/router_priors/runtime_key_crosscheck.py \
        --features <home>/ny/scratchpad/vbs/general/instance_features.tsv \
        --benchmarks <home>/ny/benchmarks
    # last line: KEY_CROSSCHECK_DONE rc=<0|1> models=N agree=N key_mismatch=N params_mismatch=N missing=N

Known definitional differences (none fires on the v1 training corpus, measured 2026-10-02): a Mul / MatMul /
Gemm / Div whose constant operand is a `Constant` NODE is bilinear at runtime but static in the features
table; Group/InstanceNormalization is `nonpwl` at runtime only; the features table ignores dead nodes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import priorslib as P  # noqa: E402

NONPWL_KINDS = {"sigmoid", "tanh", "softmax", "bilinear", "norm"}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return r, i


def _fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
            yield f, "v", v
        elif w == 1:
            i += 8
            yield f, "s", None
        elif w == 2:
            n, i = _varint(b, i)
            yield f, "b", b[i:i + n]
            i += n
        elif w == 5:
            i += 4
            yield f, "s", None
        else:
            raise ValueError("unsupported wire type")


def scan(model: bytes) -> tuple[int, int, set]:
    """(param_count, conv_layers, op_tokens): `plan_resolver::scan_onnx_graph`, token for token."""
    graph = None
    for f, w, v in _fields(model):
        if f == 7 and w == "b":
            graph = v
    if graph is None:
        raise ValueError("no graph field")
    init: dict = {}
    tokens: set = set()
    operand: list = []
    params = conv_layers = 0
    for f, w, v in _fields(graph):
        if f == 5 and w == "b":
            dims, name = [], None
            for tf, tw, tv in _fields(v):
                if tf == 1 and tw == "v":
                    dims.append(tv)
                elif tf == 1 and tw == "b":
                    j = 0
                    while j < len(tv):
                        d, j = _varint(tv, j)
                        dims.append(d)
                elif tf == 8 and tw == "b":
                    name = tv.decode("utf-8", "replace")
            e = 1
            for d in dims:
                e *= d
            params += e
            if name is not None:
                init[name] = dims
        elif f == 1 and w == "b":
            op, inputs = "", []
            for nf, nw, nv in _fields(v):
                if nf == 1 and nw == "b":
                    inputs.append(nv.decode("utf-8", "replace"))
                elif nf == 4 and nw == "b":
                    op = nv.decode("utf-8", "replace")
            if op in ("Conv", "ConvTranspose"):
                conv_layers += 1
            if op in ("Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "Conv", "ConvTranspose"):
                operand.append((op, inputs))
            else:
                tokens.add(op)
    for op, inputs in operand:
        def fixed(k: int) -> bool:
            return k < len(inputs) and inputs[k] in init
        if op == "Div":
            t = "Div:const" if fixed(1) else "Div"
        elif op == "Gemm":
            t = "Gemm:const" if (fixed(0) or fixed(1)) else "Gemm"
        elif op in ("Conv", "ConvTranspose"):
            t = op if fixed(1) else op + ":dynamic"
        elif any(x in init for x in inputs):
            t = op + ":const"
        else:
            t = op
        tokens.add(t)
    return params, conv_layers, tokens


def kinds(token: str) -> set:
    """The subset of `router::facts::op_kinds` the cost key reads."""
    return {
        "Conv": {"conv"}, "ConvTranspose": {"conv"},
        "Conv:dynamic": {"conv", "bilinear"}, "ConvTranspose:dynamic": {"conv", "bilinear"},
        "Gemm": {"bilinear"}, "MatMul": {"bilinear"}, "Mul": {"bilinear"}, "Div": {"bilinear"},
        "Sigmoid": {"sigmoid"}, "Tanh": {"tanh"}, "Softmax": {"softmax"}, "LogSoftmax": {"softmax"},
        "LayerNormalization": {"norm"}, "GroupNormalization": {"norm"}, "InstanceNormalization": {"norm"},
    }.get(token, set())


def runtime_key(path: str) -> tuple[int, int, int, bool, list]:
    data = open(path, "rb").read()
    if path.endswith(".gz"):
        data = gzip.decompress(data)
    params, conv_layers, tokens = scan(data)
    ks: set = set()
    for t in tokens:
        ks |= kinds(t)
    conv = int(conv_layers > 0 or "conv" in ks)
    nonpwl = int(bool(ks & NONPWL_KINDS))
    why = sorted(t for t in tokens if kinds(t) & NONPWL_KINDS)
    return params, conv, nonpwl, bool(tokens), why


def model_path(bench: str, feat: dict) -> str | None:
    base = os.path.join(bench, f"vnncomp{feat['year']}", "benchmarks", feat["category"])
    rel = feat["onnx"][2:] if feat["onnx"].startswith("./") else feat["onnx"]
    stems = [rel, rel + ".gz"] + ([rel[:-3]] if rel.endswith(".gz") else [])
    for sub in ("", "1.0", "2.0"):
        for r in stems:
            p = os.path.join(base, sub, r)
            if os.path.isfile(p):
                return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainset", default=os.path.join(HERE, "data", "priors_v1_trainset.tsv"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--benchmarks", required=True)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.benchmarks):
        print(f"benchmarks root does not resolve: {a.benchmarks}")
        print("KEY_CROSSCHECK_DONE rc=2 refused=1")
        return 2
    feats: dict = {}
    with open(a.features, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            feats.setdefault(r["model_md5"], r)
    models: dict = {}
    for r in P.read_rows(a.trainset):
        models.setdefault(r.model, r)
    agree = key_mis = par_mis = missing = 0
    for md5, r in sorted(models.items()):
        f = feats.get(md5)
        path = model_path(a.benchmarks, f) if f else None
        if path is None:
            missing += 1
            print(f"MISSING {md5} {r.category}")
            continue
        params, conv, nonpwl, complete, why = runtime_key(path)
        ok = True
        if (conv, nonpwl) != (r.conv, r.nonpwl) or not complete:
            key_mis += 1
            ok = False
            print(f"KEY_MISMATCH {md5} {r.category} train={(r.conv, r.nonpwl)} runtime={(conv, nonpwl)} "
                  f"complete={int(complete)} nonpwl_tokens={','.join(why) or '-'}")
        if params != int(r.params):
            par_mis += 1
            ok = False
            print(f"PARAMS_MISMATCH {md5} {r.category} train={int(r.params)} runtime={params}")
        agree += int(ok)
    rc = 0 if (key_mis == 0 and par_mis == 0 and missing == 0) else 1
    print(f"KEY_CROSSCHECK_DONE rc={rc} models={len(models)} agree={agree} key_mismatch={key_mis} "
          f"params_mismatch={par_mis} missing={missing}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
