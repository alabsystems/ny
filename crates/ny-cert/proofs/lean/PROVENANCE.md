<!-- Copyright 2026 Andrew Yates -->
<!-- Author: Andrew Yates <andrewyates.name@gmail.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Lean proof ownership and dependency provenance

This directory is an NY-owned Lean overlay. It does **not** vendor Clean source.
The local Lake package is `nyproof`, its library is `NyProof`, and Clean's
`Crownproof` library is consumed as a normal Git dependency:

```text
repository: https://github.com/alabsystems/clean.git
revision:   bc05e5f8b2fb4336dc20112503dfe0d90f80c69d
subDir:     crown-proofs/lean
```

The pin is declared in `lakefile.toml` and recorded in `lake-manifest.json`.
Local modules live under `NyProof/` and import each other as `NyProof.*`;
`Crownproof.*` imports resolve to the pinned Clean dependency. Existing theorem
namespaces are kept where useful (several overlay modules still declare
`Crownproof.*` namespaces), so moving `Crownproof/Foo.lean` to
`NyProof/Foo.lean` does not rename public theorem constants.

The root also pins Mathlib to the v4.30.0 release commit
`c5ea00351c28e24afc9f0f84379aa41082b1188f`. Without that pin, Lake resolved
Clean's Mathlib requirement to an older rc2 snapshot whose toolchain did not
match Lean v4.30.0.

## Removed Clean-origin mirrors

These 78 modules were byte-identical to Clean at the pin and have been removed
from NY. Each name denotes `crown-proofs/lean/Crownproof/<name>.lean` in Clean,
which is their single source of truth. Do not copy them back.

- `AcasTree8`, `AcasWholeBox`, `BabProof`, `Block`, `Block2`, `Branch`,
  `BranchTree`, `CakeRepro`
- `CertChecker`, `CertCheckerZ`, `CertEquiv`, `CertRealZ_node10`,
  `CertRunZ_node10`
- `Complete`, `CompleteCrown`, `CompleteCrownVector`, `CompleteDeep`,
  `CompleteGeneralDepth`, `CompleteIBP`, `CompleteSafenlpReal`,
  `CompleteVector`
- `ConvHullCoupled`, `ConvHullCoupledGeneral`, `ConvHullExact`, `Deep`,
  `DeepK`, `DeepPair`, `DeepPairL3`, `DerivedLN`, `FacetCut`
- `Gelu`, `GeluFull`, `Hull3D`, `HullArrangement2D`,
  `HullArrangement2DGeneral`, `HullArrangement2DK3`, `HullKGeneral`, `HullND`
- `LayerNorm`, `McCormick`, `MetaroomConvPair`, `MetaroomConvPair4cnn`,
  `MetaroomConvPairB`, `MultiHead`, `MultiReluCutK`
- `NetAcas2Layer`, `NetAcas3Layer`, `NetAcas7Layer`, `NetAcasLayer`,
  `NetAcasWide`, `NetOnnx`, `NetSmall`, `Network`, `Quant`,
  `ReflectDecodeSpike`, `Rsqrt`, `SafenlpRealSlice`, `Sbar`
- `SlackCertZ`, `SlackFarkas`, `SoftmaxBridge`, `SoftmaxOp`, `StackTwo`,
  `TinyBlock`, `TinyBlockWitness`, `TwoReluCut`, `TwoReluCutAcas`,
  `TwoReluCutGeneral`
- `VFBao`, `VFBbn1`, `VFBbn2`, `VFBcompose`, `VFBhid`, `VFBmlp`,
  `VFBvalue`, `Variance`, `VitFullBlock`, `VitRealAttention`

## NY-origin proofs adopted by Clean

Three modules were authored in NY first and later adopted by Clean. At the pin,
Clean carries the exact NY-authored blobs, so NY imports the upstream copies
instead of keeping duplicate modules. Their NY authorship is in this
repository's history:

| Module | First NY commit | Clean adoption commit |
|---|---|---|
| `Basic` | `75ae60d2be2b08f2dc73608732fc13c0c21fa8f3` (2026-05-30) | `0f9933ea09b567a242aee725f55337370e59e2c1` (2026-05-30) |
| `Bridge` | `335c90d474c7d2f9d443c140d1d7bbcbbe27c939` (2026-05-30) | `0f9933ea09b567a242aee725f55337370e59e2c1` (2026-05-30) |
| `Pow2Envelope` | `f647603a099c260e3fc8b2e7881154172c482a01` (2026-07-02) | `17b1eff857a5bb6dce7d8dd843e7de5f4bf3aa71` (2026-07-08) |

## Retained NY-owned overlay

The 70 modules under `NyProof/` are NY-origin and are not present in Clean at
the pin.

Fifteen reusable proof modules:

- `AristotleLemmas`, `CersyveInduction`, `CertifiedDecision`,
  `FarkasInterval`, `FloatAdequacy`, `IntegerInterval`, `MeanValueChain`,
  `MeanValueForm`, `RupChecker`, `RupCheckerFast`, `SatReluCnf`,
  `SatReluGadget`, `SatReluVerdict`, `SignFusion`, `SoftmaxFloatRange`

Fifty-five NY-specific diagnostic, audit, instance, and generated modules:

- `AlphaGradientDefect`, `AxiomAudit`
- `CersyveInstance_DoubleIntegrator`, `CersyveInstance_Pendulum`,
  `CersyveInstance_Unicycle`
- `SatReluDemo_v10c26`, `SatReluDemo_v92c117`, `SatReluDemo_v99c485`,
  `SatReluDemo_v100c373`
- `SatReluSweepAll`
- the 45 generated modules under `NyProof/SatReluSweep/`:
  `V6C30`, `V9C9`, `V9C14`, `V14C57`, `V16C39`, `V18C88`, `V20C40`,
  `V24C66`, `V25C40`, `V25C45`, `V26C84`, `V30C38`, `V30C44`,
  `V33C140`, `V34C98`, `V37C63`, `V38C132`, `V39C77`, `V40C142`,
  `V44C52`, `V46C142`, `V51C255`, `V54C125`, `V56C239`, `V58C75`,
  `V63C267`, `V63C286`, `V64C234`, `V65C187`, `V67C154`, `V70C178`,
  `V73C156`, `V75C99`, `V82C130`, `V83C328`, `V85C186`, `V85C207`,
  `V87C243`, `V90C111`, `V90C449`, `V91C217`, `V92C180`, `V94C313`,
  `V94C377`, `V95C275`

`NyProof.lean` imports, directly or transitively, every retained module except
two: `AxiomAudit`, a diagnostic that emits `#print axioms` output, and
`AlphaGradientDefect`, which nothing imports. `lake build NyProof` therefore
does not check either; build them by module name.

## Reproducible validation

Run with no Cargo/Targo/rustc build competing for resources:

```sh
lake update
lake exe cache get
lake build NyProof
lake build NyProof.AxiomAudit
lake env lean NyProof/AxiomAudit.lean > AXIOM_AUDIT.txt
```

Build `NyProof.AxiomAudit` before invoking Lean on that file directly; a stale
dependency `.olean` makes the audit machine-dependent and can fail before any
axiom is checked.

The committed `AXIOM_AUDIT.txt` is captured output. Regenerate it after changing
the Clean pin or any overlay proof; `sorryAx`, an unexpected axiom, or an
unknown constant is a soundness regression.
