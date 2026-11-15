"""Meaningful fail-closed checks for the experiment's property/ONNX bridge."""
import importlib.util
import argparse
import sys
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

SPEC = importlib.util.spec_from_file_location("paired_exporter", Path(__file__).with_name("export_paired_difference_instances.py"))
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
ROOT = Path("benchmarks/vnncomp2026/benchmarks/isomorphic_acasxu_2026/2.0")


class ExportChecks(unittest.TestCase):
    def test_directed_rounding_contains_exact_decimal(self):
        for value in ("0.6", "-0.45", "0.679857769", "0.05", "1e-40", "-1e-40", "0"):
            exact = Decimal(value)
            lower = exporter.directed_f32(value, False)
            upper = exporter.directed_f32(value, True)
            self.assertLessEqual(Decimal.from_float(lower), exact)
            self.assertGreaterEqual(Decimal.from_float(upper), exact)

    def test_extra_constraint_cannot_be_silently_dropped(self):
        original = (ROOT / "vnnlib/instance_4.vnnlib").read_text()
        mutations = [original + "\n(assert (> X_f[0,0,0,0] 0.7))\n",
                     original + "\n(foo ignored)\n",
                     original.replace("X_g[0,0,0,0]))", "X_g[0,0,0,1]))", 1),
                     original.replace("(> Y_g[0,4] (+ Y_f[0,4] 0.05))", "(> Y_g[0,4] (+ Y_f[0,4] 0.06))"),
                     original.replace("declare-input X_f real", "declare-input X_f float32")]
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(mutations):
                path = Path(directory) / f"bad_{index}.vnnlib"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    exporter.parse_property(path)

    def test_initializers_widen_exactly_and_affines_remain_separate(self):
        path = ROOT / "onnx/original/ACASXU_run2a_2_7_batch_2000.onnx"
        model = onnx.load(path)
        stages = exporter.source_stages(path)
        self.assertEqual(len(stages), len(model.graph.node) - 1)
        self.assertEqual([s.get("source_op") for s in stages[:4]], ["Sub", "MatMul", "Add", None])
        constants = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        np.testing.assert_array_equal(np.asarray(stages[1]["weights"]).T, constants["Operation_1_MatMul_W"].astype(np.float64))
        np.testing.assert_array_equal(np.asarray(stages[2]["bias"]), constants["Operation_1_Add_B"].astype(np.float64))

    def test_extra_node_is_rejected(self):
        model = onnx.load(ROOT / "onnx/original/ACASXU_run2a_2_7_batch_2000.onnx")
        model.graph.node.append(onnx.helper.make_node("Relu", ["input"], ["ignored_extra_node"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.onnx"
            onnx.save(model, path)
            with self.assertRaises(ValueError):
                exporter.source_stages(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark-root", type=Path)
    options, unittest_options = parser.parse_known_args()
    if options.benchmark_root:
        ROOT = options.benchmark_root
    unittest.main(argv=[sys.argv[0], *unittest_options])
