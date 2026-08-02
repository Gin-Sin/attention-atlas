"""Tests for the generated implementation asset and PyTorch references."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNC_PATH = ROOT / "tools" / "sync_pytorch_examples.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except (ImportError, OSError) as torch_error:
    torch = None
    TORCH_SKIP_REASON = f"PyTorch is unavailable: {torch_error}"
else:
    TORCH_SKIP_REASON = ""


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_pytorch_examples", SYNC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SYNC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MarkerAndAssetTests(unittest.TestCase):
    """These synchronization tests deliberately do not require PyTorch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sync = load_sync_module()

    def test_chapter_source_mapping_is_complete_and_ordered(self) -> None:
        self.assertEqual(
            list(self.sync.CHAPTER_SOURCES.items()),
            [
                ("mha", "pytorch/mha.py"),
                ("mqa", "pytorch/mqa.py"),
                ("gqa", "pytorch/gqa.py"),
                ("mla", "pytorch/mla.py"),
                ("dsa", "pytorch/dsa.py"),
                ("csa", "pytorch/csa.py"),
                ("hca", "pytorch/hca.py"),
                ("linear", "pytorch/linear_attention.py"),
                ("gated-delta", "pytorch/gated_delta.py"),
                ("kda", "pytorch/kda.py"),
            ],
        )

    def test_all_markers_are_contiguous_and_code_matches_line_ranges(self) -> None:
        implementations = self.sync.build_implementations(ROOT)
        self.assertEqual(
            list(implementations),
            list(self.sync.CHAPTER_SOURCES),
        )

        for chapter_id, implementation in implementations.items():
            with self.subTest(chapter=chapter_id):
                source = implementation["source"]
                lines = source.splitlines(keepends=True)
                blocks = implementation["blocks"]
                self.assertTrue(blocks)
                self.assertEqual(
                    [block["id"] for block in blocks],
                    [f"{number:02d}" for number in range(1, len(blocks) + 1)],
                )
                for block in blocks:
                    start, end = block["start"], block["end"]
                    self.assertLessEqual(start, end)
                    self.assertEqual(
                        block["code"],
                        "".join(lines[start - 1 : end]),
                    )
                    self.assertNotIn("# [Block ", block["code"])
                    self.assertNotIn("# [/Block ", block["code"])

    def test_extractor_rejects_invalid_marker_structures(self) -> None:
        invalid_sources = {
            "gap": (
                "# [Block 01] First\nx = 1\n# [/Block 01]\n"
                "# [Block 03] Third\nx = 3\n# [/Block 03]\n"
            ),
            "mismatch": "# [Block 01] First\nx = 1\n# [/Block 02]\n",
            "nested": (
                "# [Block 01] First\n# [Block 02] Nested\n"
                "# [/Block 02]\n# [/Block 01]\n"
            ),
            "orphan": "# [/Block 01]\n",
            "unclosed": "# [Block 01] First\nx = 1\n",
            "malformed": "# [Block 1] First\nx = 1\n# [/Block 1]\n",
        }
        for label, source in invalid_sources.items():
            with self.subTest(case=label):
                with self.assertRaises(self.sync.MarkerError):
                    self.sync.extract_blocks(source, label)

    def test_generated_browser_asset_is_synchronized_and_parseable(self) -> None:
        implementations = self.sync.build_implementations(ROOT)
        expected = self.sync.render_asset(implementations)
        asset_path = ROOT / "assets" / "implementations.js"
        actual = asset_path.read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        self.assertTrue(actual.startswith("window.ATTENTION_IMPLEMENTATIONS = {"))
        self.assertTrue(actual.endswith(";\n"))

        payload = actual.removeprefix(
            "window.ATTENTION_IMPLEMENTATIONS = "
        ).removesuffix(";\n")
        decoded = json.loads(payload)
        self.assertEqual(list(decoded), list(self.sync.CHAPTER_SOURCES))
        self.assertEqual(decoded, implementations)

    def test_check_mode_reports_current_asset(self) -> None:
        self.assertTrue(self.sync.synchronize(check=True, root=ROOT))

    def test_check_mode_rejects_a_stale_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path in self.sync.CHAPTER_SOURCES.values():
                source_path = root / relative_path
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(
                    "# [Block 01] Example\nvalue = 1\n# [/Block 01]\n",
                    encoding="utf-8",
                )
            asset_path = root / "assets" / "implementations.js"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_text("stale\n", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(self.sync.synchronize(check=True, root=root))
            self.assertEqual(asset_path.read_text(encoding="utf-8"), "stale\n")
            self.assertIn("is stale", stderr.getvalue())


@unittest.skipUnless(torch is not None, TORCH_SKIP_REASON)
class PyTorchReferenceTests(unittest.TestCase):
    """Tiny CPU forwards tailored to each reference module's public API."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.modules = {
            name: importlib.import_module(f"pytorch.{module_name}")
            for name, module_name in {
                "mha": "mha",
                "mqa": "mqa",
                "gqa": "gqa",
                "mla": "mla",
                "dsa": "dsa",
                "csa": "csa",
                "hca": "hca",
                "linear": "linear_attention",
                "gated-delta": "gated_delta",
                "kda": "kda",
            }.items()
        }
        torch.manual_seed(1234)

    def assert_finite_shape(self, tensor, shape) -> None:
        self.assertEqual(tuple(tensor.shape), shape)
        self.assertTrue(torch.isfinite(tensor).all().item())

    def test_mha_tiny_forward(self) -> None:
        module = self.modules["mha"]
        model = module.MultiHeadAttention(d_model=16, num_heads=4).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, cache = model(x, use_cache=True)
        self.assert_finite_shape(output, (2, 5, 16))
        self.assertIsNotNone(cache)
        self.assert_finite_shape(cache[0], (2, 4, 5, 4))
        self.assert_finite_shape(cache[1], (2, 4, 5, 4))

    def test_mqa_tiny_forward(self) -> None:
        module = self.modules["mqa"]
        model = module.MultiQueryAttention(d_model=16, num_query_heads=4).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, cache = model(x, use_cache=True)
        self.assert_finite_shape(output, (2, 5, 16))
        self.assertIsNotNone(cache)
        self.assert_finite_shape(cache[0], (2, 1, 5, 4))
        self.assert_finite_shape(cache[1], (2, 1, 5, 4))

    def test_gqa_tiny_forward(self) -> None:
        module = self.modules["gqa"]
        model = module.GroupedQueryAttention(
            d_model=16,
            num_query_heads=4,
            num_kv_heads=2,
        ).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, cache = model(x, use_cache=True)
        self.assert_finite_shape(output, (2, 5, 16))
        self.assertIsNotNone(cache)
        self.assert_finite_shape(cache[0], (2, 2, 5, 4))
        self.assert_finite_shape(cache[1], (2, 2, 5, 4))

    def test_mla_tiny_forward(self) -> None:
        module = self.modules["mla"]
        model = module.MultiHeadLatentAttention(
            d_model=16,
            num_heads=2,
            q_lora_rank=6,
            kv_lora_rank=4,
            qk_content_dim=4,
            qk_rope_dim=2,
            value_dim=4,
        ).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, cache = model(x, use_cache=True, implementation="absorbed")
            rebuilt, _ = model(x, implementation="reconstruct")
        self.assert_finite_shape(output, (2, 5, 16))
        self.assert_finite_shape(rebuilt, (2, 5, 16))
        torch.testing.assert_close(output, rebuilt, rtol=1e-5, atol=1e-6)
        self.assertIsNotNone(cache)
        self.assert_finite_shape(cache[0], (2, 5, 4))
        self.assert_finite_shape(cache[1], (2, 1, 5, 2))

    def test_dsa_tiny_forward(self) -> None:
        module = self.modules["dsa"]
        model = module.DynamicSparseAttention(
            d_model=16,
            num_heads=4,
            index_dim=4,
            num_index_heads=2,
            top_k=3,
            rotary_dim=4,
        ).eval()
        x = torch.randn(2, 6, 16)
        with torch.no_grad():
            teacher = model.dense_teacher_probs(x)
            output, aux = model(x, teacher_probs=teacher, return_aux=True)
        self.assert_finite_shape(output, (2, 6, 16))
        self.assert_finite_shape(aux["selected_indices"], (2, 6, 3))
        self.assertTrue(torch.isfinite(aux["indexer_kl"]).item())

    def test_csa_tiny_forward(self) -> None:
        module = self.modules["csa"]
        model = module.CompressedSparseAttention(
            d_model=16,
            num_heads=4,
            compression_block=2,
            top_k=2,
            local_window=3,
            index_dim=4,
            num_index_heads=2,
            rotary_dim=4,
        ).eval()
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            output, aux = model(x, return_aux=True)
        self.assert_finite_shape(output, (2, 7, 16))
        self.assert_finite_shape(aux["summaries"], (2, 3, 16))
        self.assertTrue(torch.isfinite(aux["attention_probs"]).all().item())

    def test_hca_tiny_forward(self) -> None:
        module = self.modules["hca"]
        model = module.HeavilyCompressedAttention(
            d_model=16,
            num_heads=4,
            compression_block=2,
            local_window=3,
            rotary_dim=4,
        ).eval()
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            output, aux = model(x, return_aux=True)
        self.assert_finite_shape(output, (2, 7, 16))
        self.assert_finite_shape(aux["compressed_kv"], (2, 3, 32))
        self.assertTrue(torch.isfinite(aux["attention_probs"]).all().item())

    def test_linear_attention_tiny_forward(self) -> None:
        module = self.modules["linear"]
        model = module.CausalLinearAttention(d_model=16, num_heads=4).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            recurrent, state = model(x, mode="recurrent")
            prefix, prefix_state = model(x, mode="prefix")
        self.assert_finite_shape(recurrent, (2, 5, 16))
        self.assert_finite_shape(prefix, (2, 5, 16))
        torch.testing.assert_close(recurrent, prefix, rtol=1e-5, atol=1e-6)
        self.assert_finite_shape(state.s, (2, 4, 4, 4))
        self.assert_finite_shape(state.z, (2, 4, 4))
        self.assert_finite_shape(prefix_state.s, (2, 4, 4, 4))

    def test_gated_delta_tiny_forward(self) -> None:
        module = self.modules["gated-delta"]
        model = module.GatedDeltaAttention(
            d_model=16,
            num_heads=4,
            conv_kernel_size=3,
        ).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, state = model(x)
        self.assert_finite_shape(output, (2, 5, 16))
        self.assert_finite_shape(state.memory, (2, 4, 4, 4))
        self.assert_finite_shape(state.conv_cache, (2, 2, 16))

    def test_kda_tiny_forward_and_hybrid(self) -> None:
        module = self.modules["kda"]
        model = module.KimiDeltaAttention(
            d_model=12,
            num_heads=3,
            conv_kernel_size=3,
        ).eval()
        hybrid = module.EducationalKDAHybrid(
            d_model=12,
            num_heads=3,
            num_layers=4,
            kda_per_global=3,
            conv_kernel_size=3,
        ).eval()
        x = torch.randn(2, 5, 12)
        with torch.no_grad():
            output, state = model(x)
            hybrid_output = hybrid(x)
        self.assert_finite_shape(output, (2, 5, 12))
        self.assert_finite_shape(state.memory, (2, 3, 4, 4))
        self.assert_finite_shape(state.conv_cache, (2, 2, 12))
        self.assert_finite_shape(hybrid_output, (2, 5, 12))
        self.assertEqual(hybrid.schedule, ("kda", "kda", "kda", "global"))


if __name__ == "__main__":
    unittest.main()
