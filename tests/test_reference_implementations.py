"""Tests for the generated implementation asset and PyTorch references."""

from __future__ import annotations

import ast
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
                ("mfa", "pytorch/mfa.py"),
                ("tpa", "pytorch/tpa.py"),
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


class StaticRecurrentArchitectureTests(unittest.TestCase):
    """AST checks for paper-critical paths; these do not import PyTorch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = {
            name: (ROOT / relative_path).read_text(encoding="utf-8")
            for name, relative_path in {
                "gated-delta": "pytorch/gated_delta.py",
                "kda": "pytorch/kda.py",
            }.items()
        }
        cls.trees = {
            name: ast.parse(source)
            for name, source in cls.sources.items()
        }

    def method(self, chapter: str, class_name: str, method_name: str):
        tree = self.trees[chapter]
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return next(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )

    def function(self, chapter: str, function_name: str):
        return next(
            node
            for node in self.trees[chapter].body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )

    def call_arguments(self, node, attribute: str) -> list[str]:
        calls = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == attribute
        ]
        return [ast.unparse(call.args[0]) for call in calls]

    def evaluate_schedule_expression(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(
                self.evaluate_schedule_expression(element)
                for element in node.elts
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return (
                self.evaluate_schedule_expression(node.left)
                + self.evaluate_schedule_expression(node.right)
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return (
                self.evaluate_schedule_expression(node.left)
                * self.evaluate_schedule_expression(node.right)
            )
        raise AssertionError(f"unsupported schedule expression: {ast.dump(node)}")

    def test_gdn_qkv_shortconv_and_direct_gate_inputs(self) -> None:
        forward = self.method(
            "gated-delta", "GatedDeltaAttention", "forward"
        )
        self.assertEqual(
            self.call_arguments(forward, "q_conv1d"),
            ["self.q_proj(x)"],
        )
        self.assertEqual(
            self.call_arguments(forward, "k_conv1d"),
            ["self.k_proj(x)"],
        )
        self.assertEqual(
            self.call_arguments(forward, "v_conv1d"),
            ["self.v_proj(x)"],
        )
        self.assertEqual(self.call_arguments(forward, "alpha_proj"), ["x"])
        self.assertEqual(self.call_arguments(forward, "beta_proj"), ["x"])
        self.assertEqual(
            self.call_arguments(forward, "output_gate_proj"),
            ["x"],
        )

        forward_text = ast.unparse(forward)
        self.assertIn("q = self._heads(F.silu(q))", forward_text)
        self.assertIn("k = self._heads(F.silu(k))", forward_text)
        self.assertIn("v = self._heads(F.silu(v))", forward_text)
        self.assertIn("output_gate = F.silu", forward_text)
        self.assertIn("log_alpha = -torch.exp(self.A_log)", forward_text)
        recurrence = ast.unparse(
            self.function("gated-delta", "gated_delta_recurrence")
        )
        self.assertIn("q = F.normalize(q", recurrence)
        self.assertIn("k = F.normalize(k", recurrence)
        self.assertIn("S = F^T", self.sources["gated-delta"])

    def test_gdn_exposes_conceptual_progression_recurrences(self) -> None:
        """The chapter narrative needs additive → delta → gated functions."""
        function_names = {
            node.name
            for node in self.trees["gated-delta"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertLessEqual(
            {"additive_recurrence", "delta_recurrence", "gated_delta_recurrence"},
            function_names,
        )

        additive = ast.unparse(
            self.function("gated-delta", "additive_recurrence")
        )
        # The baseline must stay a blind append: no normalization, no gates.
        self.assertNotIn("normalize", additive)
        self.assertIn("memory + torch.einsum", additive)

        delta = self.function("gated-delta", "delta_recurrence")
        delta_args = {argument.arg for argument in delta.args.args}
        self.assertEqual(
            delta_args, {"q", "k", "v", "beta", "memory", "norm_eps"}
        )
        delta_text = ast.unparse(delta)
        self.assertIn("q = F.normalize(q", delta_text)
        self.assertIn("k = F.normalize(k", delta_text)
        self.assertIn("error = v[:, t] - prediction", delta_text)

    def test_kda_qkv_shortconv_and_low_rank_direct_gates(self) -> None:
        forward = self.method("kda", "KimiDeltaAttention", "forward")
        self.assertEqual(
            self.call_arguments(forward, "q_conv1d"),
            ["self.q_proj(x)"],
        )
        self.assertEqual(
            self.call_arguments(forward, "k_conv1d"),
            ["self.k_proj(x)"],
        )
        self.assertEqual(
            self.call_arguments(forward, "v_conv1d"),
            ["self.v_proj(x)"],
        )
        self.assertEqual(self.call_arguments(forward, "alpha_down"), ["x"])
        self.assertEqual(self.call_arguments(forward, "beta_proj"), ["x"])
        self.assertEqual(
            self.call_arguments(forward, "output_gate_down"),
            ["x"],
        )

        forward_text = ast.unparse(forward)
        self.assertIn("self.alpha_up(self.alpha_down(x))", forward_text)
        self.assertIn(
            "log_alpha = -torch.exp(self.A_log)", forward_text
        )
        self.assertIn("F.softplus(alpha_logits + self.alpha_bias)", forward_text)
        self.assertIn(
            "self.output_gate_up(self.output_gate_down(x))",
            forward_text,
        )
        self.assertIn("torch.sigmoid(output_gate_logits)", forward_text)
        recurrence = ast.unparse(self.function("kda", "kda_recurrence"))
        self.assertIn("q = F.normalize(q", recurrence)
        self.assertIn("k = F.normalize(k", recurrence)
        self.assertIn("S = F^T", self.sources["kda"])

    def test_released_27_layer_schedule_and_nope_description(self) -> None:
        assignment = next(
            node
            for node in self.trees["kda"].body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "KIMI_LINEAR_27_LAYER_SCHEDULE"
                for target in node.targets
            )
        )
        schedule = self.evaluate_schedule_expression(assignment.value)
        expected = (
            ("kda", "kda", "kda", "mla") * 6
            + ("kda", "kda", "mla")
        )
        self.assertEqual(schedule, expected)
        self.assertEqual(len(schedule), 27)
        self.assertIn("low-rank MLA", self.sources["kda"])
        self.assertIn("token-indexed", self.sources["kda"])


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
                "mfa": "mfa",
                "tpa": "tpa",
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

        # The absorbed (decode-flavoured) graph must never call the value
        # up-projection module; doing so would mean it silently reconstructed
        # per-head historical values.
        value_up_calls = []
        hook = model.value_up_proj.register_forward_pre_hook(
            lambda *_: value_up_calls.append(1)
        )
        try:
            with torch.no_grad():
                # Prefill-flavoured MHA-like graph: explicit K/V expansion.
                prefill, prefill_cache = model(
                    x, use_cache=True, implementation="reconstruct"
                )
                self.assertEqual(len(value_up_calls), 1)

                # Absorbed MQA-like graph over the full sequence.
                absorbed, absorbed_cache = model(
                    x, use_cache=True, implementation="absorbed"
                )
                self.assertEqual(
                    len(value_up_calls), 1,
                    "absorbed graph reconstructed values",
                )

                # Token-by-token absorbed decode against the growing cache.
                decode_cache = None
                pieces = []
                for step in range(x.size(1)):
                    piece, decode_cache = model(
                        x[:, step : step + 1],
                        kv_cache=decode_cache,
                        use_cache=True,
                        implementation="absorbed",
                    )
                    pieces.append(piece)
                decoded = torch.cat(pieces, dim=1)
                self.assertEqual(
                    len(value_up_calls), 1,
                    "absorbed decode reconstructed values",
                )
        finally:
            hook.remove()

        self.assert_finite_shape(prefill, (2, 5, 16))
        self.assert_finite_shape(absorbed, (2, 5, 16))
        self.assert_finite_shape(decoded, (2, 5, 16))
        torch.testing.assert_close(absorbed, prefill, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(decoded, prefill, rtol=1e-5, atol=1e-6)
        for cache in (prefill_cache, absorbed_cache, decode_cache):
            self.assertIsNotNone(cache)
            # Persistent cache is latent + shared RoPE key (Dc + Dr per
            # token), not per-head K/V, in both execution graphs.
            self.assert_finite_shape(cache[0], (2, 5, 4))
            self.assert_finite_shape(cache[1], (2, 1, 5, 2))
        torch.testing.assert_close(absorbed_cache[0], prefill_cache[0])
        torch.testing.assert_close(decode_cache[0], prefill_cache[0])
        torch.testing.assert_close(decode_cache[1], prefill_cache[1])

    def test_mla_absorbed_branches_never_call_projection_modules(self) -> None:
        """Static guard: the absorbed branches must reuse weights, not modules.

        Calling ``value_up_proj``/``key_up_proj``/``out_proj`` inside an
        ``implementation == "absorbed"`` branch would reconstruct per-head
        activations and defeat the absorbed execution graph.
        """
        source = (ROOT / "pytorch" / "mla.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MultiHeadLatentAttention"
        )
        forward = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

        def is_absorbed_test(test: ast.expr) -> bool:
            return (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "implementation"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "absorbed"
            )

        absorbed_branches = [
            node.body
            for node in ast.walk(forward)
            if isinstance(node, ast.If) and is_absorbed_test(node.test)
        ]
        self.assertGreaterEqual(len(absorbed_branches), 2)
        forbidden = {"value_up_proj", "key_up_proj", "out_proj"}
        for branch in absorbed_branches:
            for statement in branch:
                for child in ast.walk(statement):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr in forbidden
                    ):
                        self.fail(
                            "absorbed branch calls projection module "
                            f"{child.func.attr}; fold its weight instead"
                        )

    def test_mfa_tiny_forward_standard_and_key_reuse(self) -> None:
        module = self.modules["mfa"]
        # Default matches the paper's common settings (RoPE base 500,000).
        self.assertEqual(
            module.MultiMatrixFactorizationAttention(
                d_model=16, num_heads=3, shared_dim=6
            ).rope_base,
            500_000.0,
        )
        for key_reuse in (False, True):
            with self.subTest(key_reuse=key_reuse):
                model = module.MultiMatrixFactorizationAttention(
                    d_model=16,
                    num_heads=3,
                    shared_dim=6,
                    key_reuse=key_reuse,
                ).eval()
                x = torch.randn(2, 5, 16)
                with torch.no_grad():
                    output, cache = model(x, use_cache=True)
                self.assert_finite_shape(output, (2, 5, 16))
                self.assertIsNotNone(cache)
                # The shared cache never gains a head axis.
                self.assert_finite_shape(cache[0], (2, 5, 6))
                if key_reuse:
                    self.assertIsNone(cache[1])
                else:
                    self.assert_finite_shape(cache[1], (2, 5, 6))

                decode_cache = None
                pieces = []
                with torch.no_grad():
                    for step in range(x.size(1)):
                        piece, decode_cache = model(
                            x[:, step : step + 1],
                            kv_cache=decode_cache,
                            use_cache=True,
                        )
                        pieces.append(piece)
                torch.testing.assert_close(
                    torch.cat(pieces, dim=1), output, rtol=1e-5, atol=1e-6
                )

    def test_tpa_tiny_forward_factorized_matches_reconstruct(self) -> None:
        module = self.modules["tpa"]
        model = module.TensorProductAttention(
            d_model=16,
            num_heads=5,
            head_dim=4,
            q_rank=3,
            k_rank=2,
            v_rank=2,
        ).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            factorized, cache = model(
                x, use_cache=True, implementation="factorized"
            )
            reconstructed, _ = model(x, implementation="reconstruct")
        self.assert_finite_shape(factorized, (2, 5, 16))
        self.assert_finite_shape(reconstructed, (2, 5, 16))
        torch.testing.assert_close(
            factorized, reconstructed, rtol=1e-5, atol=1e-6
        )
        self.assertIsNotNone(cache)
        self.assert_finite_shape(cache[0], (2, 5, 2, 5))  # a_k
        self.assert_finite_shape(cache[1], (2, 5, 2, 4))  # rotated b_k
        self.assert_finite_shape(cache[2], (2, 5, 2, 5))  # a_v
        self.assert_finite_shape(cache[3], (2, 5, 2, 4))  # b_v

        decode_cache = None
        pieces = []
        with torch.no_grad():
            for step in range(x.size(1)):
                piece, decode_cache = model(
                    x[:, step : step + 1],
                    kv_cache=decode_cache,
                    use_cache=True,
                    implementation="factorized",
                )
                pieces.append(piece)
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), factorized, rtol=1e-5, atol=1e-6
        )

    def test_dsa_tiny_forward(self) -> None:
        module = self.modules["dsa"]
        model = module.DynamicSparseAttention(
            d_model=16,
            num_heads=4,
            q_lora_rank=6,
            kv_lora_rank=5,
            qk_content_dim=4,
            qk_rope_dim=2,
            value_dim=4,
            index_dim=4,
            num_index_heads=2,
            top_k=3,
        ).eval()
        x = torch.randn(2, 6, 16)
        with torch.no_grad():
            teacher = model.dense_teacher_probs(x)
            output, full_cache, aux = model(
                x, use_cache=True, teacher_probs=teacher, return_aux=True
            )
        self.assert_finite_shape(output, (2, 6, 16))
        self.assert_finite_shape(aux["selected_indices"], (2, 6, 3))
        self.assertTrue(torch.isfinite(aux["indexer_kl"]).item())
        self.assertIsNotNone(full_cache)
        self.assert_finite_shape(full_cache[0], (2, 6, 5))
        self.assert_finite_shape(full_cache[1], (2, 1, 6, 2))
        self.assertEqual(full_cache[2].shape, (2, 6, 4))
        self.assertEqual(full_cache[2].dtype, torch.float8_e4m3fn)
        self.assert_finite_shape(full_cache[3], (2, 6, 1))

        cache = None
        pieces = []
        with torch.no_grad():
            for step in range(x.size(1)):
                piece, cache = model(
                    x[:, step : step + 1], kv_cache=cache, use_cache=True
                )
                pieces.append(piece)
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), output, rtol=1e-5, atol=1e-6
        )
        self.assertTrue(
            torch.equal(
                cache[2].to(torch.float32), full_cache[2].to(torch.float32)
            )
        )
        torch.testing.assert_close(cache[3], full_cache[3])

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
        gate_inputs = {}

        def capture_gate(name):
            def hook(_module, inputs):
                gate_inputs[name] = inputs[0].detach().clone()

            return hook

        handles = [
            layer.register_forward_pre_hook(capture_gate(name))
            for name, layer in {
                "alpha": model.alpha_proj,
                "beta": model.beta_proj,
                "output": model.output_gate_proj,
            }.items()
        ]
        with torch.no_grad():
            output, state = model(x)
        for handle in handles:
            handle.remove()
        self.assert_finite_shape(output, (2, 5, 16))
        self.assert_finite_shape(state.memory, (2, 4, 4, 4))
        self.assertEqual(len(state.conv_cache), 3)
        for cache in state.conv_cache:
            self.assert_finite_shape(cache, (2, 2, 16))
        for gate_input in gate_inputs.values():
            torch.testing.assert_close(gate_input, x)

    def test_gated_delta_progression_equivalences(self) -> None:
        module = self.modules["gated-delta"]

        # A reused key pollutes the additive memory but is exactly
        # overwritten by the delta rule with beta = 1 and a unit-norm key.
        key = torch.nn.functional.normalize(
            torch.randn(1, 1, 1, 4), p=2.0, dim=-1
        )
        keys = key.expand(1, 2, 1, 4)
        values = torch.randn(1, 2, 1, 3)
        full_write = torch.ones(1, 2, 1)
        additive_out, additive_memory = module.additive_recurrence(
            keys, keys, values
        )
        delta_out, _ = module.delta_recurrence(keys, keys, values, full_write)
        self.assert_finite_shape(additive_out, (1, 2, 1, 3))
        self.assert_finite_shape(additive_memory, (1, 1, 4, 3))
        torch.testing.assert_close(delta_out[:, -1], values[:, -1])
        self.assertFalse(
            torch.allclose(additive_out[:, -1], values[:, -1]),
            "additive memory should interfere on a reused key",
        )

        # alpha = 1 collapses Gated DeltaNet onto the pure delta rule.
        q = torch.randn(2, 6, 4, 4)
        k = torch.randn(2, 6, 4, 4)
        v = torch.randn(2, 6, 4, 5)
        beta = torch.rand(2, 6, 4)
        gated_out, gated_memory = module.gated_delta_recurrence(
            q, k, v, torch.ones(2, 6, 4), beta
        )
        delta_ref, delta_memory = module.delta_recurrence(q, k, v, beta)
        torch.testing.assert_close(gated_out, delta_ref)
        torch.testing.assert_close(gated_memory, delta_memory)

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
            conv_kernel_size=3,
        ).eval()
        x = torch.randn(2, 5, 12)
        gate_inputs = {}

        def capture_gate(name):
            def hook(_module, inputs):
                gate_inputs[name] = inputs[0].detach().clone()

            return hook

        handles = [
            layer.register_forward_pre_hook(capture_gate(name))
            for name, layer in {
                "alpha": model.alpha_down,
                "beta": model.beta_proj,
                "output": model.output_gate_down,
            }.items()
        ]
        with torch.no_grad():
            output, state = model(x)
            hybrid_output = hybrid(x)
        for handle in handles:
            handle.remove()
        self.assert_finite_shape(output, (2, 5, 12))
        self.assert_finite_shape(state.memory, (2, 3, 4, 4))
        self.assertEqual(len(state.conv_cache), 3)
        for cache in state.conv_cache:
            self.assert_finite_shape(cache, (2, 2, 12))
        self.assert_finite_shape(hybrid_output, (2, 5, 12))
        self.assertEqual(
            hybrid.schedule,
            ("kda", "kda", "kda", "mla") * 6
            + ("kda", "kda", "mla"),
        )
        for gate_input in gate_inputs.values():
            torch.testing.assert_close(gate_input, x)
        self.assertEqual(model.alpha_down.out_features, model.head_dim)
        self.assertEqual(model.output_gate_down.out_features, model.head_dim)


if __name__ == "__main__":
    unittest.main()
