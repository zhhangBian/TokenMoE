import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tokenmoe_collect.estimate import formula, measured


class EstimateTests(unittest.TestCase):
    def test_qwen_planning_estimate_and_dtype_boundary(self):
        result = formula(layers=48, top_k=8, experts=128)
        self.assertEqual(result["bytes_per_category_per_session"]["expert_ids"], 29952000)
        self.assertAlmostEqual(result["bytes_per_session"] / 1e6, 54.576)
        wide = formula(layers=48, top_k=8, experts=257)
        self.assertEqual(wide["bytes_per_category_per_session"]["expert_ids"], 59904000)

    def test_measured_counts_hardlinked_artifacts_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime").mkdir()
            (root / "runtime/sessions.jsonl").write_text('{"session_id":"ses_a"}\n')
            (root / "runtime/llm_requests.jsonl").write_text(
                '{"row_start":2,"row_end":9,"num_prompt_tokens":5,"num_output_tokens":5}\n'
            )
            (root / "raw/routing").mkdir(parents=True)
            (root / "runtime/routing").mkdir()
            np.savez(root / "raw/routing/req_a.npz", token_positions=[2, 3, 7, 8])
            os.link(root / "raw/routing/req_a.npz", root / "runtime/routing/req_a.npz")
            result = measured(root, layers=2, top_k=2, experts=128)
            self.assertEqual(
                result["measured_categories"]["routing"]["bytes"],
                (root / "raw/routing/req_a.npz").stat().st_size,
            )
            self.assertEqual(result["computed_tokens"], 4)
