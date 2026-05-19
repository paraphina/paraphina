import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
V2_MODULE = REPO_ROOT / "paraphina/src/v2/mod.rs"
RUNNER = REPO_ROOT / "paraphina/src/live/runner.rs"


class TestV2ShadowSkeletonIsolation(unittest.TestCase):
    def test_v2_module_has_no_execution_surface_calls(self):
        source = V2_MODULE.read_text()
        forbidden_patterns = [
            r"\bLiveGateway\b",
            r"\bGatewayMux\b",
            r"\bShadowAckAdapter\b",
            r"\bPaperExecutionAdapter\b",
            r"\bplan_mm_order_actions\s*\(",
            r"\border_tx\b",
            r"\bpriority_order_tx\b",
            r"\bsubmit_intent\b",
            r"\bsubmit_batch\b",
            r"\bplace_order\s*\(",
            r"\bcancel_order\s*\(",
            r"\breplace_order\s*\(",
            r"\bsendTx\b",
            r"\bsendTxBatch\b",
            r"\bnormalize_live_client_order_ids\s*\(",
            r"\bregister_mm_decision_lineage\s*\(",
            r"\bcommit_stage\s*\(",
        ]
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(
                    re.search(pattern, source),
                    f"V2 shadow skeleton must not touch execution surface: {pattern}",
                )

    def test_v2_module_emits_hold_only_no_order_authority(self):
        source = V2_MODULE.read_text()
        self.assertIn('event_type: "V2_SHADOW_DECISION"', source)
        self.assertIn("V2ShadowAdmissionStatus::Hold", source)
        self.assertIn('admission_reason: "shadow_only_no_order_authority"', source)
        self.assertIn("can_mutate_orders: false", source)
        self.assertIn("order_intent_output_count: 0", source)
        self.assertIn("ranking_feature_only: true", source)
        self.assertIn("ranking_is_admission: false", source)
        self.assertIn("pair_edge_is_admission: false", source)
        self.assertIn("pressure_complete_claim: false", source)
        self.assertIn("blocker_cleared: false", source)

    def test_runner_hook_observes_final_pre_normalization_intents_before_submission(self):
        source = RUNNER.read_text()
        plan_pos = source.index("let mut mm_plan =")
        submission_pos = source.index("let mut intents = mm_submission_intents_for_quote_gate")
        cleanup_filter_pos = source.index("match phase51_lighter_baseline_cleanup_only_filter_intents")
        paper_filter_pos = source.index("crate::v2::apply_paper_admission_filter")
        hook_pos = source.index("crate::v2::emit_shadow_decision")
        normalize_pos = source.index("normalize_live_client_order_ids(&mut intents")
        lineage_pos = source.index("register_mm_decision_lineage(&mut state")
        order_tx_pos = source.index("send_order_fire_and_forget(", hook_pos)

        self.assertGreater(hook_pos, plan_pos)
        self.assertGreater(hook_pos, submission_pos)
        self.assertGreater(hook_pos, cleanup_filter_pos)
        self.assertGreater(paper_filter_pos, cleanup_filter_pos)
        self.assertLess(paper_filter_pos, normalize_pos)
        self.assertLess(paper_filter_pos, lineage_pos)
        self.assertLess(paper_filter_pos, order_tx_pos)
        self.assertLess(paper_filter_pos, hook_pos)
        self.assertLess(hook_pos, normalize_pos)
        self.assertLess(hook_pos, lineage_pos)
        self.assertLess(hook_pos, order_tx_pos)


if __name__ == "__main__":
    unittest.main()
