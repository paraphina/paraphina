import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import phase51ap_lighter_pressure_sidecar_schema as packet_schema  # noqa: E402
import phase51aq_lighter_pressure_sidecar_lifecycle as lifecycle  # noqa: E402


def common_fact(fact_type, sample_id="synthetic-sample-1", **overrides):
    fact = {
        "schema_version": 1,
        "fact_type": fact_type,
        "sample_id": sample_id,
        "venue_id": "lighter",
        "gate_status": "HOLD",
        "fixture_provenance": "SYNTHETIC_FIXTURE_ONLY",
        "raw_identifier_redaction_status": "PASS",
        "is_synthetic_fixture": True,
        "derived_from_real_evidence": False,
        "runtime_observation": False,
        "capture_enabled": False,
        "no_live_flag": True,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_financial_claim": False,
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "blocker_cleared": False,
    }
    fact.update(overrides)
    return fact


def target_key_fact(sample_id="synthetic-sample-1", **overrides):
    fact = common_fact("target_key_fact", sample_id)
    fact.update(
        {
            "baseline_commit": "synthetic-baseline",
            "run_id": "synthetic-run",
            "canonical_group_id": "synthetic-canonical-group",
            "order_key": "synthetic-order-key",
            "public_market_key": "lighter:synthetic-market",
            "transform_version": "synthetic-transform-v1",
            "redaction_policy_version": "synthetic-redaction-v1",
        }
    )
    fact.update(overrides)
    return fact


def active_order_fact(sample_id="synthetic-sample-1", **overrides):
    fact = common_fact("active_order_pressure_fact", sample_id)
    fact.update(
        {
            "active_order_headroom_account": 90,
            "active_order_headroom_market": 12,
        }
    )
    fact.update(overrides)
    return fact


def pressure_submission_fact(sample_id="synthetic-sample-1", **overrides):
    fact = common_fact("sendtx_pressure_fact", sample_id)
    fact.update(
        {
            "sendtx_per_minute_limit": 60,
            "sendtx_per_minute_remaining": 58,
        }
    )
    fact.update(overrides)
    return fact


def request_pressure_fact(sample_id="synthetic-sample-1", **overrides):
    fact = common_fact("request_pressure_fact", sample_id)
    fact.update(
        {
            "rest_requests_per_minute_limit": 120,
            "rest_requests_per_minute_remaining": 117,
        }
    )
    fact.update(overrides)
    return fact


def event_time_fact(sample_id="synthetic-sample-1", **overrides):
    fact = common_fact("event_time_fact", sample_id)
    fact.update(
        {
            "source_event_time_ms": 1_700_000_000_000,
            "observed_at_ms": 1_700_000_000_250,
        }
    )
    fact.update(overrides)
    return fact


def complete_facts(sample_id="synthetic-sample-1"):
    return [
        target_key_fact(sample_id),
        active_order_fact(sample_id),
        pressure_submission_fact(sample_id),
        request_pressure_fact(sample_id),
        event_time_fact(sample_id),
    ]


class Phase51AqLighterPressureSidecarLifecycleTests(unittest.TestCase):
    def assert_rejected_with(self, fact, expected_reason):
        reasons = lifecycle.validate_fact(fact)
        self.assertTrue(any(expected_reason in reason for reason in reasons), reasons)
        return reasons

    def test_complete_synthetic_fact_set_emits_one_valid_packet(self):
        machine = lifecycle.SyntheticLighterPressureSidecarLifecycle()
        results = [machine.ingest_fact(fact) for fact in complete_facts()]

        self.assertTrue(all(result.accepted for result in results), results)
        self.assertEqual([result.packet is not None for result in results], [False, False, False, False, True])
        packet = results[-1].packet
        self.assertIsNotNone(packet)
        packet_result = packet_schema.validate_packet(packet)
        self.assertTrue(packet_result.accepted, packet_result.reject_reasons)
        self.assertEqual(packet["source_count"], 5)
        self.assertEqual(results[-1].state, "EMITTABLE_SYNTHETIC_PACKET")

    def test_partial_facts_remain_internal_and_emit_no_packet(self):
        machine = lifecycle.SyntheticLighterPressureSidecarLifecycle()

        for fact in complete_facts()[:4]:
            result = machine.ingest_fact(fact)
            self.assertTrue(result.accepted, result.reject_reasons)
            self.assertEqual(result.state, "OBSERVED_PARTIAL_INTERNAL_ONLY")
            self.assertIsNone(result.packet)

        self.assertEqual(machine.pending_sample_count(), 1)

    def test_mismatched_sample_ids_do_not_join(self):
        machine = lifecycle.SyntheticLighterPressureSidecarLifecycle()
        facts = [
            target_key_fact("synthetic-left"),
            active_order_fact("synthetic-right"),
            pressure_submission_fact("synthetic-left"),
            request_pressure_fact("synthetic-left"),
            event_time_fact("synthetic-left"),
        ]

        results = [machine.ingest_fact(fact) for fact in facts]

        self.assertTrue(all(result.accepted for result in results), results)
        self.assertTrue(all(result.packet is None for result in results))
        self.assertEqual(machine.pending_sample_count(), 2)

    def test_duplicate_identical_fact_is_ignored_and_conflicting_duplicate_rejected(self):
        machine = lifecycle.SyntheticLighterPressureSidecarLifecycle()
        first = target_key_fact()

        self.assertEqual(machine.ingest_fact(first).state, "OBSERVED_PARTIAL_INTERNAL_ONLY")
        self.assertEqual(machine.ingest_fact(dict(first)).state, "DUPLICATE_FACT_IGNORED")
        rejected = machine.ingest_fact(target_key_fact(order_key="synthetic-different-order-key"))

        self.assertFalse(rejected.accepted)
        self.assertIn("conflicting duplicate fact", rejected.reject_reasons)

    def test_rejects_raw_identifier_secret_and_nested_payload_facts_without_echoing_values(self):
        reasons = self.assert_rejected_with(
            target_key_fact(order_id="raw-sensitive-order"),
            "raw identifier field order_id is prohibited",
        )
        self.assertFalse(any("raw-sensitive-order" in reason for reason in reasons))

        reasons = self.assert_rejected_with(
            target_key_fact(api_key="secret-value"),
            "secret-shaped field api_key is prohibited",
        )
        self.assertFalse(any("secret-value" in reason for reason in reasons))

        reasons = self.assert_rejected_with(
            target_key_fact(metadata={"order_id": "raw-sensitive-order"}),
            "raw identifier field order_id is prohibited",
        )
        self.assertFalse(any("raw-sensitive-order" in reason for reason in reasons))
        self.assert_rejected_with(target_key_fact(metadata={"safe": "value"}), "nested payload field metadata")

    def test_rejects_non_synthetic_or_unsafe_fact_markers(self):
        self.assert_rejected_with(
            target_key_fact(fixture_provenance="REAL_EVIDENCE"),
            "fixture_provenance must be",
        )
        self.assert_rejected_with(
            target_key_fact(runtime_observation=True),
            "runtime_observation must be False",
        )
        self.assert_rejected_with(
            target_key_fact(capture_enabled=True),
            "capture_enabled must be False",
        )
        self.assert_rejected_with(
            target_key_fact(approved_for_live=True),
            "approved_for_live must be False",
        )
        self.assert_rejected_with(
            target_key_fact(no_live_flag=1),
            "no_live_flag must be a JSON boolean",
        )

    def test_rejects_incomplete_pressure_facts(self):
        bad_request = request_pressure_fact()
        del bad_request["rest_requests_per_minute_remaining"]
        self.assert_rejected_with(
            bad_request,
            "rest_requests_per_minute_limit/rest_requests_per_minute_remaining must be present as a complete pair",
        )
        self.assert_rejected_with(
            pressure_submission_fact(sendtx_per_minute_remaining=61),
            "sendtx_per_minute_remaining must be <= sendtx_per_minute_limit",
        )
        self.assert_rejected_with(
            active_order_fact(active_order_headroom_account="90"),
            "active_order_headroom_account must be a non-negative integer",
        )

    def test_rejects_non_synthetic_sample_ids_and_raw_shaped_values(self):
        self.assert_rejected_with(
            target_key_fact(sample_id="sample-1"),
            "sample_id must be synthetic-only",
        )
        self.assert_rejected_with(
            target_key_fact(public_market_key="0x" + "a" * 40),
            "public_market_key contains secret-shaped or raw-identifier-shaped value",
        )

    def test_capacity_is_bounded(self):
        machine = lifecycle.SyntheticLighterPressureSidecarLifecycle(max_samples=1)
        self.assertTrue(machine.ingest_fact(target_key_fact("synthetic-a")).accepted)
        rejected = machine.ingest_fact(target_key_fact("synthetic-b"))

        self.assertFalse(rejected.accepted)
        self.assertIn("sample capacity exceeded", rejected.reject_reasons)

    def test_lifecycle_module_has_no_runtime_or_output_surface(self):
        source = (
            REPO_ROOT / "tools" / "phase51aq_lighter_pressure_sidecar_lifecycle.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("argparse", source)
        self.assertNotIn("DEFAULT_OUTPUT_ROOT", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("sendTx", source)
        self.assertNotIn("sendTxBatch", source)
        self.assertNotIn("nextNonce", source)
        self.assertNotIn("paraphina.live", source)
        self.assertNotIn("lighter_signer", source)


if __name__ == "__main__":
    unittest.main()
