import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import phase51ap_lighter_pressure_sidecar_schema as schema  # noqa: E402


def complete_packet(**overrides):
    packet = {
        "schema_version": 1,
        "producer": "Phase51LighterPressureSource",
        "target_type": "lighter_native_limit",
        "venue_id": "lighter",
        "baseline_commit": "synthetic-baseline",
        "run_id": "synthetic-run",
        "gate_status": "HOLD",
        "canonical_group_id": "synthetic-canonical-group",
        "order_key": "synthetic-order-key",
        "public_market_key": "lighter:synthetic-market",
        "source_event_time_ms": 1_700_000_000_000,
        "observed_at_ms": 1_700_000_000_250,
        "native_limit_event_time_status": "SYNTHETIC_EVENT_TIME_ALIGNED",
        "active_order_headroom_account": 90,
        "active_order_headroom_market": 12,
        "sendtx_per_minute_limit": 60,
        "sendtx_per_minute_remaining": 58,
        "rest_requests_per_minute_limit": 120,
        "rest_requests_per_minute_remaining": 117,
        "target_key_provenance_state": "SYNTHETIC_EXPLICIT_TARGET_KEY",
        "active_order_provenance_state": "SYNTHETIC_EVENT_TIME_SOURCE",
        "sendtx_provenance_state": "SYNTHETIC_EVENT_TIME_SOURCE",
        "request_pressure_provenance_state": "SYNTHETIC_EVENT_TIME_SOURCE",
        "pressure_packet_state": "SYNTHETIC_SANITIZED_EVENT_TIME_COMPLETE",
        "pressure_state": "pressure_complete",
        "raw_identifier_redaction_status": "PASS",
        "fixture_provenance": "SYNTHETIC_FIXTURE_ONLY",
        "native_limit_pressure_source": "SYNTHETIC_FIXTURE_PRESSURE_SOURCE",
        "transform_version": "synthetic-transform-v1",
        "redaction_policy_version": "synthetic-redaction-v1",
        "source_count": 5,
        "completeness_flag": True,
        "is_synthetic_fixture": True,
        "derived_from_real_evidence": False,
        "runtime_observation": False,
        "capture_enabled": False,
        "gap_or_staleness_flag": False,
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
    packet.update(overrides)
    return packet


def unavailable_packet(**overrides):
    packet = {
        "schema_version": 1,
        "producer": "Phase51LighterPressureSource",
        "target_type": "lighter_native_limit",
        "venue_id": "lighter",
        "run_id": "governance-closeout",
        "gate_status": "HOLD",
        "native_limit_event_time_status": "PRESSURE_UNAVAILABLE",
        "active_order_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "sendtx_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "request_pressure_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "pressure_packet_state": "AUDITED_SANITIZED_PRESSURE_UNAVAILABLE",
        "pressure_state": "pressure_unavailable",
        "raw_identifier_redaction_status": "PASS",
        "fixture_provenance": "SANITIZED_GOVERNANCE_CLOSEOUT",
        "native_limit_pressure_source": "LIGHTER_SOURCE_ROUTE_CLOSED_NEGATIVE",
        "transform_version": "governance-transform-v1",
        "redaction_policy_version": "governance-redaction-v1",
        "source_count": 5,
        "account_limits_probe_status": "REQUIRED_DIMENSIONS_ABSENT",
        "passive_sendtx_observation_status": "REQUIRED_DIMENSIONS_ABSENT",
        "repo_docs_sdk_audit_status": "NO_EXPLICIT_SOURCE_FOUND",
        "websocket_schema_audit_status": "NO_COMPLETE_PRESSURE_DIMENSIONS",
        "pressure_unavailable_reason": "LIGHTER_EXPLICIT_PRESSURE_SOURCE_CLOSED_NEGATIVE",
        "governance_decision_sha256": "a" * 64,
        "completeness_flag": False,
        "is_synthetic_fixture": False,
        "derived_from_real_evidence": True,
        "runtime_observation": False,
        "capture_enabled": False,
        "gap_or_staleness_flag": True,
        "missing_pressure_values_inferred": False,
        "volume_quota_substitute_rejected": True,
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
    packet.update(overrides)
    return packet


class Phase51ApLighterPressureSidecarSchemaTests(unittest.TestCase):
    def assert_rejected_with(self, packet, expected_reason):
        result = schema.validate_packet(packet)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sanitized_packet)
        self.assertTrue(
            any(expected_reason in reason for reason in result.reject_reasons),
            result.reject_reasons,
        )
        return result

    def test_complete_rest_packet_is_accepted(self):
        packet = complete_packet()

        result = schema.validate_packet(packet)

        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.sanitized_packet, packet)
        self.assertEqual(schema.classify_pressure_state(packet), "pressure_complete")
        self.assertFalse(result.sanitized_packet["blocker_cleared"])

    def test_lighter_pressure_unavailable_packet_is_accepted_as_distinct_state(self):
        packet = unavailable_packet()

        result = schema.validate_packet(packet)

        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.sanitized_packet["pressure_state"], "pressure_unavailable")
        self.assertEqual(
            result.sanitized_packet["native_limit_event_time_status"],
            "PRESSURE_UNAVAILABLE",
        )
        self.assertFalse(result.sanitized_packet["completeness_flag"])
        self.assertFalse(result.sanitized_packet["blocker_cleared"])
        self.assertNotIn("sendtx_per_minute_remaining", result.sanitized_packet)
        self.assertEqual(schema.classify_pressure_state(packet), "pressure_unavailable")

    def test_pressure_unavailable_rejects_inferred_or_synthesized_pressure_values(self):
        self.assert_rejected_with(
            unavailable_packet(sendtx_per_minute_remaining=59),
            "sendtx_per_minute_remaining must be absent/null when pressure_state is pressure_unavailable",
        )
        self.assert_rejected_with(
            unavailable_packet(missing_pressure_values_inferred=True),
            "missing_pressure_values_inferred must be False",
        )
        self.assert_rejected_with(
            unavailable_packet(volume_quota_substitute_rejected=False),
            "volume_quota_substitute_rejected must be True",
        )
        self.assert_rejected_with(
            unavailable_packet(volume_quota_remaining=10780),
            "unsupported field volume_quota_remaining",
        )

    def test_pressure_unavailable_requires_sanitized_governance_evidence(self):
        self.assert_rejected_with(
            unavailable_packet(governance_decision_sha256="not-a-sha"),
            "governance_decision_sha256 must be a lowercase sanitized sha256",
        )
        self.assert_rejected_with(
            unavailable_packet(pressure_unavailable_reason="UNKNOWN"),
            "pressure_unavailable_reason must be",
        )
        self.assert_rejected_with(
            unavailable_packet(repo_docs_sdk_audit_status=""),
            "repo_docs_sdk_audit_status must be a non-empty string",
        )

    def test_incomplete_or_unknown_state_is_rejected(self):
        packet = complete_packet(pressure_state="pressure_incomplete_or_unknown")
        del packet["sendtx_per_minute_remaining"]

        result = self.assert_rejected_with(
            packet,
            "pressure_incomplete_or_unknown is not an accepted Phase 5.1 pressure state",
        )
        self.assertIsNone(result.sanitized_packet)

    def test_missing_pressure_state_is_rejected_instead_of_silently_complete(self):
        packet = complete_packet()
        del packet["pressure_state"]

        self.assert_rejected_with(packet, "pressure_state must be one of")

    def test_complete_weighted_packet_is_accepted(self):
        packet = complete_packet(
            rest_requests_per_minute_limit=None,
            rest_requests_per_minute_remaining=None,
            weighted_requests_per_minute_limit=240,
            weighted_requests_per_minute_remaining=239,
        )
        del packet["rest_requests_per_minute_limit"]
        del packet["rest_requests_per_minute_remaining"]

        result = schema.validate_packet(packet)

        self.assertTrue(result.accepted, result.reject_reasons)

    def test_complete_packet_with_both_request_pairs_is_accepted(self):
        packet = complete_packet(
            weighted_requests_per_minute_limit=240,
            weighted_requests_per_minute_remaining=238,
        )

        result = schema.validate_packet(packet)

        self.assertTrue(result.accepted, result.reject_reasons)

    def test_missing_target_key_fields_are_rejected(self):
        packet = complete_packet(canonical_group_id="")

        self.assert_rejected_with(packet, "canonical_group_id must be a non-empty string")

        packet = complete_packet(order_key="")
        self.assert_rejected_with(packet, "order_key must be a non-empty string")

    def test_source_link_only_packet_is_rejected(self):
        packet = complete_packet(sanitized_source_record_sha256="a" * 64)
        for field in (
            "active_order_headroom_account",
            "active_order_headroom_market",
            "sendtx_per_minute_limit",
            "sendtx_per_minute_remaining",
            "rest_requests_per_minute_limit",
            "rest_requests_per_minute_remaining",
        ):
            del packet[field]

        self.assert_rejected_with(packet, "active_order_headroom_account must be a non-negative integer")
        self.assert_rejected_with(packet, "REST-or-weighted request pressure pair is required")

    def test_missing_or_incomplete_pressure_pair_is_rejected(self):
        packet = complete_packet()
        del packet["rest_requests_per_minute_limit"]
        del packet["rest_requests_per_minute_remaining"]

        self.assert_rejected_with(packet, "REST-or-weighted request pressure pair is required")

        packet = complete_packet()
        del packet["rest_requests_per_minute_remaining"]
        self.assert_rejected_with(
            packet,
            "rest_requests_per_minute_limit/rest_requests_per_minute_remaining must be present as a complete pair",
        )

        packet = complete_packet(
            weighted_requests_per_minute_limit=240,
            weighted_requests_per_minute_remaining=239,
        )
        del packet["weighted_requests_per_minute_remaining"]
        self.assert_rejected_with(
            packet,
            "weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining must be present as a complete pair",
        )

    def test_rejects_docs_config_snapshot_local_or_inferred_provenance(self):
        rejected = (
            ("target_key_provenance_state", "DERIVED_FROM_CLIENT_ORDER_ID"),
            ("active_order_provenance_state", "CURRENT_SNAPSHOT"),
            ("sendtx_provenance_state", "DERIVED_FROM_DOCS"),
            ("request_pressure_provenance_state", "LOCAL_COUNTER"),
            ("request_pressure_provenance_state", "CONFIGURED_CAP"),
            ("request_pressure_provenance_state", "INFERRED"),
            ("request_pressure_provenance_state", "OBSERVED_EVENT_TIME_SOURCE"),
            ("request_pressure_provenance_state", "READONLY_CAPTURE"),
            ("request_pressure_provenance_state", "RUNTIME_CONFIG"),
        )
        for field, value in rejected:
            with self.subTest(field=field, value=value):
                packet = complete_packet(**{field: value})
                self.assert_rejected_with(packet, f"{field} must be")

    def test_rejects_missing_synthetic_provenance_markers(self):
        self.assert_rejected_with(
            complete_packet(fixture_provenance="REAL_EVIDENCE"),
            "fixture_provenance must be",
        )
        self.assert_rejected_with(
            complete_packet(native_limit_pressure_source="SANITIZED_EXTERNAL_PRESSURE_SOURCE"),
            "native_limit_pressure_source must be",
        )
        self.assert_rejected_with(
            complete_packet(is_synthetic_fixture=False),
            "is_synthetic_fixture must be True",
        )
        self.assert_rejected_with(
            complete_packet(derived_from_real_evidence=True),
            "derived_from_real_evidence must be False",
        )
        self.assert_rejected_with(
            complete_packet(runtime_observation=True),
            "runtime_observation must be False",
        )
        self.assert_rejected_with(
            complete_packet(capture_enabled=True),
            "capture_enabled must be False",
        )

    def test_rejects_raw_identifier_and_secret_fields_without_echoing_values(self):
        packet = complete_packet(order_id="raw-sensitive-order")

        result = self.assert_rejected_with(packet, "raw identifier field order_id is prohibited")
        self.assertFalse(any("raw-sensitive-order" in reason for reason in result.reject_reasons))

        packet = complete_packet(api_key="secret-value")
        result = self.assert_rejected_with(packet, "secret-shaped field api_key is prohibited")
        self.assertFalse(any("secret-value" in reason for reason in result.reject_reasons))

        packet = complete_packet(sanitized_source_record_sha256="a" * 64)
        result = schema.validate_packet(packet)
        self.assertTrue(result.accepted, result.reject_reasons)

    def test_rejects_nested_raw_identifier_and_secret_fields(self):
        packet = complete_packet(metadata={"order_id": "raw-sensitive-order"})

        result = self.assert_rejected_with(packet, "raw identifier field order_id is prohibited")
        self.assertFalse(any("raw-sensitive-order" in reason for reason in result.reject_reasons))

        packet = complete_packet(metadata={"api_key": "secret-value"})
        result = self.assert_rejected_with(packet, "secret-shaped field api_key is prohibited")
        self.assertFalse(any("secret-value" in reason for reason in result.reject_reasons))

    def test_rejects_unsafe_true_flags_and_missing_non_live_flag(self):
        self.assert_rejected_with(
            complete_packet(approved_for_live=True),
            "approved_for_live must be False",
        )
        self.assert_rejected_with(
            complete_packet(no_live_flag=False),
            "no_live_flag must be True",
        )

    def test_rejects_numeric_boolean_impostors(self):
        self.assert_rejected_with(
            complete_packet(no_live_flag=1),
            "no_live_flag must be a JSON boolean",
        )
        self.assert_rejected_with(
            complete_packet(capture_enabled=0),
            "capture_enabled must be a JSON boolean",
        )
        self.assert_rejected_with(
            complete_packet(approved_for_live=0),
            "approved_for_live must be a JSON boolean",
        )

    def test_rejects_non_event_time_status_and_staleness(self):
        self.assert_rejected_with(
            complete_packet(native_limit_event_time_status="CURRENT_SNAPSHOT"),
            "native_limit_event_time_status must be",
        )
        self.assert_rejected_with(
            complete_packet(observed_at_ms=1_700_000_120_001),
            "event-time observation lag exceeds",
        )
        self.assert_rejected_with(
            complete_packet(observed_at_ms=1_699_999_999_999),
            "observed_at_ms must not precede",
        )
        self.assert_rejected_with(
            complete_packet(gap_or_staleness_flag=True),
            "gap_or_staleness_flag must be False",
        )

    def test_rejects_wrong_venue_or_target_type(self):
        self.assert_rejected_with(
            complete_packet(venue_id="extended"),
            "venue_id must be",
        )
        self.assert_rejected_with(
            complete_packet(target_type="native_role"),
            "target_type must be",
        )

    def test_rejects_non_integer_negative_and_remaining_above_limit(self):
        self.assert_rejected_with(
            complete_packet(sendtx_per_minute_limit="60"),
            "sendtx_per_minute_limit must be a non-negative integer",
        )
        self.assert_rejected_with(
            complete_packet(active_order_headroom_account=-1),
            "active_order_headroom_account must be a non-negative integer",
        )
        self.assert_rejected_with(
            complete_packet(sendtx_per_minute_remaining=61),
            "sendtx_per_minute_remaining must be <= sendtx_per_minute_limit",
        )
        self.assert_rejected_with(
            complete_packet(rest_requests_per_minute_remaining=121),
            "rest_requests_per_minute_remaining must be <= rest_requests_per_minute_limit",
        )

    def test_rejects_nested_unknown_and_raw_identifier_shaped_values(self):
        self.assert_rejected_with(
            complete_packet(raw_payload={"safe": "synthetic"}),
            "nested payload field raw_payload is prohibited",
        )
        self.assert_rejected_with(
            complete_packet(account_snapshot_source="snapshot"),
            "unsupported field account_snapshot_source",
        )
        self.assert_rejected_with(
            complete_packet(public_market_key="0x" + "a" * 40),
            "secret-shaped or raw-identifier-shaped value",
        )

    def test_packet_digest_is_deterministic_for_synthetic_packet(self):
        left = schema.packet_digest(complete_packet())
        right = schema.packet_digest(complete_packet())

        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_library_has_no_runtime_or_output_surface(self):
        source = (REPO_ROOT / "tools" / "phase51ap_lighter_pressure_sidecar_schema.py").read_text(
            encoding="utf-8"
        )

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
