from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("/home/ubuntu/paraphina_mm_pnl_harness/tools/ws_soak_report.py")
SPEC = importlib.util.spec_from_file_location("ws_soak_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ws_soak_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ws_soak_report
SPEC.loader.exec_module(ws_soak_report)


class WsSoakReportTests(unittest.TestCase):
    def test_classify_no_data_transport_gap(self) -> None:
        defect = ws_soak_report.classify_extended_stale_episode(
            {
                "stale_ms": "1800",
                "age_ws_rx_ms": "100",
                "age_data_rx_ms": "1900",
                "age_book_event_ms": "1900",
                "age_published_ms": "1900",
            },
            None,
        )
        self.assertEqual(defect, "no_data_transport_gap")

    def test_classify_data_seen_no_publish(self) -> None:
        defect = ws_soak_report.classify_extended_stale_episode(
            {
                "stale_ms": "1800",
                "age_ws_rx_ms": "100",
                "age_data_rx_ms": "200",
                "age_book_event_ms": "2200",
                "age_published_ms": "2200",
            },
            None,
        )
        self.assertEqual(defect, "data_seen_no_publish")

    def test_classify_runner_freeze_apply_gap(self) -> None:
        defect = ws_soak_report.classify_extended_stale_episode(
            {
                "stale_ms": "1800",
                "age_ws_rx_ms": "100",
                "age_data_rx_ms": "200",
                "age_book_event_ms": "200",
                "age_published_ms": "200",
            },
            {
                "venue_state_stale_ms": "1500",
                "age_apply_ms": "1700",
                "age_event_ms": "200",
                "ext_apply_frozen_total": "3",
                "ext_future_total": "0",
            },
        )
        self.assertEqual(defect, "runner_freeze_apply_gap")

    def test_classify_runner_freeze_apply_gap_from_warning_fallback(self) -> None:
        defect = ws_soak_report.classify_extended_stale_episode(
            {
                "stale_ms": "1800",
                "age_ws_rx_ms": "100",
                "age_data_rx_ms": "200",
                "age_book_event_ms": "200",
                "age_published_ms": "200",
            },
            {
                "venue_state_stale_ms": "1500",
                "age_apply_ms": "1700",
                "age_event_ms": "200",
                "ext_apply_frozen_total": "0",
                "_freeze_warning_count": "2",
            },
        )
        self.assertEqual(defect, "runner_freeze_apply_gap")

    def test_classify_future_timestamp_deferral(self) -> None:
        defect = ws_soak_report.classify_extended_stale_episode(
            {
                "stale_ms": "1800",
                "age_ws_rx_ms": "100",
                "age_data_rx_ms": "200",
                "age_book_event_ms": "200",
                "age_published_ms": "1200",
            },
            {
                "venue_state_stale_ms": "1500",
                "age_apply_ms": "1700",
                "age_event_ms": "200",
                "ext_apply_frozen_total": "0",
                "ext_future_total": "2",
            },
        )
        self.assertEqual(defect, "future_timestamp_deferral")

    def test_summarize_extended_defects(self) -> None:
        summary = ws_soak_report.summarize_extended_defects(
            [
                {"defect_class": "runner_freeze_apply_gap"},
                {"defect_class": "runner_freeze_apply_gap"},
                {"defect_class": "data_seen_no_publish"},
            ]
        )
        self.assertEqual(summary["dominant_class"], "runner_freeze_apply_gap")
        self.assertEqual(summary["dominant_count"], 2)
        self.assertAlmostEqual(summary["confidence_pct"], 66.6666, places=3)

    def test_parse_run_log_collects_extended_reconnect_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            run_log.write_text(
                "\n".join(
                    [
                        "WS_AUDIT venue=extended reconnect_reason=stale_watchdog count=2",
                        "WS_AUDIT venue=extended component=reconnect_policy reason=stale_watchdog sleep_ms=100 failure_escalation_suppressed=1 consecutive_failures=0",
                        "WS_AUDIT venue=extended reconnect_reason=bootstrap_no_first_frame count=1",
                        "WS_AUDIT venue=extended component=reconnect_policy reason=bootstrap_no_first_frame reason_family=bootstrap_no_data sleep_ms=100 failure_escalation_suppressed=1 consecutive_failures=0 bootstrap_count_window=1 bootstrap_window_ms=120000 bootstrap_limit=2 bootstrap_churn_escalated=0",
                        "WS_AUDIT venue=extended extended_read_timeout_ms=45000 extended_connect_first_frame_timeout_ms=1100 extended_control_frame_only_timeout_ms=1400 extended_connect_book_timeout_ms=750 extended_state_stale_ms=1500",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = ws_soak_report.parse_run_log(run_log)
            audit_reconnect = parsed[0]
            extended_cfg_stats = parsed[9]
            extended_reconnect_policy_stats = parsed[10]

            self.assertEqual(audit_reconnect[("extended", "stale_watchdog")], 2)
            self.assertEqual(audit_reconnect[("extended", "bootstrap_no_first_frame")], 1)
            self.assertEqual(
                extended_reconnect_policy_stats["stale_watchdog"]["last_sleep_ms"], 100
            )
            self.assertEqual(
                extended_reconnect_policy_stats["bootstrap_no_first_frame"][
                    "last_failure_escalation_suppressed"
                ],
                1,
            )
            self.assertEqual(
                extended_cfg_stats["extended"]["last_extended_connect_book_timeout_ms"],
                750,
            )
            self.assertEqual(
                extended_cfg_stats["extended"][
                    "last_extended_connect_first_frame_timeout_ms"
                ],
                1100,
            )

    def test_parse_run_log_collects_extended_transport_gap_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            run_log.write_text(
                "\n".join(
                    [
                        "WS_AUDIT venue=extended component=stale_watchdog_churn action=record stale_watchdog_count_window=3 stale_watchdog_window_ms=120000 stale_watchdog_limit=2 stale_watchdog_fast_reconnect_allowed=0 stale_watchdog_churn_escalated=1 session_duration_ms=2400",
                        "WS_AUDIT venue=extended component=session_progress stage=first_publish time_to_first_control_frame_ms=7 time_to_first_message_ms=31 time_to_first_book_ms=87 time_to_first_publish_ms=104",
                        "WS_AUDIT venue=extended component=stale_watchdog_churn action=reset stale_watchdog_count_window=0 stale_watchdog_window_ms=120000 stale_watchdog_limit=2 stale_watchdog_fast_reconnect_allowed=1 stale_watchdog_churn_escalated=0 healthy_session_ms_before_reset=30500 previous_stale_watchdog_count_window=3",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = ws_soak_report.parse_run_log(run_log)
            extended_transport_gap_stats = parsed[11]

            self.assertEqual(
                extended_transport_gap_stats["extended"][
                    "max_stale_watchdog_count_window"
                ],
                3,
            )
            self.assertEqual(
                extended_transport_gap_stats["extended"][
                    "last_stale_watchdog_fast_reconnect_allowed"
                ],
                1,
            )
            self.assertEqual(
                extended_transport_gap_stats["extended"]["max_time_to_first_publish_ms"],
                104,
            )
            self.assertEqual(
                extended_transport_gap_stats["extended"][
                    "max_healthy_session_ms_before_reset"
                ],
                30500,
            )
            self.assertEqual(
                extended_transport_gap_stats["extended"][
                    "max_time_to_first_control_frame_ms"
                ],
                7,
            )

    def test_parse_run_log_collects_extended_bootstrap_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            run_log.write_text(
                "\n".join(
                    [
                        "WS_AUDIT venue=extended component=rest_snapshot_seed status=ok latency_ms=41 endpoint_kind=official_orderbook seeded=1 bid_levels=1 ask_levels=1 market=ETH-USD http_status=200",
                        "WS_AUDIT venue=extended component=socket_establishment action=tcp_connected socket_role=primary stream_kind=depth1 host=api.starknet.extended.exchange path=/stream.extended.exchange/v1/orderbooks/ETH-USD?depth=1 disable_nagle=1 elapsed_ms=17 tcp_connect_ms=17",
                        "WS_AUDIT venue=extended component=socket_establishment action=ws_upgraded socket_role=primary stream_kind=depth1 host=api.starknet.extended.exchange path=/stream.extended.exchange/v1/orderbooks/ETH-USD?depth=1 disable_nagle=1 elapsed_ms=61 tcp_connect_ms=17 ws_upgrade_ms=44",
                        "WS_AUDIT venue=extended component=socket_establishment action=failed socket_role=hedge stream_kind=full_orderbook host=api.starknet.extended.exchange path=/stream.extended.exchange/v1/orderbooks/ETH-USD disable_nagle=1 elapsed_ms=1450 tcp_connect_ms=12 failure_stage=ws_upgrade failure_class=timeout",
                        "WS_AUDIT venue=extended component=bootstrap_seed_bridge action=activated rest_snapshot_seeded=1 rest_seed_bridge_active=1 seed_age_ms=0 venue_state_stale_ms=1500 connect_first_frame_timeout_ms=1100",
                        "WS_AUDIT venue=extended component=bootstrap_control_frame_grace action=armed reason_family=bootstrap_no_data control_frame_only_timeout_ms=1400 connect_first_frame_timeout_ms=1100 seed_age_ms=1002 venue_state_stale_ms=1500 first_control_frame_kind=ping first_control_frame_seen=1 first_data_frame_seen=0 rest_seed_bridge_active=1",
                        "WS_AUDIT venue=extended component=bootstrap_session_hedge action=started connect_first_frame_timeout_ms=1100 control_frame_only_timeout_ms=1450 seed_age_ms=1002 venue_state_stale_ms=1500 first_control_frame_kind=ping rest_seed_bridge_active=1 hedge_started_at_ms=1000",
                        "WS_AUDIT venue=extended component=backend_attach_fallback action=started primary_stream_kind=depth1 fallback_stream_kind=full_orderbook connect_first_frame_timeout_ms=1100 control_frame_only_timeout_ms=1450 seed_age_ms=1002 rest_seed_bridge_active=1 hedge_started_at_ms=1000",
                        "WS_AUDIT venue=extended component=bootstrap_timeout reason=bootstrap_control_frame_only_session_establishment reason_family=bootstrap_no_data bootstrap_timeout_stage=session_establishment_hedge connect_first_frame_timeout_ms=1100 connect_book_timeout_ms=750 rest_snapshot_seeded=1 rest_seed_bridge_active=1 first_control_frame_seen=1 first_control_frame_kind=ping first_data_frame_seen=0 first_message_seen=0 first_book_seen=0 first_publish_seen=0 stale_watchdog_armed=0 stale_watchdog_deferred_until_first_publish=1 last_frame_kind=ping last_data_kind=none last_seq=0 last_snapshot_seq=0 last_book_seq=0 last_publish_seq=0 rest_snapshot_latency_ms=41 rest_snapshot_bid_levels=1 rest_snapshot_ask_levels=1 control_frame_only_timeout_ms=1450 seed_age_ms=1451 time_to_first_control_frame_ms=500",
                        "WS_AUDIT venue=extended component=session_progress stage=first_message socket_role=primary stream_kind=depth1 ws_upgrade_completed=1 time_to_first_control_frame_ms=7 time_to_first_message_ms=35",
                        "WS_AUDIT venue=extended component=session_progress stage=first_publish socket_role=primary stream_kind=depth1 ws_upgrade_completed=1 time_to_first_control_frame_ms=7 time_to_first_message_ms=35 time_to_first_book_ms=40 time_to_first_publish_ms=42",
                        "WS_AUDIT venue=extended component=bootstrap_timeout reason=bootstrap_book_no_publish reason_family=bootstrap_no_data bootstrap_timeout_stage=post_first_frame connect_first_frame_timeout_ms=1100 connect_book_timeout_ms=750 rest_snapshot_seeded=0 rest_seed_bridge_active=0 first_control_frame_seen=0 first_control_frame_kind=none first_data_frame_seen=1 first_message_seen=1 first_book_seen=1 first_publish_seen=0 stale_watchdog_armed=0 stale_watchdog_deferred_until_first_publish=1 time_to_first_message_ms=35 time_to_first_book_ms=40 last_frame_kind=text last_data_kind=snapshot last_seq=44 last_snapshot_seq=44 last_book_seq=44 last_publish_seq=0 rest_snapshot_latency_ms=52 rest_snapshot_bid_levels=0 rest_snapshot_ask_levels=0",
                        "WS_AUDIT venue=extended component=bootstrap_churn action=record bootstrap_reason=bootstrap_control_frame_only_session_establishment bootstrap_reason_family=bootstrap_no_data bootstrap_count_window=3 bootstrap_window_ms=120000 bootstrap_limit=2 bootstrap_fast_reconnect_allowed=0 bootstrap_churn_escalated=1 session_duration_ms=2400",
                        "WS_AUDIT venue=extended component=watchdog_bootstrap_transition first_publish_observed=1 watchdog_armed_now=1 stale_watchdog_deferred_until_first_publish=1 time_to_first_publish_ms=42",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = ws_soak_report.parse_run_log(run_log)
            extended_rest_seed_summary = parsed[12]
            extended_seed_bridge_summary = parsed[13]
            extended_control_frame_grace_summary = parsed[14]
            extended_session_hedge_stats = parsed[15]
            extended_backend_attach_fallback_summary = parsed[16]
            extended_bootstrap_timeout_stats = parsed[17]
            extended_bootstrap_churn_stats = parsed[18]
            extended_bootstrap_summary = parsed[19]
            extended_control_frame_summary = parsed[24]
            extended_socket_establishment_summary = parsed[25]
            extended_socket_role_progress_summary = parsed[26]
            extended_stream_kind_progress_summary = parsed[27]

            self.assertEqual(
                extended_rest_seed_summary["extended"]["status_ok_count"],
                1,
            )
            self.assertEqual(extended_seed_bridge_summary["activated_count"], 1)
            self.assertEqual(extended_control_frame_grace_summary["armed_count"], 1)
            self.assertEqual(
                extended_session_hedge_stats["extended"]["action_started_count"], 1
            )
            self.assertEqual(
                extended_backend_attach_fallback_summary["started_count"], 1
            )
            self.assertEqual(
                extended_control_frame_grace_summary["last_control_frame_only_timeout_ms"],
                1400,
            )
            self.assertEqual(
                extended_bootstrap_timeout_stats[
                    "bootstrap_control_frame_only_session_establishment"
                ]["samples"],
                1,
            )
            self.assertEqual(
                extended_bootstrap_timeout_stats["bootstrap_book_no_publish"][
                    "max_time_to_first_book_ms"
                ],
                40,
            )
            self.assertEqual(
                extended_bootstrap_timeout_stats[
                    "bootstrap_control_frame_only_session_establishment"
                ][
                    "last_bootstrap_timeout_stage"
                ],
                "session_establishment_hedge",
            )
            self.assertEqual(
                extended_bootstrap_timeout_stats[
                    "bootstrap_control_frame_only_session_establishment"
                ][
                    "last_connect_first_frame_timeout_ms"
                ],
                1100,
            )
            self.assertEqual(
                extended_bootstrap_timeout_stats[
                    "bootstrap_control_frame_only_session_establishment"
                ][
                    "last_first_control_frame_kind"
                ],
                "ping",
            )
            self.assertEqual(
                extended_bootstrap_churn_stats["extended"]["max_bootstrap_count_window"],
                3,
            )
            self.assertEqual(
                extended_bootstrap_summary["counts"][
                    "bootstrap_control_frame_only_session_establishment"
                ],
                1,
            )
            self.assertEqual(
                extended_socket_establishment_summary["tcp_connected_count"], 1
            )
            self.assertEqual(
                extended_socket_establishment_summary["ws_upgraded_count"], 1
            )
            self.assertEqual(
                extended_socket_establishment_summary["failed_count"], 1
            )
            self.assertEqual(
                extended_socket_establishment_summary["last_stream_kind"],
                "full_orderbook",
            )
            self.assertEqual(
                extended_socket_establishment_summary["last_failure_stage"],
                "ws_upgrade",
            )
            self.assertEqual(
                extended_socket_role_progress_summary["primary"]["stage_first_message_count"],
                1,
            )
            self.assertEqual(
                extended_stream_kind_progress_summary["depth1"]["stage_first_message_count"],
                1,
            )
            self.assertEqual(
                parsed[11]["extended"]["watchdog_bootstrap_transition_samples"],
                1,
            )
            self.assertEqual(
                extended_control_frame_summary["dominant_shape"],
                "control_frame_only",
            )

    def test_build_metrics_payload_includes_rest_seed_summary(self) -> None:
        payload = ws_soak_report.build_metrics_payload(
            telemetry_summary=ws_soak_report.TelemetrySummary(
                rows=1,
                first_tick=1,
                last_tick=1,
                first_ts_ms=1,
                last_ts_ms=1,
            ),
            extended_cfg_stats={},
            extended_reconnect_policy_stats={},
            extended_transport_gap_stats={},
            extended_rest_seed_summary={
                "extended": {"status_ok_count": 2},
                "failures_by_status": {"http_error": 1},
                "failures_by_http_status": {"404": 1},
            },
            extended_seed_bridge_summary={"activated_count": 2},
            extended_control_frame_grace_summary={"armed_count": 1},
            extended_session_hedge_summary={"started_count": 1, "hedge_won_count": 1},
            extended_backend_attach_fallback_summary={"started_count": 1, "fallback_won_count": 1},
            extended_post_publish_fallback_summary={"started_count": 2, "fallback_won_count": 1},
            extended_post_publish_gap_stage_summary={"dominant_stage": "fallback_won"},
            extended_stream_preference_summary={"last_stream_preference": "full_orderbook_degraded"},
            extended_bootstrap_timeout_stats={},
            extended_bootstrap_churn_stats={},
            extended_bootstrap_summary={},
            extended_control_frame_before_data_summary={"dominant_shape": "no_frame"},
            extended_socket_establishment_summary={"ws_upgraded_count": 2},
            extended_socket_role_progress_summary={"primary": {"samples": 1}},
            extended_stream_kind_progress_summary={"depth1": {"samples": 1}},
            extended_defect_summary={},
        )
        self.assertEqual(payload["extended_rest_seed_summary"]["extended"]["status_ok_count"], 2)
        self.assertEqual(
            payload["extended_rest_seed_failures"]["by_status"]["http_error"], 1
        )
        self.assertEqual(payload["extended_seed_bridge_summary"]["activated_count"], 2)
        self.assertEqual(payload["extended_control_frame_grace_summary"]["armed_count"], 1)
        self.assertEqual(payload["extended_session_hedge_summary"]["hedge_won_count"], 1)
        self.assertEqual(
            payload["extended_backend_attach_fallback_summary"]["fallback_won_count"], 1
        )
        self.assertEqual(
            payload["extended_post_publish_fallback_summary"]["fallback_won_count"], 1
        )
        self.assertEqual(
            payload["extended_post_publish_gap_stage_summary"]["dominant_stage"],
            "fallback_won",
        )
        self.assertEqual(
            payload["extended_stream_preference_summary"]["last_stream_preference"],
            "full_orderbook_degraded",
        )
        self.assertIn("extended_bootstrap_reason_summary", payload)
        self.assertIn("extended_bootstrap_stage_summary", payload)
        self.assertIn("extended_first_frame_timeout_summary", payload)
        self.assertIn("extended_first_data_timeout_summary", payload)
        self.assertIn("extended_control_frame_before_data_summary", payload)
        self.assertIn("extended_pre_first_data_shape_summary", payload)
        self.assertIn("extended_socket_establishment_summary", payload)
        self.assertIn("extended_socket_role_progress_summary", payload)
        self.assertIn("extended_stream_kind_progress_summary", payload)
        self.assertIn("extended_watchdog_bootstrap_transition_summary", payload)

    def test_parse_run_log_collects_extended_post_publish_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            run_log.write_text(
                "\n".join(
                    [
                        "WS_AUDIT venue=extended component=post_publish_stream_fallback action=armed active_stream_kind=depth1 fallback_stream_kind=full_orderbook post_publish_fallback_after_ms=1200 post_publish_fallback_deadline_ms=1450 age_ws_rx_ms=40 age_data_rx_ms=1300 age_book_event_ms=1300 age_published_ms=1300 last_frame_kind=ping last_data_kind=snapshot stream_preference=depth1",
                        "WS_AUDIT venue=extended component=post_publish_stream_fallback action=started active_stream_kind=depth1 fallback_stream_kind=full_orderbook post_publish_fallback_after_ms=1200 post_publish_fallback_deadline_ms=1450 started_at_ms=1201 age_ws_rx_ms=50 age_data_rx_ms=1301 age_book_event_ms=1301 age_published_ms=1301 last_frame_kind=ping last_data_kind=snapshot stream_preference=depth1",
                        "WS_AUDIT venue=extended component=post_publish_stream_fallback action=fallback_won active_stream_kind=depth1 fallback_stream_kind=full_orderbook winner_stream_kind=full_orderbook post_publish_fallback_after_ms=1200 post_publish_fallback_deadline_ms=1450 started_at_ms=1201 age_ws_rx_ms=60 age_data_rx_ms=1400 age_book_event_ms=1400 age_published_ms=1400 last_frame_kind=ping last_data_kind=snapshot stream_preference=full_orderbook_degraded",
                        "WS_AUDIT venue=extended component=post_publish_stream_fallback action=preference_set active_stream_kind=full_orderbook fallback_stream_kind=full_orderbook winner_stream_kind=full_orderbook post_publish_fallback_after_ms=1200 post_publish_fallback_deadline_ms=1450 started_at_ms=1201 age_ws_rx_ms=60 age_data_rx_ms=1400 age_book_event_ms=1400 age_published_ms=1400 last_frame_kind=text last_data_kind=delta stream_preference=full_orderbook_degraded",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = ws_soak_report.parse_run_log(run_log)
            post_publish_summary = parsed[28]
            post_publish_stage_summary = parsed[29]
            stream_preference_summary = parsed[30]

            self.assertEqual(post_publish_summary["armed_count"], 1)
            self.assertEqual(post_publish_summary["started_count"], 1)
            self.assertEqual(post_publish_summary["fallback_won_count"], 1)
            self.assertEqual(post_publish_summary["preference_set_count"], 1)
            self.assertEqual(
                post_publish_summary["last_stream_preference"],
                "full_orderbook_degraded",
            )
            self.assertEqual(
                post_publish_stage_summary["dominant_stage"],
                "fallback_won",
            )
            self.assertEqual(
                stream_preference_summary["last_stream_preference"],
                "full_orderbook_degraded",
            )
            self.assertEqual(stream_preference_summary["degraded_active"], 1)


if __name__ == "__main__":
    unittest.main()
