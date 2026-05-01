import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "telemetry_analyzer.py"
    spec = importlib.util.spec_from_file_location("telemetry_analyzer_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTelemetryAnalyzerResidualFallback(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_metrics_snapshot_summarizes_emergency_residual_fallback(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 100,
                "execution_mode": "live",
                "emergency_residual_fallback": {
                    "aster_inventory_brake_fee_guard_enabled": True,
                    "aster_inventory_brake_fee_guard_skipped_orders": 2,
                    "aster_inventory_brake_fee_guard_skipped_base_tao": 0.012,
                    "aster_inventory_brake_fee_guard_skipped_notional_usd": 27.5,
                    "records": [
                        {
                            "class": "soft_unwind",
                            "venue_index": 0,
                            "venue_id": "extended",
                            "status": "used",
                            "reason": "no_fresh_account",
                        },
                        {
                            "class": "inventory_brake",
                            "venue_index": 4,
                            "venue_id": "paradex",
                            "status": "rejected",
                            "reason": "live_orders_present",
                        },
                    ]
                },
            }
        )
        acc.process({"t": 101, "execution_mode": "live"})

        payload = self.mod.build_metrics_snapshot(acc)
        summary = payload["emergency_residual_fallback_summary"]
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["by_status"]["used"], 1)
        self.assertEqual(summary["by_status"]["rejected"], 1)
        self.assertEqual(summary["by_reason"]["no_fresh_account"], 1)
        self.assertEqual(summary["by_reason"]["live_orders_present"], 1)
        self.assertEqual(summary["by_class"]["soft_unwind"], 1)
        self.assertEqual(summary["by_class"]["inventory_brake"], 1)
        self.assertEqual(summary["by_venue"]["extended"], 1)
        self.assertEqual(summary["by_venue"]["paradex"], 1)
        self.assertEqual(summary["first_tick"], 100)
        self.assertEqual(summary["last_tick"], 100)
        self.assertTrue(summary["aster_inventory_brake_fee_guard_enabled"])
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_orders"], 2)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_base_tao"], 0.012)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_notional_usd"], 27.5)

    def test_metrics_snapshot_summarizes_quote_gate_by_venue_side(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "quote_levels": [
                    {
                        "venue_index": 0,
                        "venue_id": "extended",
                        "side": "Bid",
                        "quote_state": "suppressed",
                        "suppression_reason": "edge_below_min",
                        "engine_terminal_reason": "edge_below_min",
                        "edge_threshold": 2.75,
                        "edge_threshold_base": 1.79,
                        "hedge_cost_edge_floor": 0.96,
                        "edge_local": 0.0,
                    },
                    {
                        "venue_index": 0,
                        "venue_id": "extended",
                        "side": "Ask",
                        "quote_state": "active",
                        "engine_terminal_reason": "active",
                        "edge_threshold": 1.08,
                        "edge_threshold_base": 0.12,
                        "hedge_cost_edge_floor": 0.96,
                        "edge_local": 1.20,
                    },
                ],
            }
        )

        summary = self.mod.build_metrics_snapshot(acc)["quote_gate_by_venue_side"]["extended"]
        self.assertEqual(summary["Bid"]["samples"], 1)
        self.assertEqual(summary["Bid"]["active"], 0)
        self.assertEqual(summary["Bid"]["suppressed"], 1)
        self.assertEqual(summary["Bid"]["suppression_reasons"]["edge_below_min"], 1)
        self.assertEqual(summary["Bid"]["engine_terminal_reasons"]["edge_below_min"], 1)
        self.assertAlmostEqual(summary["Bid"]["edge_threshold_base"]["mean"], 1.79)
        self.assertAlmostEqual(summary["Bid"]["hedge_cost_edge_floor"]["mean"], 0.96)
        self.assertEqual(summary["Ask"]["active"], 1)
        self.assertAlmostEqual(summary["Ask"]["active_edge_local"]["mean"], 1.20)

    def test_metrics_snapshot_builds_opportunity_adjusted_scorecard(self):
        acc = self.mod.TelemetryAccumulator()
        for i in range(199):
            acc.process(
                {
                    "t": i,
                    "execution_mode": "live",
                    "quote_levels": [
                        {
                            "venue_index": 1,
                            "venue_id": "hyperliquid",
                            "side": "Ask",
                            "quote_state": "active",
                            "engine_terminal_reason": "active",
                            "edge_threshold": 1.0,
                            "edge_threshold_base": 0.1,
                            "hedge_cost_edge_floor": 0.9,
                            "edge_local": 1.1,
                        }
                    ],
                }
            )
        payload = self.mod.build_metrics_snapshot(acc)
        hyperliquid = payload["opportunity_adjusted_scorecard"]["hyperliquid"]
        self.assertTrue(hyperliquid["passed"])
        self.assertEqual(hyperliquid["reason"], "insufficient_active_quote_sample")
        self.assertEqual(hyperliquid["active_quote_samples"], 199)

        acc.process(
            {
                "t": 200,
                "execution_mode": "live",
                "quote_levels": [
                    {
                        "venue_index": 1,
                        "venue_id": "hyperliquid",
                        "side": "Ask",
                        "quote_state": "active",
                        "engine_terminal_reason": "active",
                        "edge_threshold": 1.0,
                        "edge_threshold_base": 0.1,
                        "hedge_cost_edge_floor": 0.9,
                        "edge_local": 1.1,
                    }
                ],
            }
        )
        hyperliquid = self.mod.build_metrics_snapshot(acc)["opportunity_adjusted_scorecard"]["hyperliquid"]
        self.assertFalse(hyperliquid["passed"])
        self.assertEqual(hyperliquid["reason"], "active_quote_underconversion")

    def test_opportunity_adjusted_scorecard_passes_with_mm_fill(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "fills": [
                    {
                        "venue_index": 4,
                        "venue_id": "paradex",
                        "purpose": "Mm",
                        "size": 0.01,
                    }
                ],
            }
        )

        paradex = self.mod.build_metrics_snapshot(acc)["opportunity_adjusted_scorecard"]["paradex"]
        self.assertTrue(paradex["passed"])
        self.assertEqual(paradex["reason"], "mm_fill_evidence")
        self.assertEqual(paradex["mm_fills"], 1)

    def test_metrics_snapshot_uses_cumulative_inventory_brake_fee_guard_maxima(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 100,
                "execution_mode": "live",
                "emergency_residual_fallback": {
                    "aster_inventory_brake_fee_guard_enabled": True,
                    "aster_inventory_brake_fee_guard_skipped_orders": 1,
                    "aster_inventory_brake_fee_guard_skipped_base_tao": 0.004,
                    "aster_inventory_brake_fee_guard_skipped_notional_usd": 9.25,
                    "records": [],
                },
            }
        )
        acc.process(
            {
                "t": 101,
                "execution_mode": "live",
                "emergency_residual_fallback": {
                    "aster_inventory_brake_fee_guard_enabled": True,
                    "aster_inventory_brake_fee_guard_skipped_orders": 3,
                    "aster_inventory_brake_fee_guard_skipped_base_tao": 0.011,
                    "aster_inventory_brake_fee_guard_skipped_notional_usd": 25.0,
                    "records": [],
                },
            }
        )

        summary = self.mod.build_metrics_snapshot(acc)["emergency_residual_fallback_summary"]
        self.assertTrue(summary["aster_inventory_brake_fee_guard_enabled"])
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_orders"], 3)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_base_tao"], 0.011)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_notional_usd"], 25.0)

    def test_metrics_snapshot_defaults_when_no_residual_fallback_records(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process({"t": 1, "execution_mode": "live"})

        payload = self.mod.build_metrics_snapshot(acc)
        summary = payload["emergency_residual_fallback_summary"]
        self.assertEqual(summary["total_records"], 0)
        self.assertEqual(summary["by_status"], {})
        self.assertEqual(summary["by_reason"], {})
        self.assertEqual(summary["by_class"], {})
        self.assertEqual(summary["by_venue"], {})
        self.assertIsNone(summary["first_tick"])
        self.assertIsNone(summary["last_tick"])
        self.assertFalse(summary["aster_inventory_brake_fee_guard_enabled"])
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_orders"], 0)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_base_tao"], 0.0)
        self.assertEqual(summary["aster_inventory_brake_fee_guard_skipped_notional_usd"], 0.0)

    def test_metrics_snapshot_summarizes_projected_mm_budget(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "projected_mm_budget": {
                    "configured": True,
                    "applied": True,
                    "selected_venues": ["extended", "hyperliquid", "aster", "lighter", "paradex"],
                    "suppressed_venues": ["hyperliquid"],
                    "net_limit_tao": 0.06,
                    "gross_limit_tao": 0.06,
                    "venue_limit_tao": 0.0225,
                    "projected_q_global_after_tao": 0.0,
                    "projected_q_gross_after_tao": 0.05,
                    "projected_q_max_abs_venue_after_tao": 0.01,
                },
            }
        )
        acc.process(
            {
                "t": 2,
                "execution_mode": "live",
                "projected_mm_budget": {
                    "configured": True,
                    "applied": False,
                    "selected_venues": ["lighter"],
                    "suppressed_venues": [],
                    "net_limit_tao": 0.06,
                    "gross_limit_tao": 0.06,
                    "venue_limit_tao": 0.0225,
                    "projected_q_global_after_tao": 0.0,
                    "projected_q_gross_after_tao": 0.01,
                    "projected_q_max_abs_venue_after_tao": 0.01,
                },
            }
        )

        payload = self.mod.build_metrics_snapshot(acc)
        summary = payload["projected_mm_budget_summary"]
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["configured_ticks"], 2)
        self.assertEqual(summary["applied_ticks"], 1)
        self.assertEqual(summary["selected_counts"]["lighter"], 2)
        self.assertEqual(summary["selected_counts"]["hyperliquid"], 1)
        self.assertEqual(summary["suppressed_counts"]["hyperliquid"], 1)
        self.assertEqual(summary["all5_selected_ticks"], 1)
        self.assertEqual(summary["last_limits"]["gross_limit_tao"], 0.06)
        self.assertEqual(summary["projected_after"]["q_gross_tao"]["max"], 0.05)

    def test_metrics_snapshot_summarizes_canary_zero_target_hold(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 10,
                "execution_mode": "live",
                "canary_breach_response": {
                    "active": True,
                    "response_mode": "zero_target_hold",
                    "candidate_target_venues": ["aster", "lighter"],
                    "candidate_positioned_target_venues": ["aster", "lighter"],
                    "target_venues": [],
                    "flatten_venues": [],
                    "request_dispatches": [],
                    "observation_active": True,
                    "observation_venues": ["aster"],
                    "observation_covers_candidate_targets": False,
                    "zero_target_hold_this_tick": True,
                },
            }
        )
        acc.process(
            {
                "t": 11,
                "execution_mode": "live",
                "canary_breach_response": {
                    "active": True,
                    "response_mode": "observation",
                    "candidate_target_venues": ["aster"],
                    "candidate_positioned_target_venues": ["aster"],
                    "target_venues": [],
                    "flatten_venues": [],
                    "request_dispatches": [],
                    "observation_active": True,
                    "observation_venues": ["aster"],
                    "observation_covers_candidate_targets": True,
                    "zero_target_hold_this_tick": False,
                },
            }
        )

        payload = self.mod.build_metrics_snapshot(acc)
        summary = payload["canary_breach_response_summary"]
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["active_ticks"], 2)
        self.assertEqual(summary["candidate_target_ticks"], 2)
        self.assertEqual(summary["observation_active_ticks"], 2)
        self.assertEqual(summary["observation_uncovered_target_ticks"], 1)
        self.assertEqual(summary["zero_target_hold_ticks"], 1)
        self.assertEqual(summary["zero_target_hold_windows"], 1)
        self.assertEqual(summary["max_zero_target_hold_run"], 1)
        self.assertEqual(summary["first_zero_target_hold_tick"], 10)
        self.assertEqual(summary["last_zero_target_hold_tick"], 10)
        self.assertEqual(summary["response_modes"]["zero_target_hold"], 1)
        self.assertEqual(summary["response_modes"]["observation"], 1)
        self.assertEqual(payload["anomalies"]["by_category"]["canary_zero_target_hold"], 1)


class TestTelemetryAnalyzerEconomicsAttribution(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_metrics_snapshot_exports_per_venue_economics_attribution(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "venue_markout_ewma_usd_per_tao": [0.0, -0.12, 0.2, 0.0, 0.0],
                "venue_toxicity": [0.0, 0.7, 0.2, 0.0, 0.0],
                "fills": [
                    {
                        "purpose": "Mm",
                        "decision_id": "d_hl",
                        "venue_index": 1,
                        "venue_id": "hyperliquid",
                        "price": 2000.0,
                        "size": 0.01,
                        "fee_bps": 5.0,
                        "realised_pnl_usd": -0.02,
                        "markout_pnl_short": -0.03,
                    },
                    {
                        "purpose": "Mm",
                        "decision_id": "d_as",
                        "venue_index": 2,
                        "venue_id": "aster",
                        "price": 2100.0,
                        "size": 0.02,
                        "fee_bps": 0.0,
                        "realised_pnl_usd": 0.04,
                        "markout_pnl_short": 0.01,
                    },
                ],
            }
        )

        payload = self.mod.build_metrics_snapshot(acc)
        econ = payload["economics_attribution"]
        self.assertEqual(econ["mm_fill_attributed_count"], 2)
        self.assertEqual(econ["mm_fill_unattributed_count"], 0)

        hl = econ["per_venue"]["hyperliquid"]
        self.assertEqual(hl["mm_fill_attributed_count"], 1)
        self.assertAlmostEqual(hl["mm_fee_usd"], 0.01, places=6)
        self.assertAlmostEqual(hl["mm_realised_net_usd"], -0.02, places=6)
        self.assertAlmostEqual(hl["mm_gross_before_fee_usd"], -0.01, places=6)
        self.assertAlmostEqual(hl["mm_markout_short_usd"], -0.03, places=6)
        self.assertEqual(hl["venue_markout_stats"]["count"], 1)
        self.assertAlmostEqual(hl["venue_markout_stats"]["mean"], -0.12, places=6)
        self.assertEqual(hl["venue_toxicity_stats"]["count"], 1)
        self.assertAlmostEqual(hl["venue_toxicity_stats"]["mean"], 0.7, places=6)

        aster = econ["per_venue"]["aster"]
        self.assertEqual(aster["mm_fill_attributed_count"], 1)
        self.assertAlmostEqual(aster["mm_fee_usd"], 0.0, places=6)
        self.assertAlmostEqual(aster["mm_realised_net_usd"], 0.04, places=6)
        self.assertAlmostEqual(aster["mm_markout_short_usd"], 0.01, places=6)
        self.assertEqual(aster["venue_markout_stats"]["count"], 1)
        self.assertAlmostEqual(aster["venue_markout_stats"]["mean"], 0.2, places=6)

    def test_metrics_snapshot_defaults_economics_attribution_without_fills(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process({"t": 1, "execution_mode": "live"})

        payload = self.mod.build_metrics_snapshot(acc)
        econ = payload["economics_attribution"]
        self.assertEqual(econ["mm_fill_attributed_count"], 0)
        self.assertEqual(econ["mm_fill_unattributed_count"], 0)
        for venue, venue_payload in econ["per_venue"].items():
            self.assertEqual(venue_payload["mm_fill_attributed_count"], 0, msg=venue)
            self.assertEqual(venue_payload["venue_markout_stats"]["count"], 0, msg=venue)
            self.assertEqual(venue_payload["venue_toxicity_stats"]["count"], 0, msg=venue)

    def test_metrics_snapshot_charges_unattributed_hedge_exec_cost_to_top_level_net(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "hedges": [
                    {
                        "venue_index": 2,
                        "venue_id": "aster",
                        "intended_size": 0.2,
                        "cost_components": {
                            "exec_cost": 1.5,
                            "total_cost": 2.0,
                        },
                    }
                ],
            }
        )

        payload = self.mod.build_metrics_snapshot(acc)
        econ = payload["economics_attribution"]
        self.assertEqual(econ["hedge_fill_unattributed_count"], 0)
        self.assertAlmostEqual(econ["hedge_exec_cost_model_unattributed_usd"], 0.3, places=6)
        self.assertAlmostEqual(econ["hedge_total_cost_model_unattributed_usd"], 0.4, places=6)
        self.assertAlmostEqual(econ["net_after_hedge_exec_model_attributed_usd"], 0.0, places=6)
        self.assertAlmostEqual(econ["net_after_hedge_exec_model_unattributed_usd"], -0.3, places=6)
        self.assertAlmostEqual(econ["net_after_hedge_exec_model_usd"], -0.3, places=6)

    def test_metrics_snapshot_keeps_default_venues_with_sparse_indexed_metadata(self):
        acc = self.mod.TelemetryAccumulator()
        acc.process(
            {
                "t": 1,
                "execution_mode": "live",
                "inventory_attribution": [
                    {
                        "venue_index": 4,
                        "venue_id": "paradex",
                    },
                ],
                "venue_status": ["Healthy", "Healthy", "Healthy", "Healthy", "Healthy"],
            }
        )

        payload = self.mod.build_metrics_snapshot(acc)
        self.assertEqual(
            payload["venue_names"],
            ["extended", "hyperliquid", "aster", "lighter", "paradex"],
        )
        self.assertEqual(set(payload["venue_health"]), set(payload["venue_names"]))
        self.assertIn("extended", payload["economics_attribution"]["per_venue"])
        self.assertIn("paradex", payload["economics_attribution"]["per_venue"])


if __name__ == "__main__":
    unittest.main()
