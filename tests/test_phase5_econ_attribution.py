import importlib.util
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "phase5_econ_attribution.py"
    tools_dir = module_path.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("phase5_econ_attribution_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_aster_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "phase5_aster_fee_path_attribution.py"
    tools_dir = module_path.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("phase5_aster_fee_path_attribution_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPhase5EconAttribution(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def _row(
        self,
        venue: str,
        *,
        baseline: str,
        current: str,
        realized: str = "0",
        fees: str = "0",
        funding: str = "0",
        transfers: str = "0",
        spot: str = "0",
        confidence: str = "medium",
    ):
        return self.mod.VenueLedger(
            venue=venue,
            baseline_balance_usd=Decimal(baseline),
            current_balance_usd=Decimal(current),
            realized_pnl_usd=Decimal(realized),
            fees_usd=Decimal(fees),
            funding_usd=Decimal(funding),
            transfers_usd=Decimal(transfers),
            spot_revaluation_usd=Decimal(spot),
            confidence=confidence,
        ).finalize()

    def test_load_baseline_manifest_preserves_lighter_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.yaml"
            path.write_text(
                "\n".join(
                    [
                        'captured_at_utc: "2026-04-16T00:00:00Z"',
                        "source: manual_user_baseline",
                        "lighter_spot_included: true",
                        "venues:",
                        "  hyperliquid:",
                        "    balance_usd: 81.92",
                        "  extended:",
                        "    balance_usd: 74.71",
                        "  lighter:",
                        "    perps_usd: 103.18",
                        "    spot_usd: 10.95",
                        "    total_usd: 114.13",
                        "  aster:",
                        "    balance_usd: 74.84",
                        "  paradex:",
                        "    balance_usd: 55.63",
                    ]
                ),
                encoding="utf-8",
            )

            payload = self.mod.load_baseline_manifest(path)
            self.assertEqual(payload["captured_at_utc"], "2026-04-16T00:00:00Z")
            self.assertTrue(payload["lighter_spot_included"])
            self.assertEqual(payload["venues"]["lighter_perp"], Decimal("103.18"))
            self.assertEqual(payload["venues"]["lighter_spot"], Decimal("10.95"))
            self.assertEqual(payload["venues"]["lighter_total"], Decimal("114.13"))

    def test_recommend_child_prefers_lighter_when_zero_fee_and_dominant(self):
        rows = {
            "hyperliquid": self._row("hyperliquid", baseline="81.92", current="75.35", realized="-2.34", fees="-8.62", funding="0.001"),
            "extended": self._row("extended", baseline="74.71", current="70.75", realized="-2.65", fees="-1.50", funding="-0.003"),
            "lighter": self._row("lighter", baseline="114.13", current="97.30", realized="-16.70", fees="0", funding="0.007", spot="-0.23", confidence="high"),
            "aster": self._row("aster", baseline="74.84", current="60.64", realized="-6.23", fees="-9.04", funding="-0.001"),
            "paradex": self._row("paradex", baseline="55.63", current="54.15", realized="-1.20", fees="-0.35", funding="-0.0004"),
        }

        recommendation = self.mod.recommend_child(rows)
        self.assertEqual(
            recommendation["recommended_child"],
            "phase5_all5_current_surface_lighter_adverse_selection_markout_requal",
        )
        self.assertIn("Lighter", recommendation["recommended_child_reason"])
        self.assertEqual(
            recommendation["non_target_venues_frozen"],
            ["extended", "paradex"],
        )

    def test_recommend_child_falls_back_to_soft_cap_when_no_venue_dominates(self):
        rows = {
            "hyperliquid": self._row("hyperliquid", baseline="81.92", current="79.50", realized="-1.00", fees="-0.50", funding="0.01"),
            "extended": self._row("extended", baseline="74.71", current="73.20", realized="-0.90", fees="-0.30", funding="-0.01"),
            "lighter": self._row("lighter", baseline="114.13", current="110.90", realized="-1.50", fees="0", funding="0.00", spot="-0.20"),
            "aster": self._row("aster", baseline="74.84", current="72.00", realized="-1.10", fees="-1.00", funding="-0.01"),
            "paradex": self._row("paradex", baseline="55.63", current="54.80", realized="-0.40", fees="-0.10", funding="0.00"),
        }

        recommendation = self.mod.recommend_child(rows)
        self.assertEqual(
            recommendation["recommended_child"],
            "phase5_all5_current_surface_soft_cap_starvation_quote_budget_activation_requal",
        )


class TestPhase5AsterFeePathAttribution(unittest.TestCase):
    def setUp(self):
        self.mod = _load_aster_module()

    def test_iter_jsonl_accepts_concatenated_recovered_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry_bounded.jsonl"
            path.write_text('{"t":1,"fills":[]}{"t":2,"orders":[]}\n', encoding="utf-8")

            rows = list(self.mod.iter_jsonl(path))

        self.assertEqual([row["t"] for row in rows], [1, 2])

    def test_join_classifies_matched_mm_gtx_maker(self):
        telemetry = self.mod.TelemetryTruth()
        telemetry.fills_by_order_id["100"] = {
            "order_id": "100",
            "client_order_id": "co_mm",
            "purpose": "Mm",
            "size": 0.01,
        }
        telemetry.orders_by_client_order_id["co_mm"] = {
            "client_order_id": "co_mm",
            "post_only": True,
            "reduce_only": False,
            "purpose": "Mm",
        }
        trades = [
            {
                "id": "t1",
                "orderId": "100",
                "price": "2324.50",
                "qty": "0.01",
                "quoteQty": "23.245",
                "commission": "0",
                "realizedPnl": "-0.01",
                "maker": True,
                "time": 1776981542244,
                "side": "SELL",
            }
        ]
        orders = [{"orderId": "100", "clientOrderId": "co_mm", "timeInForce": "GTX", "reduceOnly": False}]

        joined = self.mod.join_trades(trades, orders, telemetry)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].category, self.mod.CATEGORY_MAKER_MM)
        self.assertTrue(joined[0].telemetry_matched)
        self.assertTrue(joined[0].order_matched)

    def test_join_classifies_taker_hedge_as_unwind_fee_path(self):
        telemetry = self.mod.TelemetryTruth()
        telemetry.fills_by_order_id["101"] = {
            "order_id": "101",
            "client_order_id": "co_hedge",
            "purpose": "Hedge",
            "size": 0.002,
        }
        trades = [
            {
                "id": "t2",
                "orderId": "101",
                "price": "2336.22",
                "qty": "0.002",
                "quoteQty": "4.67244",
                "commission": "0.001868976",
                "realizedPnl": "0",
                "maker": False,
                "time": 1776981537762,
                "side": "BUY",
            }
        ]
        orders = [{"orderId": "101", "clientOrderId": "co_hedge", "timeInForce": "IOC", "reduceOnly": True}]

        joined = self.mod.join_trades(trades, orders, telemetry)

        self.assertEqual(joined[0].category, self.mod.CATEGORY_TAKER_UNWIND)
        self.assertEqual(joined[0].reduce_only, True)

    def test_low_join_quality_recommends_order_truth_child(self):
        telemetry = self.mod.TelemetryTruth()
        joined = self.mod.join_trades(
            [
                {
                    "id": "t3",
                    "orderId": "missing",
                    "price": "2324.50",
                    "qty": "0.01",
                    "quoteQty": "23.245",
                    "commission": "0.009298",
                    "realizedPnl": "0",
                    "maker": False,
                    "time": 1776981542244,
                }
            ],
            [],
            telemetry,
        )
        summary = self.mod.summarize_joined(
            joined,
            [],
            telemetry,
            {"telemetry_path": "fixture", "telemetry_available": True},
            start_ms=1,
            end_ms=2,
        )

        recommendation = self.mod.recommend_next_child(summary)

        self.assertEqual(recommendation["confidence"], "low")
        self.assertEqual(
            recommendation["recommended_child"],
            "phase5_all5_current_surface_aster_order_truth_join_requal",
        )

    def test_unwind_commission_dominance_recommends_fee_guard(self):
        telemetry = self.mod.TelemetryTruth()
        telemetry.fills_by_order_id["101"] = {"order_id": "101", "client_order_id": "co_hedge", "purpose": "Hedge"}
        telemetry.fills_by_order_id["102"] = {"order_id": "102", "client_order_id": "co_mm", "purpose": "Mm"}
        trades = [
            {
                "id": "t1",
                "orderId": "101",
                "price": "2300",
                "qty": "0.01",
                "quoteQty": "23",
                "commission": "0.0092",
                "realizedPnl": "0",
                "maker": False,
                "time": 1,
            },
            {
                "id": "t2",
                "orderId": "102",
                "price": "2300",
                "qty": "0.01",
                "quoteQty": "23",
                "commission": "0",
                "realizedPnl": "-0.01",
                "maker": True,
                "time": 2,
            },
        ]
        orders = [
            {"orderId": "101", "clientOrderId": "co_hedge", "timeInForce": "IOC", "reduceOnly": True},
            {"orderId": "102", "clientOrderId": "co_mm", "timeInForce": "GTX", "reduceOnly": False},
        ]
        joined = self.mod.join_trades(trades, orders, telemetry)
        summary = self.mod.summarize_joined(
            joined,
            [],
            telemetry,
            {"telemetry_path": "fixture", "telemetry_available": True},
            start_ms=1,
            end_ms=2,
        )

        recommendation = self.mod.recommend_next_child(summary)

        self.assertEqual(recommendation["confidence"], "high")
        self.assertEqual(
            recommendation["recommended_child"],
            "phase5_all5_current_surface_aster_reduce_only_inventory_brake_fee_guard_requal",
        )


if __name__ == "__main__":
    unittest.main()
