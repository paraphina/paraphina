import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TestPhase51aiNonLighterOrderHistoryBridgeDiagnostic(unittest.TestCase):
    def test_materializes_extended_and_paradex_source_links_via_order_history_join(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ai_non_lighter_order_history_bridge_diagnostic.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            request_pack.mkdir()
            output_root = tmp_path / "runs"

            extended_external_id = "extended-client-001"
            extended_order_id = "9223372036854775808"
            extended_trade = {
                "id": "extended-trade-id",
                "accountId": "extended-account",
                "market": "ETH-USD",
                "orderId": extended_order_id,
                "price": "2288.1",
                "qty": "0.01",
                "isTaker": False,
            }
            extended_order = {
                "id": extended_order_id,
                "externalId": extended_external_id,
                "accountId": "extended-account",
                "market": "ETH-USD",
                "status": "FILLED",
                "postOnly": True,
                "takeProfit": {"starkExSignature": "secret-signature-value"},
            }

            paradex_client_id = "paradex-client-001"
            paradex_order_id = "1681462103821101699438490000"
            paradex_fill = {
                "id": "paradex-fill-id",
                "account": "0xabc123",
                "market": "ETH-USD-PERP",
                "order_id": paradex_order_id,
                "liquidity": "MAKER",
                "size": "0.01",
            }
            paradex_order = {
                "id": paradex_order_id,
                "client_id": paradex_client_id,
                "account": "0xabc123",
                "market": "ETH-USD-PERP",
                "status": "CLOSED",
            }

            (request_pack / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (request_pack / "candidate_manifest_with_empty_sidecar.json").write_text(
                json.dumps({"schema_version": 1, "sources": [], "source_links": []}),
                encoding="utf-8",
            )
            (request_pack / "source_link_request_targets.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "baseline_commit": BASELINE_COMMIT,
                            "venue_id": "extended",
                            "canonical_group_id": "extended-target",
                            "order_key": "extended-order-key",
                            "order_id_hash": None,
                            "client_order_id_hash": stable_hash(extended_external_id),
                            "no_live_flag": True,
                            "approved_for_live": False,
                            "live_orders_allowed": False,
                        },
                        {
                            "baseline_commit": BASELINE_COMMIT,
                            "venue_id": "paradex",
                            "canonical_group_id": "paradex-target",
                            "order_key": "paradex-order-key",
                            "order_id_hash": None,
                            "client_order_id_hash": stable_hash(paradex_client_id),
                            "no_live_flag": True,
                            "approved_for_live": False,
                            "live_orders_allowed": False,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (request_pack / "source_link_request_sources.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "baseline_commit": BASELINE_COMMIT,
                            "venue_id": "extended",
                            "source_record_sha256": stable_hash(extended_trade),
                            "no_live_flag": True,
                            "approved_for_live": False,
                            "live_orders_allowed": False,
                        },
                        {
                            "baseline_commit": BASELINE_COMMIT,
                            "venue_id": "paradex",
                            "source_record_sha256": stable_hash(paradex_fill),
                            "no_live_flag": True,
                            "approved_for_live": False,
                            "live_orders_allowed": False,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            extended_trades = tmp_path / "extended_trades.json"
            extended_trades.write_text(json.dumps({"data": [extended_trade]}), encoding="utf-8")
            extended_orders = tmp_path / "extended_orders_history.json"
            extended_orders.write_text(json.dumps({"data": [extended_order]}), encoding="utf-8")
            paradex_fills = tmp_path / "paradex_fills.json"
            paradex_fills.write_text(json.dumps({"results": [paradex_fill]}), encoding="utf-8")
            paradex_orders = tmp_path / "paradex_orders_history.json"
            paradex_orders.write_text(json.dumps({"results": [paradex_order]}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--extended-trades-json",
                    str(extended_trades),
                    "--extended-orders-history-json",
                    str(extended_orders),
                    "--paradex-fills-json",
                    str(paradex_fills),
                    "--paradex-orders-history-json",
                    str(paradex_orders),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ai_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ai_test"
            summary = json.loads(
                (run_dir / "phase51ai_non_lighter_order_history_bridge_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["materializable_source_link_count"], 2)
            self.assertEqual(
                summary["bridge_status_counts"],
                {"SOURCE_LINK_PROPOSED_ORDER_HISTORY_BRIDGE": 2},
            )

            sidecar = {
                (row["canonical_group_id"], row["source_record_sha256"])
                for row in (
                    json.loads(line)
                    for line in (run_dir / "source_links.proposed.sanitized.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            self.assertEqual(
                sidecar,
                {
                    ("extended-target", stable_hash(extended_trade)),
                    ("paradex-target", stable_hash(paradex_fill)),
                },
            )

            sanitized_extended_orders = (
                run_dir / "source_snapshots" / "extended_orders_history.sanitized.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(extended_external_id, sanitized_extended_orders)
            self.assertNotIn(extended_order_id, sanitized_extended_orders)
            self.assertNotIn("secret-signature-value", sanitized_extended_orders)
            self.assertNotIn("starkExSignature", sanitized_extended_orders)
            self.assertNotIn("value_sha256", sanitized_extended_orders)
            self.assertIn("externalId_sha256", sanitized_extended_orders)

            artifact_index = json.loads((run_dir / "evidence_pack" / "artifact_index.json").read_text(encoding="utf-8"))
            artifact_paths = {row["path"] for row in artifact_index["artifacts"]}
            self.assertIn("source_links.proposed.sanitized.jsonl", artifact_paths)
            self.assertIn("phase51ai_non_lighter_order_history_bridge_summary.json", artifact_paths)

    def test_rejects_direct_native_match_without_order_history_bridge(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ai_non_lighter_order_history_bridge_diagnostic.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            request_pack.mkdir()
            output_root = tmp_path / "runs"

            extended_order_id = "9223372036854775809"
            extended_trade = {
                "id": "extended-trade-id-direct-only",
                "market": "ETH-USD",
                "orderId": extended_order_id,
                "price": "2288.1",
                "qty": "0.01",
                "isTaker": False,
            }

            (request_pack / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (request_pack / "candidate_manifest_with_empty_sidecar.json").write_text(
                json.dumps({"schema_version": 1, "sources": [], "source_links": []}),
                encoding="utf-8",
            )
            (request_pack / "source_link_request_targets.jsonl").write_text(
                json.dumps(
                    {
                        "baseline_commit": BASELINE_COMMIT,
                        "venue_id": "extended",
                        "canonical_group_id": "extended-direct-only-target",
                        "order_key": "extended-direct-only-order-key",
                        "order_id_hash": stable_hash(extended_order_id),
                        "client_order_id_hash": None,
                        "no_live_flag": True,
                        "approved_for_live": False,
                        "live_orders_allowed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (request_pack / "source_link_request_sources.jsonl").write_text(
                json.dumps(
                    {
                        "baseline_commit": BASELINE_COMMIT,
                        "venue_id": "extended",
                        "source_record_sha256": stable_hash(extended_trade),
                        "no_live_flag": True,
                        "approved_for_live": False,
                        "live_orders_allowed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            extended_trades = tmp_path / "extended_trades.json"
            extended_trades.write_text(json.dumps({"data": [extended_trade]}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--venue",
                    "extended",
                    "--extended-trades-json",
                    str(extended_trades),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ai_direct_only_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ai_direct_only_test"
            summary = json.loads(
                (run_dir / "phase51ai_non_lighter_order_history_bridge_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["materializable_source_link_count"], 0)
            self.assertEqual(
                summary["bridge_status_counts"],
                {"DIRECT_NATIVE_TARGET_MATCH_WITHOUT_ORDER_HISTORY_BRIDGE_REJECTED": 1},
            )
            self.assertEqual(
                (run_dir / "source_links.proposed.sanitized.jsonl").read_text(encoding="utf-8"),
                "",
            )


if __name__ == "__main__":
    unittest.main()
