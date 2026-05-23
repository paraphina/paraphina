import json
import os
import tempfile
import unittest
from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import phase51ar_lighter_pressure_source_probe as probe  # noqa: E402


class EnvGuard:
    def __init__(self, *keys: str):
        self._saved = {key: os.environ.get(key) for key in keys}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Phase51ArLighterPressureSourceProbeTests(unittest.TestCase):
    def test_complete_captured_account_limits_reports_dimensions_without_clearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "account_limits.json"
            source.write_text(
                json.dumps(
                    {
                        "active_order_headroom_account": 17,
                        "sendtx_per_minute_limit": 4000,
                        "sendtx_per_minute_remaining": 3999,
                        "rest_requests_per_minute_limit": 24000,
                        "rest_requests_per_minute_remaining": 23900,
                    }
                ),
                encoding="utf-8",
            )

            out_dir = probe.run(
                output_root=tmp_path / "runs",
                run_id="complete",
                account_limits_json=source,
                fetch_readonly=False,
                base_url=probe.DEFAULT_BASE_URL,
                account_index=None,
                auth_token_env="LIGHTER_AUTH_TOKEN",
                timeout_s=1.0,
            )

            summary = read_json(out_dir / "lighter_pressure_source_probe_summary.json")
            pressure = read_json(out_dir / "account_limits_pressure.sanitized.json")

            self.assertEqual(summary["fetch_status"], "OBSERVED_FROM_CAPTURED_JSON")
            self.assertTrue(summary["pressure_dimensions_complete_from_account_limits_surface"])
            self.assertFalse(summary["blocker_cleared"])
            self.assertFalse(summary["safe_to_run_phase51_validators"])
            self.assertFalse(summary["phase51_validators_run"])
            self.assertEqual(
                summary["native_limit_event_time_status"],
                "READONLY_ACCOUNT_LIMITS_SOURCE_AVAILABILITY_ONLY",
            )
            self.assertEqual(
                pressure["observed_values"]["sendtx_per_minute_remaining"],
                3999,
            )

    def test_missing_source_fields_routes_to_passive_sendtx_tap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "account_limits.json"
            source.write_text(
                json.dumps({"active_orders_per_account_limit": 1500, "user_tier": "premium"}),
                encoding="utf-8",
            )

            out_dir = probe.run(
                output_root=tmp_path / "runs",
                run_id="missing",
                account_limits_json=source,
                fetch_readonly=False,
                base_url=probe.DEFAULT_BASE_URL,
                account_index=None,
                auth_token_env="LIGHTER_AUTH_TOKEN",
                timeout_s=1.0,
            )

            summary = read_json(out_dir / "lighter_pressure_source_probe_summary.json")
            pressure = read_json(out_dir / "account_limits_pressure.sanitized.json")

            self.assertFalse(summary["pressure_dimensions_complete_from_account_limits_surface"])
            self.assertEqual(summary["recommended_next_path"], "passive_sendtx_sendtxbatch_observation_tap")
            self.assertIn(
                "sendtx_per_minute_limit/sendtx_per_minute_remaining",
                pressure["required_missing_dimensions"],
            )

    def test_sanitized_headers_drop_auth_and_map_numeric_pressure_candidates(self):
        headers = probe._sanitize_response_headers(
            {
                "Authorization": "secret-token",
                "Set-Cookie": "session=secret",
                "X-RateLimit-Limit": "24000",
                "X-RateLimit-Remaining": "23999",
                "X-SendTx-Limit": "4000",
                "X-SendTx-Remaining": "3998",
            }
        )

        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Set-Cookie", headers)
        pressure = probe._extract_pressure({}, headers)
        self.assertEqual(pressure["observed_values"]["rest_requests_per_minute_limit"], 24000)
        self.assertEqual(pressure["observed_values"]["rest_requests_per_minute_remaining"], 23999)
        self.assertEqual(pressure["observed_values"]["sendtx_per_minute_limit"], 4000)
        self.assertEqual(pressure["observed_values"]["sendtx_per_minute_remaining"], 3998)

    def test_outputs_do_not_persist_raw_identifier_or_secret_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "account_limits.json"
            source.write_text(
                json.dumps(
                    {
                        "api_key": "secret-value",
                        "order_id": "raw-order-value",
                        "sendtx_per_minute_limit": 4000,
                    }
                ),
                encoding="utf-8",
            )

            out_dir = probe.run(
                output_root=tmp_path / "runs",
                run_id="redaction",
                account_limits_json=source,
                fetch_readonly=False,
                base_url=probe.DEFAULT_BASE_URL,
                account_index=None,
                auth_token_env="LIGHTER_AUTH_TOKEN",
                timeout_s=1.0,
            )

            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(out_dir.iterdir())
                if path.is_file()
            )
            self.assertNotIn("secret-value", combined)
            self.assertNotIn("raw-order-value", combined)
            summary = read_json(out_dir / "lighter_pressure_source_probe_summary.json")
            self.assertGreaterEqual(summary["payload_fingerprint"]["raw_or_secret_shaped_key_count"], 2)
            self.assertFalse(summary["raw_identifiers_persisted"])
            self.assertFalse(summary["secrets_persisted"])

    def test_fetch_readonly_missing_auth_writes_hold_without_network_or_secret(self):
        with tempfile.TemporaryDirectory() as tmp, EnvGuard("LIGHTER_AUTH_TOKEN", "LIGHTER_ACCOUNT_INDEX"):
            os.environ.pop("LIGHTER_AUTH_TOKEN", None)
            os.environ["LIGHTER_ACCOUNT_INDEX"] = "123"
            tmp_path = Path(tmp)

            out_dir = probe.run(
                output_root=tmp_path / "runs",
                run_id="missing-auth",
                account_limits_json=None,
                fetch_readonly=True,
                base_url=probe.DEFAULT_BASE_URL,
                account_index=None,
                auth_token_env="LIGHTER_AUTH_TOKEN",
                timeout_s=1.0,
            )

            summary = read_json(out_dir / "lighter_pressure_source_probe_summary.json")
            self.assertEqual(summary["fetch_status"], "NOT_RUN_MISSING_AUTH_TOKEN")
            self.assertFalse(summary["auth_token_env_present"])
            self.assertFalse(summary["blocker_cleared"])
            self.assertFalse(summary["phase51_validators_run"])


if __name__ == "__main__":
    unittest.main()
