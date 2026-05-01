import importlib.util
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "phase5_balance_snapshot.py"
    tools_dir = module_path.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("phase5_balance_snapshot_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decimal_string_preserves_precision_without_float_quantization():
    mod = _load_module()

    assert mod.decimal_str(Decimal("81.920000123456789")) == "81.920000123456789"
    assert mod.decimal_str("0.001000000000001") == "0.001000000000001"


def test_collectors_extract_expected_fields_and_write_raw_details(tmp_path):
    mod = _load_module()
    env = {
        "HL_VAULT_ADDRESS": "0xhl",
        "EXTENDED_API_KEY": "extended-key",
        "LIGHTER_ACCOUNT_INDEX": "7",
        "ASTER_API_KEY": "aster-key",
        "PARADEX_READONLY_TOKEN": "paradex-token",
    }

    def fake_http_json(method, url, *, headers=None, payload=None, params=None):
        if method == "POST" and payload == {"type": "userAbstraction", "user": "0xhl"}:
            return "disabled"
        if method == "POST" and payload == {"type": "clearinghouseState", "user": "0xhl"}:
            return {"marginSummary": {"accountValue": "63.660000123456789"}}
        if method == "POST" and payload == {"type": "spotClearinghouseState", "user": "0xhl"}:
            return {"balances": [{"coin": "USDC", "total": "0.006288"}]}
        if url.endswith("/api/v1/user/account/info"):
            return {"data": {"accountId": 123}}
        if url.endswith("/api/v1/portfolio/charts/equities"):
            return {"data": [{"value": "74.710000123456789"}]}
        if url.endswith("/api/v1/account"):
            return {"accounts": [{"collateral": "103.180000123456789", "l1_address": "0xabc"}]}
        if url.endswith("/api/v1/pnl"):
            return {"pnl": [{"trade_spot_pnl": "10.950000000000001"}]}
        if "/fapi/v2/account" in url:
            return {"totalWalletBalance": "74.840000123456789", "assets": []}
        if url.endswith("/account"):
            return {"account_value": "55.630000123456789", "account": "pdx"}
        raise AssertionError(f"unexpected request: {method} {url} {payload} {params}")

    with mock.patch.object(mod, "http_json", side_effect=fake_http_json), mock.patch.object(
        mod, "lighter_auth_token", return_value="lighter-token"
    ), mock.patch.object(mod, "paradex_token", return_value="paradex-token"), mock.patch.object(
        mod, "sign_aster_query", return_value="timestamp=1&signature=x"
    ):
        rows = mod.collect_rows(env, tmp_path, "pre", "2026-04-28T18:00:00Z", 1, 2)

    by_venue = {row["venue"]: row for row in rows}
    assert by_venue["hyperliquid"]["balance_usd"] == "63.666288123456789"
    assert by_venue["hyperliquid"]["balance_components"] == {
        "account_mode": "disabled",
        "perps_account_value_usd": "63.660000123456789",
        "spot_usdc_total": "0.006288",
        "total_usd": "63.666288123456789",
    }
    assert by_venue["extended"]["balance_usd"] == "74.710000123456789"
    assert by_venue["lighter"]["balance_components"] == {
        "perps_usd": "103.180000123456789",
        "spot_usd": "10.950000000000001",
        "total_usd": "114.130000123456790",
    }
    assert by_venue["aster"]["balance_usd"] == "74.840000123456789"
    assert by_venue["paradex"]["balance_usd"] == "55.630000123456789"

    for venue in mod.VENUE_ORDER:
        assert (tmp_path / "balances" / "pre_raw" / f"{venue}.json").exists()
        assert (tmp_path / "balances" / "pre_details" / f"{venue}.json").exists()
        row = by_venue[venue]
        assert isinstance(row["balance_usd"], str)
        assert all(isinstance(value, str) for value in row["balance_components"].values())


def test_builds_attribution_compatible_baseline_manifest(tmp_path):
    mod = _load_module()
    rows = [
        {
            "venue": "hyperliquid",
            "balance_usd": "81.920000123456789",
            "balance_components": {"usdc_total": "81.920000123456789"},
        },
        {
            "venue": "extended",
            "balance_usd": "74.710000123456789",
            "balance_components": {"equity_value_usd": "74.710000123456789"},
        },
        {
            "venue": "lighter",
            "balance_usd": "114.130000123456790",
            "balance_components": {
                "perps_usd": "103.180000123456789",
                "spot_usd": "10.950000000000001",
                "total_usd": "114.130000123456790",
            },
        },
        {
            "venue": "aster",
            "balance_usd": "74.840000123456789",
            "balance_components": {"total_wallet_balance_usd": "74.840000123456789"},
        },
        {
            "venue": "paradex",
            "balance_usd": "55.630000123456789",
            "balance_components": {"account_value_usd": "55.630000123456789"},
        },
    ]

    manifest = mod.manifest_from_rows(rows, "2026-04-28T18:00:00Z", "pre")
    path = tmp_path / "baseline.yaml"
    mod.write_yaml(path, manifest)

    loaded = mod.load_baseline_manifest(path)
    assert loaded["captured_at_utc"] == "2026-04-28T18:00:00Z"
    assert loaded["lighter_spot_included"] is True
    assert loaded["venues"]["hyperliquid"] == Decimal("81.920000123456789")
    assert loaded["venues"]["lighter_perp"] == Decimal("103.180000123456789")
    assert loaded["venues"]["lighter_spot"] == Decimal("10.950000000000001")
    assert loaded["venues"]["lighter_total"] == Decimal("114.130000123456790")


def test_post_snapshot_comparison_uses_exact_decimal_strings(tmp_path):
    mod = _load_module()
    pre_snapshot = tmp_path / "balance_pre_snapshot.json"
    pre_snapshot.write_text(
        """
{
  "total_balance_usd": "2.000000000000001",
  "rows": [
    {"venue": "hyperliquid", "balance_usd": "1.000000000000001"},
    {"venue": "extended", "balance_usd": "0.100000000000001"},
    {"venue": "lighter", "balance_usd": "0.200000000000001"},
    {"venue": "aster", "balance_usd": "0.300000000000001"},
    {"venue": "paradex", "balance_usd": "0.399999999999997"}
  ]
}
""",
        encoding="utf-8",
    )
    post_rows = [
        {"venue": "hyperliquid", "balance_usd": "1.000000000000002"},
        {"venue": "extended", "balance_usd": "0.100000000000002"},
        {"venue": "lighter", "balance_usd": "0.200000000000002"},
        {"venue": "aster", "balance_usd": "0.300000000000002"},
        {"venue": "paradex", "balance_usd": "0.399999999999998"},
    ]

    comparison = mod.compare_to_pre(pre_snapshot, post_rows, tmp_path / "balance_post_snapshot.json")

    assert comparison["total"]["pre_usd"] == "2.000000000000001"
    assert comparison["total"]["post_usd"] == "2.000000000000006"
    assert comparison["total"]["delta_usd"] == "0.000000000000005"
    assert comparison["total"]["abs_delta_usd"] == "0.000000000000005"
    assert comparison["total"]["abs_delta_usd_float"] == 0.000000000000005
    assert comparison["venue_count"] == 5
    assert comparison["per_venue"]["hyperliquid"]["delta_usd"] == "0.000000000000001"


class TestPhase5BalanceSnapshot(unittest.TestCase):
    def test_decimal_string_preserves_precision_without_float_quantization(self):
        test_decimal_string_preserves_precision_without_float_quantization()

    def test_collectors_extract_expected_fields_and_write_raw_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_collectors_extract_expected_fields_and_write_raw_details(Path(tmpdir))

    def test_builds_attribution_compatible_baseline_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_builds_attribution_compatible_baseline_manifest(Path(tmpdir))

    def test_post_snapshot_comparison_uses_exact_decimal_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_post_snapshot_comparison_uses_exact_decimal_strings(Path(tmpdir))
