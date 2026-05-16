#![cfg(feature = "live")]

use std::fs;

use paraphina::config::Phase51ForwardRefreshCaptureConfig;
use paraphina::live::phase51_forward_refresh_capture::{
    Phase51CaptureExecutionMode, Phase51CaptureTargetKey, Phase51ForwardRefreshCapture,
    Phase51LighterNativeLimitPressure, Phase51VenueNativeRole,
};
use paraphina::live::types::{
    Fill, Phase51ForwardRefreshLighterNativeLimit, Phase51ForwardRefreshNativeRole,
    Phase51ForwardRefreshTargetKey,
};
use paraphina::types::{OrderPurpose, Side};
use serde_json::Value;
use tempfile::tempdir;

fn enabled_config(path: &std::path::Path) -> Phase51ForwardRefreshCaptureConfig {
    Phase51ForwardRefreshCaptureConfig {
        enabled: true,
        output_path: path.display().to_string(),
        allow_live: false,
        append_only: true,
        max_rows: 5_000,
    }
}

fn read_rows(path: &std::path::Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect()
}

fn assert_safe_flags(row: &Value) {
    assert_eq!(row.get("no_live_flag").and_then(Value::as_bool), Some(true));
    for flag in [
        "approved_for_live",
        "approved_for_canary",
        "approved_for_model_training",
        "approved_for_capital_escalation",
        "admissible_for_financial_claim",
        "admissible_for_ev_admission",
        "live_orders_allowed",
        "capital_change_allowed",
        "risk_limit_relaxation_allowed",
    ] {
        assert_eq!(
            row.get(flag).and_then(Value::as_bool),
            Some(false),
            "{flag}"
        );
    }
}

fn runtime_fill() -> Fill {
    Fill {
        venue_index: 0,
        venue_id: "extended".to_string(),
        seq: 42,
        timestamp_ms: 1_700_000_000_000,
        order_id: Some("raw-order-id-runtime".to_string()),
        client_order_id: Some("raw-client-order-id-runtime".to_string()),
        fill_id: Some("raw-fill-id-runtime".to_string()),
        phase51_target_key: None,
        phase51_native_role: None,
        phase51_lighter_native_limit: None,
        side: Side::Buy,
        price: 100.0,
        size: 1.0,
        purpose: OrderPurpose::Mm,
        fee_bps: 0.0,
    }
}

#[test]
fn disabled_config_emits_no_file_or_rows() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let cfg = Phase51ForwardRefreshCaptureConfig {
        enabled: false,
        output_path: output.display().to_string(),
        allow_live: false,
        append_only: true,
        max_rows: 5_000,
    };
    let mut capture =
        Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
            .unwrap();
    let target = Phase51CaptureTargetKey::new("cg-disabled", "ok-disabled");

    let emitted = capture
        .capture_native_role(
            Some(&target),
            "extended",
            Some(Phase51VenueNativeRole::Extended { is_taker: true }),
        )
        .unwrap();

    assert!(emitted.is_none());
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
}

#[test]
fn fail_closed_for_allow_live_or_live_execution_mode() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");

    let mut cfg = enabled_config(&output);
    cfg.enabled = false;
    cfg.allow_live = true;
    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
        .unwrap_err();
    assert!(err.to_string().contains("allow_live=true"));

    let mut cfg = enabled_config(&output);
    cfg.allow_live = true;
    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
        .unwrap_err();
    assert!(err.to_string().contains("allow_live=true"));

    let mut cfg = enabled_config(&output);
    cfg.allow_live = false;
    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Live)
        .unwrap_err();
    assert!(err.to_string().contains("non-live/shadow-safe"));
}

#[test]
fn fail_closed_for_canary_and_unknown_execution_modes() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");

    for mode in [
        Phase51CaptureExecutionMode::Canary,
        Phase51CaptureExecutionMode::Unknown("production".to_string()),
    ] {
        let err =
            Phase51ForwardRefreshCapture::from_config(&enabled_config(&output), mode).unwrap_err();
        assert!(err.to_string().contains("non-live/shadow-safe"));
    }
}

#[test]
fn fail_closed_for_non_append_only_enabled_capture() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");

    let mut cfg = enabled_config(&output);
    cfg.append_only = false;
    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
        .unwrap_err();
    assert!(err.to_string().contains("append_only=true"));
}

#[test]
fn fail_closed_for_env_like_output_path() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("unsafe.env");

    let err = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap_err();
    assert!(err.to_string().contains(".env"));
}

#[test]
fn existing_output_over_max_rows_fails_closed() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    fs::write(&output, "{}\n{}\n").unwrap();

    let mut cfg = enabled_config(&output);
    cfg.max_rows = 1;
    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
        .unwrap_err();
    assert!(err.to_string().contains("already exceeds max_rows"));
}

#[test]
fn max_rows_limit_rejects_writes_at_limit() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut cfg = enabled_config(&output);
    cfg.max_rows = 1;
    let target = Phase51CaptureTargetKey::new("cg-max-rows", "ok-max-rows");
    let mut capture =
        Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Shadow)
            .unwrap();

    capture
        .capture_native_role(
            Some(&target),
            "hyperliquid",
            Some(Phase51VenueNativeRole::Hyperliquid { crossed: true }),
        )
        .unwrap()
        .unwrap();
    let err = capture
        .capture_native_role(
            Some(&target),
            "hyperliquid",
            Some(Phase51VenueNativeRole::Hyperliquid { crossed: false }),
        )
        .unwrap_err();

    assert!(err.to_string().contains("max_rows reached"));
    assert_eq!(read_rows(&output).len(), 1);
}

#[test]
fn no_exact_target_key_means_no_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();

    let emitted = capture
        .capture_native_role(
            None,
            "paradex",
            Some(Phase51VenueNativeRole::Paradex {
                liquidity: "MAKER".to_string(),
            }),
        )
        .unwrap();

    assert!(emitted.is_none());
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
}

#[test]
fn runtime_fill_without_phase51_fields_emits_no_rows_and_safe_audit() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let fill = runtime_fill();

    let audit = capture.capture_fill(&fill).unwrap();

    assert!(audit.enabled);
    assert!(!audit.sanitized_row_emitted);
    assert_eq!(audit.canonical_group_id, None);
    assert_eq!(audit.order_key, None);
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
    let audit_json = serde_json::to_string(&audit).unwrap();
    assert!(!audit_json.contains("raw-order-id-runtime"));
    assert!(!audit_json.contains("raw-client-order-id-runtime"));
    assert!(!audit_json.contains("raw-fill-id-runtime"));
}

#[test]
fn runtime_already_keyed_native_fill_emits_sanitized_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = runtime_fill();
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "cg-runtime-native".to_string(),
        order_key: "ok-runtime-native".to_string(),
    });
    fill.phase51_native_role = Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: true });

    let audit = capture.capture_fill(&fill).unwrap();

    assert!(audit.sanitized_row_emitted);
    assert_eq!(audit.target_type.as_deref(), Some("native_role"));
    assert_eq!(
        audit.native_role_source.as_deref(),
        Some("extended.isTaker")
    );
    assert_eq!(
        audit.canonical_group_id.as_deref(),
        Some("cg-runtime-native")
    );
    assert_eq!(audit.order_key.as_deref(), Some("ok-runtime-native"));

    let raw_output = fs::read_to_string(&output).unwrap();
    assert!(!raw_output.contains("raw-order-id-runtime"));
    assert!(!raw_output.contains("raw-client-order-id-runtime"));
    assert!(!raw_output.contains("raw-fill-id-runtime"));
    let rows = read_rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("target_type").and_then(Value::as_str),
        Some("native_role")
    );
    assert_eq!(rows[0].get("isTaker").and_then(Value::as_bool), Some(true));
    assert_safe_flags(&rows[0]);
}

#[test]
fn runtime_lighter_pressure_requires_complete_event_time_fields() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = runtime_fill();
    fill.venue_id = "lighter".to_string();
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "cg-runtime-lighter".to_string(),
        order_key: "ok-runtime-lighter".to_string(),
    });
    fill.phase51_lighter_native_limit = Some(Phase51ForwardRefreshLighterNativeLimit {
        active_order_headroom_account: Some(20),
        active_order_sendtx_utilization_account: None,
        rest_open_orders_count: Some(4),
        rest_open_orders_cap: Some(200),
        weighted_open_order_slots_used: None,
        weighted_open_order_slots_cap: None,
        native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
    });

    let audit = capture.capture_fill(&fill).unwrap();
    assert!(!audit.sanitized_row_emitted);
    assert!(!output.exists());

    fill.phase51_lighter_native_limit = Some(Phase51ForwardRefreshLighterNativeLimit {
        active_order_headroom_account: Some(20),
        active_order_sendtx_utilization_account: Some(7),
        rest_open_orders_count: Some(4),
        rest_open_orders_cap: Some(200),
        weighted_open_order_slots_used: None,
        weighted_open_order_slots_cap: None,
        native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
    });

    let audit = capture.capture_fill(&fill).unwrap();
    assert!(audit.sanitized_row_emitted);
    assert_eq!(audit.target_type.as_deref(), Some("lighter_native_limit"));
    let rows = read_rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("target_type").and_then(Value::as_str),
        Some("lighter_native_limit")
    );
    assert_safe_flags(&rows[0]);
}

#[test]
fn venue_native_role_examples_emit_sanitized_rows() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let target = Phase51CaptureTargetKey::with_internal_raw_identity(
        "cg-native-role",
        "ok-native-role",
        "raw-order-id-123 raw-client-order-id-456 trade-id-789",
    );

    let cases = [
        (
            "extended",
            Phase51VenueNativeRole::Extended { is_taker: false },
        ),
        (
            "paradex",
            Phase51VenueNativeRole::Paradex {
                liquidity: "TAKER".to_string(),
            },
        ),
        (
            "aster",
            Phase51VenueNativeRole::Aster {
                maker: true,
                last_filled_qty: "0.1".to_string(),
            },
        ),
        (
            "hyperliquid",
            Phase51VenueNativeRole::Hyperliquid { crossed: false },
        ),
        (
            "lighter",
            Phase51VenueNativeRole::Lighter {
                account_index: 7,
                is_maker_ask: true,
                ask_account_id: 11,
                bid_account_id: 13,
            },
        ),
    ];

    for (venue, role) in cases {
        capture
            .capture_native_role(Some(&target), venue, Some(role))
            .unwrap()
            .unwrap();
    }

    let raw_output = fs::read_to_string(&output).unwrap();
    assert!(!raw_output.contains("raw-order-id-123"));
    assert!(!raw_output.contains("raw-client-order-id-456"));
    assert!(!raw_output.contains("trade-id-789"));

    let rows = read_rows(&output);
    assert_eq!(rows.len(), 5);
    for row in rows {
        assert_eq!(
            row.get("target_type").and_then(Value::as_str),
            Some("native_role")
        );
        assert_eq!(
            row.get("canonical_group_id").and_then(Value::as_str),
            Some("cg-native-role")
        );
        assert_eq!(
            row.get("order_key").and_then(Value::as_str),
            Some("ok-native-role")
        );
        assert!(row.get("order_id").is_none());
        assert!(row.get("client_order_id").is_none());
        assert!(row.get("trade_id").is_none());
        assert_safe_flags(&row);
    }
}

#[test]
fn lighter_pressure_requires_complete_event_time_fields() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let target = Phase51CaptureTargetKey::new("cg-lighter", "ok-lighter");

    let incomplete = Phase51LighterNativeLimitPressure {
        active_order_headroom_account: Some(99),
        active_order_sendtx_utilization_account: None,
        rest_open_orders_count: Some(12),
        rest_open_orders_cap: Some(200),
        weighted_open_order_slots_used: None,
        weighted_open_order_slots_cap: None,
        native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
    };
    let emitted = capture
        .capture_lighter_native_limit(Some(&target), Some(incomplete))
        .unwrap();
    assert!(emitted.is_none());
    assert!(!output.exists());

    let complete = Phase51LighterNativeLimitPressure::complete_rest(88, 14, 12, 200);
    let emitted = capture
        .capture_lighter_native_limit(Some(&target), Some(complete))
        .unwrap()
        .unwrap();
    assert_eq!(
        emitted.get("target_type").and_then(Value::as_str),
        Some("lighter_native_limit")
    );
    assert_eq!(
        emitted
            .get("native_limit_event_time_status")
            .and_then(Value::as_str),
        Some("EVENT_TIME_ALIGNED")
    );
    assert_safe_flags(&emitted);

    let rows = read_rows(&output);
    assert_eq!(rows.len(), 1);
}
