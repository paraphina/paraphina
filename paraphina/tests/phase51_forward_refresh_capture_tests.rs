#![cfg(feature = "live")]

use std::fs;

use paraphina::config::Phase51ForwardRefreshCaptureConfig;
use paraphina::live::phase51_forward_refresh_capture::{
    Phase51CaptureExecutionMode, Phase51CaptureTargetKey, Phase51ForwardRefreshCapture,
    Phase51LighterNativeLimitPressure, Phase51LiveNativeRoleCanaryContext, Phase51VenueNativeRole,
};
use paraphina::live::types::{
    Fill, Phase51ForwardRefreshLighterNativeLimit, Phase51ForwardRefreshNativeRole,
    Phase51ForwardRefreshSourceOwnerFill, Phase51ForwardRefreshSourceOwnerPfillObservation,
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
        live_native_role_canary_approved: false,
        append_only: true,
        max_rows: 5_000,
    }
}

fn approved_live_native_role_config(path: &std::path::Path) -> Phase51ForwardRefreshCaptureConfig {
    Phase51ForwardRefreshCaptureConfig {
        enabled: true,
        output_path: path.display().to_string(),
        allow_live: false,
        live_native_role_canary_approved: true,
        append_only: true,
        max_rows: 1,
    }
}

fn lighter_strict_live_context() -> Phase51LiveNativeRoleCanaryContext {
    Phase51LiveNativeRoleCanaryContext {
        canary_enabled: true,
        native_role_strict_canary_enabled: true,
        native_role_one_sided_canary_enabled: true,
        venue_ids: vec!["lighter".to_string()],
        canary_max_open_orders: Some(1),
        canary_enforce_post_only: true,
        canary_enforce_reduce_only: false,
        strict_maker_only_observation_enabled: true,
        replacements_disabled: true,
        stop_after_first_row: true,
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

fn pfill_observation_path(path: &std::path::Path) -> std::path::PathBuf {
    let file_name = path.file_name().and_then(|name| name.to_str()).unwrap();
    let stem = file_name.strip_suffix(".jsonl").unwrap_or(file_name);
    path.with_file_name(format!("{stem}.source_owner_pfill_observation.jsonl"))
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
        live_native_role_canary_approved: false,
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
    cfg.live_native_role_canary_approved = true;
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
fn live_native_role_canary_requires_all_runtime_guards() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.future_native_role.jsonl");
    let cfg = approved_live_native_role_config(&output);

    let err = Phase51ForwardRefreshCapture::from_config(&cfg, Phase51CaptureExecutionMode::Live)
        .unwrap_err();
    assert!(err.to_string().contains("explicit runtime context"));

    let mut context = lighter_strict_live_context();
    context.canary_enabled = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("canary mode"));

    let mut context = lighter_strict_live_context();
    context.native_role_strict_canary_enabled = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("strict native-role canary"));

    let mut context = lighter_strict_live_context();
    context.native_role_one_sided_canary_enabled = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("one-sided native-role canary"));

    let mut context = lighter_strict_live_context();
    context.strict_maker_only_observation_enabled = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("strict maker-only"));

    let mut context = lighter_strict_live_context();
    context.venue_ids = vec!["lighter".to_string(), "extended".to_string()];
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("Lighter-only"));

    let mut context = lighter_strict_live_context();
    context.canary_enforce_post_only = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("post-only"));

    let mut context = lighter_strict_live_context();
    context.canary_enforce_reduce_only = true;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("non-reduce-only"));

    let mut context = lighter_strict_live_context();
    context.canary_max_open_orders = Some(2);
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("max_open_orders=1"));

    let mut context = lighter_strict_live_context();
    context.replacements_disabled = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("replacements disabled"));

    let mut context = lighter_strict_live_context();
    context.stop_after_first_row = false;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&context),
    )
    .unwrap_err();
    assert!(err.to_string().contains("stop-after-first-row"));

    let mut high_rows = cfg.clone();
    high_rows.max_rows = 2;
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &high_rows,
        Phase51CaptureExecutionMode::Live,
        Some(&lighter_strict_live_context()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("max_rows"));

    let default_remaining = dir.path().join("forward_refresh.remaining.jsonl");
    let cfg = approved_live_native_role_config(&default_remaining);
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&lighter_strict_live_context()),
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("must not write forward_refresh.remaining"));

    let output = dir
        .path()
        .join("forward_refresh.future_native_role_existing.jsonl");
    fs::write(&output, "{}\n").unwrap();
    let cfg = approved_live_native_role_config(&output);
    let err = Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
        &cfg,
        Phase51CaptureExecutionMode::Live,
        Some(&lighter_strict_live_context()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("absent or empty future output"));
}

#[test]
fn strict_lighter_native_role_canary_profile_enforces_one_open_order() {
    let profile = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("configs/phase51_lighter_native_role_strict_canary.toml");
    let raw = fs::read_to_string(profile).unwrap();
    assert!(raw.contains("base_order_size = 0.01"));
    assert!(raw.contains("max_order_size = 0.01"));
    assert!(raw.contains("max_open_orders = 1"));
    assert!(raw.contains("post_only = true"));
    assert!(raw.contains("reduce_only = false"));
    assert!(raw.contains("max_mid_jump_pct = 0.03"));
}

#[test]
fn lighter_baseline_cleanup_canary_profile_is_reduce_only_cleanup_shaped() {
    let profile = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("configs/phase51_lighter_baseline_cleanup_canary.toml");
    let raw = fs::read_to_string(profile).unwrap();
    assert!(raw.contains("base_order_size = 0.01"));
    assert!(raw.contains("max_order_size = 0.01"));
    assert!(raw.contains("max_position_tao = 0.0025"));
    assert!(raw.contains("max_gross_position_tao = 0.0025"));
    assert!(raw.contains("max_abs_venue_position_tao = 0.0025"));
    assert!(raw.contains("max_open_orders = 0"));
    assert!(raw.contains("post_only = false"));
    assert!(raw.contains("reduce_only = false"));
    assert!(raw.contains("max_mid_jump_pct = 0.03"));
}

#[test]
fn live_native_role_canary_emits_native_role_only_to_future_output() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.future_native_role.jsonl");
    let cfg = approved_live_native_role_config(&output);
    let mut capture =
        Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
            &cfg,
            Phase51CaptureExecutionMode::Live,
            Some(&lighter_strict_live_context()),
        )
        .unwrap();
    assert!(capture.live_native_role_canary_mode());

    let mut fill = runtime_fill();
    fill.venue_id = "lighter".to_string();
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "future-cg".to_string(),
        order_key: "future-ok".to_string(),
    });
    fill.phase51_native_role = Some(Phase51ForwardRefreshNativeRole::Lighter {
        account_index: 7,
        is_maker_ask: true,
        ask_account_id: 7,
        bid_account_id: 42,
    });
    fill.phase51_lighter_native_limit = Some(Phase51ForwardRefreshLighterNativeLimit {
        active_order_headroom_account: Some(10),
        active_order_sendtx_utilization_account: Some(1),
        rest_open_orders_count: Some(2),
        rest_open_orders_cap: Some(100),
        weighted_open_order_slots_used: None,
        weighted_open_order_slots_cap: None,
        native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
    });

    let audit = capture.capture_fill(&fill).unwrap();
    assert_eq!(audit.target_type.as_deref(), Some("native_role"));
    assert!(audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 1);

    let rows = read_rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("target_type").and_then(Value::as_str),
        Some("native_role")
    );
    assert_eq!(
        rows[0].get("venue_id").and_then(Value::as_str),
        Some("lighter")
    );
    assert!(rows[0].get("active_order_headroom_account").is_none());
    assert_safe_flags(&rows[0]);
}

#[test]
fn live_native_role_canary_skips_non_lighter_native_role_rows() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.future_native_role.jsonl");
    let cfg = approved_live_native_role_config(&output);
    let mut capture =
        Phase51ForwardRefreshCapture::from_config_with_live_native_role_canary_context(
            &cfg,
            Phase51CaptureExecutionMode::Live,
            Some(&lighter_strict_live_context()),
        )
        .unwrap();
    let target = Phase51CaptureTargetKey::new("future-cg", "future-ok");

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
fn source_owner_fill_with_target_and_native_role_emits_sanitized_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = Phase51ForwardRefreshSourceOwnerFill::new(
        2,
        "aster",
        99,
        1_700_000_000_100,
        Some("raw-order-id-source-owner".to_string()),
        Some("raw-client-order-id-source-owner".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Aster {
            maker: true,
            last_filled_qty: "0.01".to_string(),
        }),
    );
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "cg-source-owner".to_string(),
        order_key: "ok-source-owner".to_string(),
    });

    let audit = capture.capture_source_owner_fill(&fill).unwrap();

    assert!(audit.sanitized_row_emitted);
    assert_eq!(audit.target_type.as_deref(), Some("native_role"));
    assert_eq!(
        audit.native_role_source.as_deref(),
        Some("aster.ORDER_TRADE_UPDATE")
    );

    let raw_output = fs::read_to_string(&output).unwrap();
    assert!(!raw_output.contains("raw-order-id-source-owner"));
    assert!(!raw_output.contains("raw-client-order-id-source-owner"));
    let rows = read_rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("venue_id").and_then(Value::as_str),
        Some("aster")
    );
    assert_eq!(
        rows[0].get("e").and_then(Value::as_str),
        Some("ORDER_TRADE_UPDATE")
    );
    assert_eq!(rows[0].get("m").and_then(Value::as_bool), Some(true));
    assert_eq!(rows[0].get("l").and_then(Value::as_str), Some("0.01"));
    assert_safe_flags(&rows[0]);
}

#[test]
fn source_owner_fill_with_explicit_pfill_observation_emits_separate_sanitized_sidecar() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.future_native_role.jsonl");
    let pfill_output = pfill_observation_path(&output);
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = Phase51ForwardRefreshSourceOwnerFill::new(
        3,
        "lighter",
        1_700_000_000_101_000,
        1_700_000_000_101,
        Some("raw-order-id-source-owner".to_string()),
        Some("raw-client-order-id-source-owner".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Lighter {
            account_index: 7,
            is_maker_ask: true,
            ask_account_id: 7,
            bid_account_id: 9,
        }),
    );
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "cg-source-owner-pfill".to_string(),
        order_key: "ok-source-owner-pfill".to_string(),
    });
    fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Sell,
            2_100.5,
            0.01,
            1_700_000_000_101,
        )
        .unwrap(),
    );

    let audit = capture.capture_source_owner_fill(&fill).unwrap();

    assert!(audit.sanitized_row_emitted);
    assert_eq!(audit.target_type.as_deref(), Some("native_role"));
    assert_eq!(capture.rows_written(), 1);
    assert_eq!(capture.source_owner_pfill_observation_rows_written(), 1);

    let native_rows = read_rows(&output);
    assert_eq!(native_rows.len(), 1);
    assert_eq!(
        native_rows[0].get("target_type").and_then(Value::as_str),
        Some("native_role")
    );
    assert!(native_rows[0].get("p_fill_outcome").is_none());
    assert_safe_flags(&native_rows[0]);

    let pfill_rows = read_rows(&pfill_output);
    assert_eq!(pfill_rows.len(), 1);
    let row = &pfill_rows[0];
    assert_eq!(
        row.get("target_type").and_then(Value::as_str),
        Some("source_owner_pfill_observation")
    );
    assert_eq!(
        row.get("phase51_bridge_kind").and_then(Value::as_str),
        Some("source_owner_pfill_observation")
    );
    assert_eq!(
        row.get("canonical_group_id").and_then(Value::as_str),
        Some("cg-source-owner-pfill")
    );
    assert_eq!(
        row.get("order_key").and_then(Value::as_str),
        Some("ok-source-owner-pfill")
    );
    assert_eq!(row.get("side").and_then(Value::as_str), Some("Sell"));
    assert_eq!(row.get("price").and_then(Value::as_f64), Some(2_100.5));
    assert_eq!(row.get("size").and_then(Value::as_f64), Some(0.01));
    assert_eq!(
        row.get("event_time_ms").and_then(Value::as_i64),
        Some(1_700_000_000_101)
    );
    assert_eq!(
        row.get("outcome_status").and_then(Value::as_str),
        Some("OBSERVED_FILLED")
    );
    assert_eq!(row.get("p_fill_outcome").and_then(Value::as_f64), Some(1.0));
    assert_eq!(row.get("fill_count").and_then(Value::as_u64), Some(1));
    assert_eq!(
        row.get("native_role_source").and_then(Value::as_str),
        Some("lighter.account_index.is_maker_ask.ask_account_id.bid_account_id")
    );
    assert_eq!(
        row.get("source_link_inference_allowed")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        row.get("time_price_size_inference_allowed")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        row.get("role_inference_allowed").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        row.get("blocker_cleared").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        row.get("clears_phase51_blockers").and_then(Value::as_bool),
        Some(false)
    );
    assert!(row.get("active_order_headroom_account").is_none());
    assert_safe_flags(row);

    let raw_native = fs::read_to_string(&output).unwrap();
    let raw_pfill = fs::read_to_string(&pfill_output).unwrap();
    for raw in [&raw_native, &raw_pfill] {
        assert!(!raw.contains("raw-order-id-source-owner"));
        assert!(!raw.contains("raw-client-order-id-source-owner"));
    }
}

#[test]
fn source_owner_pfill_observation_requires_target_native_role_and_valid_observation() {
    let invalid_observations = [
        Phase51ForwardRefreshSourceOwnerPfillObservation {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side: Side::Buy,
            price: 0.0,
            size: 0.01,
            event_time_ms: 1_700_000_000_111,
            fill_count: 1,
            outcome_status: "OBSERVED_FILLED".to_string(),
            p_fill_outcome: 1.0,
        },
        Phase51ForwardRefreshSourceOwnerPfillObservation {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.0,
            event_time_ms: 1_700_000_000_111,
            fill_count: 1,
            outcome_status: "OBSERVED_FILLED".to_string(),
            p_fill_outcome: 1.0,
        },
        Phase51ForwardRefreshSourceOwnerPfillObservation {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.01,
            event_time_ms: 0,
            fill_count: 1,
            outcome_status: "OBSERVED_FILLED".to_string(),
            p_fill_outcome: 1.0,
        },
        Phase51ForwardRefreshSourceOwnerPfillObservation {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.01,
            event_time_ms: 1_700_000_000_111,
            fill_count: 0,
            outcome_status: "OBSERVED_FILLED".to_string(),
            p_fill_outcome: 1.0,
        },
        Phase51ForwardRefreshSourceOwnerPfillObservation {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.01,
            event_time_ms: 1_700_000_000_111,
            fill_count: 1,
            outcome_status: "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL".to_string(),
            p_fill_outcome: 0.0,
        },
    ];

    for observation in invalid_observations {
        let dir = tempdir().unwrap();
        let output = dir.path().join("forward_refresh.future_native_role.jsonl");
        let pfill_output = pfill_observation_path(&output);
        let mut capture = Phase51ForwardRefreshCapture::from_config(
            &enabled_config(&output),
            Phase51CaptureExecutionMode::Shadow,
        )
        .unwrap();
        let mut fill = Phase51ForwardRefreshSourceOwnerFill::new(
            3,
            "lighter",
            1,
            1_700_000_000_111,
            None,
            None,
            Some(Phase51ForwardRefreshNativeRole::Lighter {
                account_index: 7,
                is_maker_ask: false,
                ask_account_id: 9,
                bid_account_id: 7,
            }),
        );
        fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
            canonical_group_id: "cg-invalid-pfill".to_string(),
            order_key: "ok-invalid-pfill".to_string(),
        });
        fill.set_phase51_source_owner_pfill_observation(observation);

        let audit = capture.capture_source_owner_fill(&fill).unwrap();

        assert!(audit.sanitized_row_emitted);
        assert_eq!(capture.rows_written(), 1);
        assert_eq!(capture.source_owner_pfill_observation_rows_written(), 0);
        assert!(!pfill_output.exists());
    }

    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.future_native_role.jsonl");
    let pfill_output = pfill_observation_path(&output);
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = Phase51ForwardRefreshSourceOwnerFill::new(
        3,
        "lighter",
        2,
        1_700_000_000_222,
        None,
        None,
        None,
    );
    fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
        canonical_group_id: "cg-missing-native".to_string(),
        order_key: "ok-missing-native".to_string(),
    });
    fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Buy,
            2_100.0,
            0.01,
            1_700_000_000_222,
        )
        .unwrap(),
    );

    let audit = capture.capture_source_owner_fill(&fill).unwrap();

    assert!(!audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 0);
    assert_eq!(capture.source_owner_pfill_observation_rows_written(), 0);
    assert!(!output.exists());
    assert!(!pfill_output.exists());
}

#[test]
fn source_owner_fill_invalid_aster_last_qty_emits_no_row() {
    for invalid_qty in ["", "0", "-0.01", "not-a-number"] {
        let dir = tempdir().unwrap();
        let output = dir.path().join("forward_refresh.remaining.jsonl");
        let mut capture = Phase51ForwardRefreshCapture::from_config(
            &enabled_config(&output),
            Phase51CaptureExecutionMode::Shadow,
        )
        .unwrap();
        let mut fill = Phase51ForwardRefreshSourceOwnerFill::new(
            2,
            "aster",
            101,
            1_700_000_000_300,
            Some("raw-order-id-source-owner".to_string()),
            Some("raw-client-order-id-source-owner".to_string()),
            Some(Phase51ForwardRefreshNativeRole::Aster {
                maker: true,
                last_filled_qty: invalid_qty.to_string(),
            }),
        );
        fill.phase51_target_key = Some(Phase51ForwardRefreshTargetKey {
            canonical_group_id: "cg-source-owner".to_string(),
            order_key: "ok-source-owner".to_string(),
        });

        let audit = capture.capture_source_owner_fill(&fill).unwrap();

        assert!(audit.enabled);
        assert!(!audit.sanitized_row_emitted);
        assert_eq!(capture.rows_written(), 0);
        assert!(!output.exists());
    }
}

#[test]
fn source_owner_fill_without_target_key_emits_no_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let fill = Phase51ForwardRefreshSourceOwnerFill::new(
        2,
        "aster",
        100,
        1_700_000_000_200,
        Some("raw-order-id-source-owner".to_string()),
        Some("raw-client-order-id-source-owner".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Aster {
            maker: false,
            last_filled_qty: "0.02".to_string(),
        }),
    );

    let audit = capture.capture_source_owner_fill(&fill).unwrap();

    assert!(audit.enabled);
    assert!(!audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
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
