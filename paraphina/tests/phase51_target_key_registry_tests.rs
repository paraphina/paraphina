#![cfg(feature = "live")]

use std::fs;

use paraphina::actions::ActionIdGenerator;
use paraphina::config::Config;
use paraphina::config::Phase51ForwardRefreshCaptureConfig;
use paraphina::live::phase51_forward_refresh_capture::{
    Phase51CaptureExecutionMode, Phase51ForwardRefreshCapture,
};
use paraphina::live::phase51_target_key_registry::{
    Phase51TargetKeyRegistry, Phase51TargetKeyRegistryStage,
};
use paraphina::live::types::{
    ExecutionEvent, Fill, OrderAccepted, Phase51ForwardRefreshNativeRole,
    Phase51ForwardRefreshSourceOwnerFill, Phase51ForwardRefreshSourceOwnerPfillObservation,
};
use paraphina::mm::{compute_mm_quotes_with_now_and_identity_authority, MmQuoteIdentityAllocator};
use paraphina::order_management::plan_mm_order_actions;
use paraphina::state::GlobalState;
use paraphina::types::{
    OrderIntent, OrderPurpose, Phase51ForwardRefreshTargetKey, PlaceOrderIntent,
    ReplaceOrderIntent, Side, TimeInForce, VenueStatus,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::Value;
use tempfile::tempdir;

fn target_key(label: &str) -> Phase51ForwardRefreshTargetKey {
    Phase51ForwardRefreshTargetKey {
        canonical_group_id: format!("canonical-{label}"),
        order_key: format!("order-{label}"),
    }
}

fn place_intent(
    client_order_id: Option<&str>,
    phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
) -> OrderIntent {
    OrderIntent::Place(PlaceOrderIntent {
        venue_index: 0,
        venue_id: "extended".into(),
        side: Side::Buy,
        price: 100.0,
        size: 1.0,
        purpose: OrderPurpose::Mm,
        time_in_force: TimeInForce::Gtc,
        post_only: true,
        reduce_only: false,
        client_order_id: client_order_id.map(str::to_string),
        phase51_target_key,
    })
}

fn replace_intent(
    client_order_id: Option<&str>,
    phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
) -> OrderIntent {
    OrderIntent::Replace(ReplaceOrderIntent {
        venue_index: 0,
        venue_id: "extended".into(),
        side: Side::Sell,
        price: 101.0,
        size: 1.0,
        purpose: OrderPurpose::Mm,
        time_in_force: TimeInForce::Gtc,
        post_only: true,
        reduce_only: false,
        order_id: "previous-handle".to_string(),
        client_order_id: client_order_id.map(str::to_string),
        phase51_target_key,
    })
}

fn accepted(client_order_id: Option<&str>, order_id: &str) -> OrderAccepted {
    OrderAccepted {
        venue_index: 0,
        venue_id: "extended".to_string(),
        seq: 1,
        timestamp_ms: 1_700_000_000_000,
        order_id: order_id.to_string(),
        client_order_id: client_order_id.map(str::to_string),
        side: Side::Buy,
        price: 100.0,
        size: 1.0,
        purpose: OrderPurpose::Mm,
    }
}

fn fill(client_order_id: Option<&str>, order_id: Option<&str>) -> Fill {
    Fill {
        venue_index: 0,
        venue_id: "extended".to_string(),
        seq: 2,
        timestamp_ms: 1_700_000_000_001,
        order_id: order_id.map(str::to_string),
        client_order_id: client_order_id.map(str::to_string),
        fill_id: Some("fill-handle".to_string()),
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

fn source_owner_fill(
    client_order_id: Option<&str>,
    order_id: Option<&str>,
) -> Phase51ForwardRefreshSourceOwnerFill {
    Phase51ForwardRefreshSourceOwnerFill::new(
        0,
        "aster",
        3,
        1_700_000_000_002,
        order_id.map(str::to_string),
        client_order_id.map(str::to_string),
        Some(Phase51ForwardRefreshNativeRole::Aster {
            maker: true,
            last_filled_qty: "0.01".to_string(),
        }),
    )
}

fn enabled_capture_config(path: &std::path::Path) -> Phase51ForwardRefreshCaptureConfig {
    Phase51ForwardRefreshCaptureConfig {
        enabled: true,
        output_path: path.display().to_string(),
        allow_live: false,
        live_native_role_canary_approved: false,
        append_only: true,
        max_rows: 5_000,
    }
}

fn keyed_mm_place_intent() -> OrderIntent {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);
    let now_ms = 10_000;
    state.fair_value = Some(300.0);
    state.fair_value_prev = 300.0;
    state.fv_available = true;
    state.sigma_eff = 0.02;
    state.spread_mult = 1.0;
    state.size_mult = 1.0;
    state.vol_ratio_clipped = 1.0;
    state.delta_limit_usd = 100_000.0;
    for venue in &mut state.venues {
        venue.mid = Some(300.0);
        venue.spread = Some(1.0);
        venue.last_mid_update_ms = Some(now_ms - 10);
        venue.depth_near_mid = 10_000.0;
        venue.margin_available_usd = 10_000.0;
        venue.dist_liq_sigma = 10.0;
        venue.status = VenueStatus::Healthy;
        venue.toxicity = 0.0;
    }
    let mut allocator = MmQuoteIdentityAllocator::with_rng(ChaCha8Rng::seed_from_u64(121));
    let authority = allocator.populated_authority(cfg.venues.len());
    let quotes =
        compute_mm_quotes_with_now_and_identity_authority(&cfg, &state, Some(now_ms), &authority);
    let mut gen = ActionIdGenerator::new(0);
    let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);

    plan.intents
        .into_iter()
        .find(|intent| matches!(intent, OrderIntent::Place(place) if place.phase51_target_key.is_some()))
        .expect("keyed MM place intent")
}

#[test]
fn explicit_place_target_key_registers() {
    let key = target_key("place");
    let mut registry = Phase51TargetKeyRegistry::default();

    assert!(registry.register_intent(&place_intent(Some("client-place"), Some(key.clone()))));

    let resolved = registry.resolve_fill(&fill(Some("client-place"), None));
    assert_eq!(resolved, Some(key));
    assert_eq!(registry.counts().client_bindings, 1);
}

#[test]
fn explicit_replace_target_key_registers() {
    let key = target_key("replace");
    let mut registry = Phase51TargetKeyRegistry::default();

    assert!(registry.register_intent(&replace_intent(Some("client-replace"), Some(key.clone()))));

    let resolved = registry.resolve_fill(&fill(Some("client-replace"), None));
    assert_eq!(resolved, Some(key));
    assert_eq!(registry.counts().client_bindings, 1);
}

#[test]
fn no_explicit_target_key_means_no_registry_entry() {
    let mut registry = Phase51TargetKeyRegistry::default();

    assert!(!registry.register_intent(&place_intent(Some("client-without-key"), None)));

    assert_eq!(
        registry.resolve_fill(&fill(Some("client-without-key"), None)),
        None
    );
    assert_eq!(registry.counts().client_bindings, 0);
}

#[test]
fn dropped_stage_after_enqueue_failure_leaves_no_registry_entry() {
    let registry = Phase51TargetKeyRegistry::default();
    let stage = Phase51TargetKeyRegistryStage::from_intents(&[place_intent(
        Some("stage-fail-client"),
        Some(target_key("stage-fail")),
    )]);
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    tx.try_send(()).expect("pre-fill channel");

    assert!(tx.try_send(()).is_err());
    drop(stage);

    assert_eq!(
        registry.resolve_fill(&fill(Some("stage-fail-client"), None)),
        None
    );
    assert_eq!(registry.counts().client_bindings, 0);
}

#[test]
fn successful_enqueue_commits_staged_registry_entry() {
    let key = target_key("stage-success");
    let mut registry = Phase51TargetKeyRegistry::default();
    let stage = Phase51TargetKeyRegistryStage::from_intents(&[place_intent(
        Some("stage-success-client"),
        Some(key.clone()),
    )]);
    let (tx, _rx) = tokio::sync::mpsc::channel(1);

    assert!(tx.try_send(()).is_ok());
    registry.commit_stage(stage);

    assert_eq!(
        registry.resolve_fill(&fill(Some("stage-success-client"), None)),
        Some(key)
    );
    assert_eq!(registry.counts().client_bindings, 1);
}

#[test]
fn generated_looking_client_ids_do_not_create_target_keys() {
    let mut registry = Phase51TargetKeyRegistry::default();

    assert!(!registry.register_intent(&place_intent(Some("co_42_v3_mm_0"), None)));

    assert_eq!(
        registry.resolve_fill(&fill(Some("co_42_v3_mm_0"), None)),
        None
    );
    assert_eq!(registry.counts().client_bindings, 0);
}

#[test]
fn order_accepted_binds_order_handle_only_after_client_binding_exists() {
    let key = target_key("accepted");
    let mut registry = Phase51TargetKeyRegistry::default();

    assert!(
        !registry.observe_order_accepted(&accepted(Some("client-accepted"), "order-before-client"))
    );
    assert_eq!(
        registry.resolve_fill(&fill(None, Some("order-before-client"))),
        None
    );

    assert!(registry.register_intent(&place_intent(Some("client-accepted"), Some(key.clone()))));
    assert!(
        registry.observe_order_accepted(&accepted(Some("client-accepted"), "order-after-client"))
    );

    assert_eq!(
        registry.resolve_fill(&fill(None, Some("order-after-client"))),
        Some(key)
    );
}

#[test]
fn fill_lookup_resolves_only_exact_in_memory_bindings() {
    let key = target_key("exact");
    let mut registry = Phase51TargetKeyRegistry::default();
    assert!(registry.register_intent(&place_intent(Some("client-exact"), Some(key.clone()))));
    assert!(registry.observe_order_accepted(&accepted(Some("client-exact"), "order-exact")));

    assert_eq!(
        registry.resolve_fill(&fill(Some("client-exact-suffix"), Some("order-exact"))),
        Some(key.clone())
    );
    assert_eq!(
        registry.resolve_fill(&fill(
            Some("client-exact-suffix"),
            Some("order-exact-suffix")
        )),
        None
    );
    assert_eq!(
        registry.resolve_fill(&fill(Some("client-exact"), Some("order-exact"))),
        Some(key)
    );
}

#[test]
fn unresolved_fills_leave_phase51_target_key_none() {
    let registry = Phase51TargetKeyRegistry::default();
    let mut fill = fill(Some("unresolved-client"), Some("unresolved-order"));

    assert!(!registry.enrich_fill(&mut fill));

    assert_eq!(fill.phase51_target_key, None);
}

#[test]
fn conflicting_exact_handles_leave_phase51_target_key_none() {
    let client_key = target_key("client-conflict");
    let order_key = target_key("order-conflict");
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.register_intent(&place_intent(
        Some("client-conflict"),
        Some(client_key.clone()),
    ));
    registry.register_intent(&place_intent(
        Some("client-for-order-conflict"),
        Some(order_key),
    ));
    registry.observe_order_accepted(&accepted(
        Some("client-for-order-conflict"),
        "order-conflict",
    ));
    let mut ambiguous_fill = fill(Some("client-conflict"), Some("order-conflict"));

    assert!(!registry.enrich_fill(&mut ambiguous_fill));

    assert_eq!(ambiguous_fill.phase51_target_key, None);
    assert_eq!(
        registry.resolve_fill(&fill(Some("client-conflict"), None)),
        Some(client_key)
    );
}

#[test]
fn execution_event_fill_is_enriched_only_after_exact_binding() {
    let key = target_key("event");
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.register_intent(&place_intent(Some("client-event"), Some(key.clone())));
    registry.observe_order_accepted(&accepted(Some("client-event"), "order-event"));

    let mut event = ExecutionEvent::Filled(fill(None, Some("order-event")));

    assert!(registry.observe_execution_event(&mut event));
    match event {
        ExecutionEvent::Filled(fill) => assert_eq!(fill.phase51_target_key, Some(key)),
        _ => panic!("expected fill event"),
    }
}

#[test]
fn registry_safe_counts_do_not_emit_raw_handles() {
    let key = target_key("counts");
    let mut registry = Phase51TargetKeyRegistry::new(4);
    registry.register_intent(&place_intent(Some("client-counts"), Some(key)));
    registry.observe_order_accepted(&accepted(Some("client-counts"), "order-counts"));

    let rendered = format!("{:?}", registry.counts());

    assert!(!rendered.contains("client-counts"));
    assert!(!rendered.contains("order-counts"));
    assert_eq!(registry.counts().client_bindings, 1);
    assert_eq!(registry.counts().order_bindings, 1);
}

#[test]
fn target_key_only_fill_emits_no_forward_refresh_row_without_native_fields() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_capture_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut fill = fill(Some("client-target-only"), Some("order-target-only"));
    fill.phase51_target_key = Some(target_key("target-only"));

    let audit = capture.capture_fill(&fill).unwrap();

    assert!(audit.enabled);
    assert!(!audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
}

#[test]
fn allocator_keyed_mm_intent_registers_enriches_and_emits_sanitized_native_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_capture_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let intent = keyed_mm_place_intent();
    let (client_order_id, target_key) = match &intent {
        OrderIntent::Place(place) => (
            place.client_order_id.clone().expect("client order handle"),
            place.phase51_target_key.clone().expect("target key"),
        ),
        _ => panic!("expected keyed place"),
    };
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents(&[intent]));

    assert_eq!(registry.counts().client_bindings, 1);

    let mut native_fill = fill(Some(&client_order_id), None);
    native_fill.phase51_native_role =
        Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: false });

    assert!(registry.enrich_fill(&mut native_fill));
    assert_eq!(native_fill.phase51_target_key, Some(target_key.clone()));

    let audit = capture.capture_fill(&native_fill).unwrap();

    assert!(audit.enabled);
    assert!(audit.sanitized_row_emitted);
    assert_eq!(
        audit.canonical_group_id.as_deref(),
        Some(target_key.canonical_group_id.as_str())
    );
    assert_eq!(
        audit.order_key.as_deref(),
        Some(target_key.order_key.as_str())
    );
    assert_eq!(capture.rows_written(), 1);

    let raw_output = fs::read_to_string(&output).unwrap();
    assert!(!raw_output.contains(&client_order_id));
    assert!(!raw_output.contains("fill-handle"));
}

#[test]
fn source_owner_fill_is_enriched_and_emits_sanitized_native_row() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_capture_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let intent = keyed_mm_place_intent();
    let (client_order_id, target_key) = match &intent {
        OrderIntent::Place(place) => (
            place.client_order_id.clone().expect("client order handle"),
            place.phase51_target_key.clone().expect("target key"),
        ),
        _ => panic!("expected keyed place"),
    };
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents(&[intent]));

    let mut native_fill = source_owner_fill(Some(&client_order_id), None);
    assert!(registry.enrich_source_owner_fill(&mut native_fill));
    assert_eq!(native_fill.phase51_target_key, Some(target_key.clone()));

    let audit = capture.capture_source_owner_fill(&native_fill).unwrap();

    assert!(audit.enabled);
    assert!(audit.sanitized_row_emitted);
    assert_eq!(
        audit.canonical_group_id.as_deref(),
        Some(target_key.canonical_group_id.as_str())
    );
    assert_eq!(
        audit.order_key.as_deref(),
        Some(target_key.order_key.as_str())
    );
    assert_eq!(capture.rows_written(), 1);

    let raw_output = fs::read_to_string(&output).unwrap();
    assert!(!raw_output.contains(&client_order_id));
    assert!(!raw_output.contains("source-owner-order"));
    let rows: Vec<Value> = raw_output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("target_type").and_then(Value::as_str),
        Some("native_role")
    );
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
}

#[test]
fn allocator_keyed_mm_native_fill_emits_no_row_when_capture_disabled() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut disabled_cfg = enabled_capture_config(&output);
    disabled_cfg.enabled = false;
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &disabled_cfg,
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let intent = keyed_mm_place_intent();
    let (client_order_id, target_key) = match &intent {
        OrderIntent::Place(place) => (
            place.client_order_id.clone().expect("client order handle"),
            place.phase51_target_key.clone().expect("target key"),
        ),
        _ => panic!("expected keyed place"),
    };
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents(&[intent]));

    let mut native_fill = fill(Some(&client_order_id), None);
    native_fill.phase51_native_role =
        Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: false });

    assert!(registry.enrich_fill(&mut native_fill));
    assert_eq!(native_fill.phase51_target_key, Some(target_key));

    let audit = capture.capture_fill(&native_fill).unwrap();

    assert!(!audit.enabled);
    assert!(!audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
}

#[test]
fn mm_intents_without_target_keys_do_not_register_or_emit_forward_refresh_rows() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("forward_refresh.remaining.jsonl");
    let mut capture = Phase51ForwardRefreshCapture::from_config(
        &enabled_capture_config(&output),
        Phase51CaptureExecutionMode::Shadow,
    )
    .unwrap();
    let mut registry = Phase51TargetKeyRegistry::default();
    let intents = vec![
        place_intent(Some("generated-client-place"), None),
        replace_intent(Some("generated-client-replace"), None),
    ];
    let stage = Phase51TargetKeyRegistryStage::from_intents(&intents);

    registry.commit_stage(stage);
    assert_eq!(registry.counts().client_bindings, 0);
    assert_eq!(registry.counts().order_bindings, 0);
    assert!(!registry.observe_order_accepted(&accepted(
        Some("generated-client-place"),
        "generated-order-place",
    )));

    let mut native_fill = fill(
        Some("generated-client-place"),
        Some("generated-order-place"),
    );
    native_fill.phase51_native_role =
        Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: true });

    assert!(!registry.enrich_fill(&mut native_fill));
    assert!(native_fill.phase51_target_key.is_none());

    let audit = capture.capture_fill(&native_fill).unwrap();

    assert!(audit.enabled);
    assert!(!audit.sanitized_row_emitted);
    assert_eq!(capture.rows_written(), 0);
    assert!(!output.exists());
}

#[test]
fn exact_runtime_source_ticks_bridge_to_source_owner_pfill_observation() {
    let key = target_key("source-tick-bridge");
    let intent = place_intent(Some("source-tick-client"), Some(key.clone()));
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents_at_source_tick(
        &[intent],
        11,
    ));
    assert!(registry
        .observe_order_accepted(&accepted(Some("source-tick-client"), "source-tick-order",)));

    let mut source_owner_fill = Phase51ForwardRefreshSourceOwnerFill::new(
        0,
        "lighter",
        3,
        1_700_000_000_002,
        Some("source-tick-order".to_string()),
        None,
        Some(Phase51ForwardRefreshNativeRole::Lighter {
            account_index: 7,
            is_maker_ask: true,
            ask_account_id: 7,
            bid_account_id: 9,
        }),
    );
    source_owner_fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Sell,
            2_100.0,
            0.01,
            1_700_000_000_002,
        )
        .unwrap(),
    );
    let mut events = vec![ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
        source_owner_fill,
    )];

    assert_eq!(
        registry.observe_execution_events_at_source_tick(&mut events, 17),
        1
    );

    let ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(enriched_fill) = &events[0] else {
        panic!("expected source-owner fill")
    };
    assert_eq!(enriched_fill.phase51_target_key, Some(key));
    let observation = enriched_fill
        .phase51_source_owner_pfill_observation
        .as_ref()
        .expect("pfill observation");
    assert_eq!(observation.order_source_tick, Some(11));
    assert_eq!(observation.fill_source_tick, Some(17));
    assert_eq!(observation.observed_horizon_source_ticks(), Some(6));
}

#[test]
fn source_owner_pfill_horizon_fails_closed_without_explicit_order_source_tick() {
    let key = target_key("source-tick-missing-order");
    let intent = place_intent(Some("missing-order-tick-client"), Some(key.clone()));
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents(&[intent]));

    let mut source_owner_fill = Phase51ForwardRefreshSourceOwnerFill::new(
        0,
        "lighter",
        3,
        1_700_000_000_002,
        None,
        Some("missing-order-tick-client".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Lighter {
            account_index: 7,
            is_maker_ask: true,
            ask_account_id: 7,
            bid_account_id: 9,
        }),
    );
    source_owner_fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Sell,
            2_100.0,
            0.01,
            1_700_000_000_002,
        )
        .unwrap(),
    );
    let mut events = vec![ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
        source_owner_fill,
    )];

    assert_eq!(
        registry.observe_execution_events_at_source_tick(&mut events, 17),
        1
    );

    let ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(enriched_fill) = &events[0] else {
        panic!("expected source-owner fill")
    };
    assert_eq!(enriched_fill.phase51_target_key, Some(key));
    let observation = enriched_fill
        .phase51_source_owner_pfill_observation
        .as_ref()
        .expect("pfill observation");
    assert_eq!(observation.order_source_tick, None);
    assert_eq!(observation.fill_source_tick, None);
    assert_eq!(observation.observed_horizon_source_ticks(), None);
}

#[test]
fn source_owner_pfill_horizon_fails_closed_without_explicit_fill_source_tick() {
    let key = target_key("source-tick-missing-fill");
    let intent = place_intent(Some("missing-fill-tick-client"), Some(key.clone()));
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents_at_source_tick(
        &[intent],
        11,
    ));

    let mut source_owner_fill = Phase51ForwardRefreshSourceOwnerFill::new(
        0,
        "lighter",
        3,
        1_700_000_000_002,
        None,
        Some("missing-fill-tick-client".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Lighter {
            account_index: 7,
            is_maker_ask: true,
            ask_account_id: 7,
            bid_account_id: 9,
        }),
    );
    source_owner_fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Sell,
            2_100.0,
            0.01,
            1_700_000_000_002,
        )
        .unwrap(),
    );
    let mut events = vec![ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
        source_owner_fill,
    )];

    assert_eq!(registry.observe_execution_events(&mut events), 1);

    let ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(enriched_fill) = &events[0] else {
        panic!("expected source-owner fill")
    };
    assert_eq!(enriched_fill.phase51_target_key, Some(key));
    let observation = enriched_fill
        .phase51_source_owner_pfill_observation
        .as_ref()
        .expect("pfill observation");
    assert_eq!(observation.order_source_tick, None);
    assert_eq!(observation.fill_source_tick, None);
    assert_eq!(observation.observed_horizon_source_ticks(), None);
}

#[test]
fn source_owner_pfill_horizon_fails_closed_when_fill_source_tick_precedes_order_tick() {
    let key = target_key("source-tick-negative");
    let intent = place_intent(Some("negative-tick-client"), Some(key.clone()));
    let mut registry = Phase51TargetKeyRegistry::default();
    registry.commit_stage(Phase51TargetKeyRegistryStage::from_intents_at_source_tick(
        &[intent],
        21,
    ));

    let mut source_owner_fill = Phase51ForwardRefreshSourceOwnerFill::new(
        0,
        "lighter",
        3,
        1_700_000_000_002,
        None,
        Some("negative-tick-client".to_string()),
        Some(Phase51ForwardRefreshNativeRole::Lighter {
            account_index: 7,
            is_maker_ask: true,
            ask_account_id: 7,
            bid_account_id: 9,
        }),
    );
    source_owner_fill.set_phase51_source_owner_pfill_observation(
        Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
            Side::Sell,
            2_100.0,
            0.01,
            1_700_000_000_002,
        )
        .unwrap(),
    );
    let mut events = vec![ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
        source_owner_fill,
    )];

    assert_eq!(
        registry.observe_execution_events_at_source_tick(&mut events, 17),
        1
    );

    let ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(enriched_fill) = &events[0] else {
        panic!("expected source-owner fill")
    };
    assert_eq!(enriched_fill.phase51_target_key, Some(key));
    let observation = enriched_fill
        .phase51_source_owner_pfill_observation
        .as_ref()
        .expect("pfill observation");
    assert_eq!(observation.order_source_tick, None);
    assert_eq!(observation.fill_source_tick, None);
    assert_eq!(observation.observed_horizon_source_ticks(), None);
}
