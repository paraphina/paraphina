#![cfg(feature = "live")]

use paraphina::config::Phase51ForwardRefreshCaptureConfig;
use paraphina::live::phase51_forward_refresh_capture::{
    Phase51CaptureExecutionMode, Phase51ForwardRefreshCapture,
};
use paraphina::live::phase51_target_key_registry::Phase51TargetKeyRegistry;
use paraphina::live::types::{ExecutionEvent, Fill, OrderAccepted};
use paraphina::types::{
    OrderIntent, OrderPurpose, Phase51ForwardRefreshTargetKey, PlaceOrderIntent,
    ReplaceOrderIntent, Side, TimeInForce,
};
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

fn enabled_capture_config(path: &std::path::Path) -> Phase51ForwardRefreshCaptureConfig {
    Phase51ForwardRefreshCaptureConfig {
        enabled: true,
        output_path: path.display().to_string(),
        allow_live: false,
        append_only: true,
        max_rows: 5_000,
    }
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
