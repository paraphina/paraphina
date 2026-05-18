use serde_json::{json, Map, Value};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};

use crate::config::Phase51ForwardRefreshCaptureConfig;

use super::types::{
    Fill, Phase51ForwardRefreshCaptureAudit, Phase51ForwardRefreshLighterNativeLimit,
    Phase51ForwardRefreshNativeRole, Phase51ForwardRefreshSourceOwnerFill,
    Phase51ForwardRefreshSourceOwnerPfillObservation, Phase51ForwardRefreshTargetKey,
};

const FORBIDDEN_FIELD_NAMES: &[&str] = &[
    "order_id",
    "orderId",
    "raw_order_id",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "raw_client_order_id",
    "trade_id",
    "tradeId",
    "fill_id",
    "fillId",
    "tx_hash",
    "txHash",
    "hash",
    "api_key",
    "apiKey",
    "jwt",
    "token",
    "signed_payload",
    "signature",
    "secret",
];

const FORBIDDEN_KEY_FRAGMENTS: &[&str] = &[
    "secret",
    "api_key",
    "apikey",
    "jwt",
    "token",
    "signed_payload",
    "signature",
    ".env",
];

const UNSAFE_TRUE_FLAGS: &[&str] = &[
    "approved_for_live",
    "approved_for_canary",
    "approved_for_model_training",
    "approved_for_capital_escalation",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase51CaptureError {
    message: String,
}

impl Phase51CaptureError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn from_message(message: impl Into<String>) -> Self {
        Self::new(message)
    }
}

impl fmt::Display for Phase51CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for Phase51CaptureError {}

pub type Phase51CaptureResult<T> = Result<T, Phase51CaptureError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase51CaptureExecutionMode {
    Shadow,
    Paper,
    Testnet,
    Replay,
    Live,
    Canary,
    Unknown(String),
}

impl Phase51CaptureExecutionMode {
    pub fn from_trade_mode_text(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "shadow" | "s" => Self::Shadow,
            "paper" | "p" => Self::Paper,
            "testnet" | "tn" | "t" => Self::Testnet,
            "replay" | "offline" => Self::Replay,
            "live" | "l" => Self::Live,
            "canary" | "c" => Self::Canary,
            other => Self::Unknown(other.to_string()),
        }
    }

    fn is_non_live_shadow_safe(&self) -> bool {
        matches!(
            self,
            Self::Shadow | Self::Paper | Self::Testnet | Self::Replay
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase51LiveNativeRoleCanaryContext {
    pub canary_enabled: bool,
    pub native_role_strict_canary_enabled: bool,
    pub native_role_one_sided_canary_enabled: bool,
    pub venue_ids: Vec<String>,
    pub canary_max_open_orders: Option<usize>,
    pub canary_enforce_post_only: bool,
    pub canary_enforce_reduce_only: bool,
    pub strict_maker_only_observation_enabled: bool,
    pub replacements_disabled: bool,
    pub stop_after_first_row: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase51CaptureTargetKey {
    pub canonical_group_id: String,
    pub order_key: String,
    pub raw_identity_debug: Option<String>,
}

impl Phase51CaptureTargetKey {
    pub fn new(canonical_group_id: impl Into<String>, order_key: impl Into<String>) -> Self {
        Self {
            canonical_group_id: canonical_group_id.into(),
            order_key: order_key.into(),
            raw_identity_debug: None,
        }
    }

    pub fn with_internal_raw_identity(
        canonical_group_id: impl Into<String>,
        order_key: impl Into<String>,
        raw_identity_debug: impl Into<String>,
    ) -> Self {
        Self {
            canonical_group_id: canonical_group_id.into(),
            order_key: order_key.into(),
            raw_identity_debug: Some(raw_identity_debug.into()),
        }
    }

    fn is_complete(&self) -> bool {
        !self.canonical_group_id.trim().is_empty() && !self.order_key.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase51VenueNativeRole {
    Aster {
        maker: bool,
        last_filled_qty: String,
    },
    Extended {
        is_taker: bool,
    },
    Hyperliquid {
        crossed: bool,
    },
    Lighter {
        account_index: i64,
        is_maker_ask: bool,
        ask_account_id: i64,
        bid_account_id: i64,
    },
    Paradex {
        liquidity: String,
    },
}

impl Phase51VenueNativeRole {
    fn to_payload(&self, venue_id: &str) -> Phase51CaptureResult<Option<Map<String, Value>>> {
        let venue = canonical_venue_id(venue_id);
        let mut payload = Map::new();
        match self {
            Self::Aster {
                maker,
                last_filled_qty,
            } => {
                if venue != "aster" {
                    return Ok(None);
                }
                let qty_ok = last_filled_qty
                    .trim()
                    .parse::<f64>()
                    .map(|qty| qty > 0.0)
                    .unwrap_or(false);
                if !qty_ok {
                    return Ok(None);
                }
                payload.insert("e".to_string(), json!("ORDER_TRADE_UPDATE"));
                payload.insert("m".to_string(), json!(maker));
                payload.insert("l".to_string(), json!(last_filled_qty));
            }
            Self::Extended { is_taker } => {
                if venue != "extended" {
                    return Ok(None);
                }
                payload.insert("isTaker".to_string(), json!(is_taker));
            }
            Self::Hyperliquid { crossed } => {
                if venue != "hyperliquid" {
                    return Ok(None);
                }
                payload.insert("crossed".to_string(), json!(crossed));
            }
            Self::Lighter {
                account_index,
                is_maker_ask,
                ask_account_id,
                bid_account_id,
            } => {
                if venue != "lighter" {
                    return Ok(None);
                }
                payload.insert("account_index".to_string(), json!(account_index));
                payload.insert("is_maker_ask".to_string(), json!(is_maker_ask));
                payload.insert("ask_account_id".to_string(), json!(ask_account_id));
                payload.insert("bid_account_id".to_string(), json!(bid_account_id));
            }
            Self::Paradex { liquidity } => {
                if venue != "paradex" {
                    return Ok(None);
                }
                let normalized = liquidity.trim().to_ascii_uppercase();
                if normalized != "MAKER" && normalized != "TAKER" {
                    return Ok(None);
                }
                payload.insert("liquidity".to_string(), json!(normalized));
            }
        }
        Ok(Some(payload))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Phase51LighterNativeLimitPressure {
    pub active_order_headroom_account: Option<i64>,
    pub active_order_sendtx_utilization_account: Option<i64>,
    pub rest_open_orders_count: Option<i64>,
    pub rest_open_orders_cap: Option<i64>,
    pub weighted_open_order_slots_used: Option<i64>,
    pub weighted_open_order_slots_cap: Option<i64>,
    pub native_limit_event_time_status: Option<String>,
}

impl Phase51LighterNativeLimitPressure {
    pub fn complete_rest(
        active_order_headroom_account: i64,
        active_order_sendtx_utilization_account: i64,
        rest_open_orders_count: i64,
        rest_open_orders_cap: i64,
    ) -> Self {
        Self {
            active_order_headroom_account: Some(active_order_headroom_account),
            active_order_sendtx_utilization_account: Some(active_order_sendtx_utilization_account),
            rest_open_orders_count: Some(rest_open_orders_count),
            rest_open_orders_cap: Some(rest_open_orders_cap),
            weighted_open_order_slots_used: None,
            weighted_open_order_slots_cap: None,
            native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
        }
    }

    pub fn complete_weighted(
        active_order_headroom_account: i64,
        active_order_sendtx_utilization_account: i64,
        weighted_open_order_slots_used: i64,
        weighted_open_order_slots_cap: i64,
    ) -> Self {
        Self {
            active_order_headroom_account: Some(active_order_headroom_account),
            active_order_sendtx_utilization_account: Some(active_order_sendtx_utilization_account),
            rest_open_orders_count: None,
            rest_open_orders_cap: None,
            weighted_open_order_slots_used: Some(weighted_open_order_slots_used),
            weighted_open_order_slots_cap: Some(weighted_open_order_slots_cap),
            native_limit_event_time_status: Some("EVENT_TIME_ALIGNED".to_string()),
        }
    }

    fn is_complete(&self) -> bool {
        let has_active_order_pair = self.active_order_headroom_account.is_some()
            && self.active_order_sendtx_utilization_account.is_some();
        let has_rest_pair =
            self.rest_open_orders_count.is_some() && self.rest_open_orders_cap.is_some();
        let has_weighted_pair = self.weighted_open_order_slots_used.is_some()
            && self.weighted_open_order_slots_cap.is_some();
        let event_time_aligned = self
            .native_limit_event_time_status
            .as_deref()
            .map(|status| status == "EVENT_TIME_ALIGNED")
            .unwrap_or(false);
        has_active_order_pair && (has_rest_pair || has_weighted_pair) && event_time_aligned
    }

    fn add_payload(&self, row: &mut Map<String, Value>) {
        if let Some(value) = self.active_order_headroom_account {
            row.insert("active_order_headroom_account".to_string(), json!(value));
        }
        if let Some(value) = self.active_order_sendtx_utilization_account {
            row.insert(
                "active_order_sendtx_utilization_account".to_string(),
                json!(value),
            );
        }
        if let Some(value) = self.rest_open_orders_count {
            row.insert("rest_open_orders_count".to_string(), json!(value));
        }
        if let Some(value) = self.rest_open_orders_cap {
            row.insert("rest_open_orders_cap".to_string(), json!(value));
        }
        if let Some(value) = self.weighted_open_order_slots_used {
            row.insert("weighted_open_order_slots_used".to_string(), json!(value));
        }
        if let Some(value) = self.weighted_open_order_slots_cap {
            row.insert("weighted_open_order_slots_cap".to_string(), json!(value));
        }
        if let Some(value) = &self.native_limit_event_time_status {
            row.insert("native_limit_event_time_status".to_string(), json!(value));
        }
    }
}

#[derive(Debug)]
pub struct Phase51ForwardRefreshCapture {
    config: Phase51ForwardRefreshCaptureConfig,
    rows_written: usize,
    source_owner_pfill_observation_rows_written: usize,
    live_native_role_canary_mode: bool,
}

impl Phase51ForwardRefreshCapture {
    pub fn from_config(
        config: &Phase51ForwardRefreshCaptureConfig,
        execution_mode: Phase51CaptureExecutionMode,
    ) -> Phase51CaptureResult<Self> {
        Self::from_config_with_live_native_role_canary_context(config, execution_mode, None)
    }

    pub fn from_config_with_live_native_role_canary_context(
        config: &Phase51ForwardRefreshCaptureConfig,
        execution_mode: Phase51CaptureExecutionMode,
        live_context: Option<&Phase51LiveNativeRoleCanaryContext>,
    ) -> Phase51CaptureResult<Self> {
        if config.allow_live {
            return Err(Phase51CaptureError::new(
                "phase51 forward-refresh capture is fail-closed: allow_live=true is prohibited",
            ));
        }

        let mut rows_written = 0usize;
        let mut source_owner_pfill_observation_rows_written = 0usize;
        let mut live_native_role_canary_mode = false;
        if config.enabled {
            if !execution_mode.is_non_live_shadow_safe() {
                if execution_mode == Phase51CaptureExecutionMode::Live
                    && config.live_native_role_canary_approved
                {
                    validate_live_native_role_canary_context(config, live_context)?;
                    live_native_role_canary_mode = true;
                } else {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 forward-refresh capture requires non-live/shadow-safe mode; got {execution_mode:?}",
                    )));
                }
            }
            if !config.append_only {
                return Err(Phase51CaptureError::new(
                    "phase51 forward-refresh capture requires append_only=true",
                ));
            }
            let output_path = Path::new(&config.output_path);
            validate_output_path(output_path)?;
            if live_native_role_canary_mode {
                validate_live_native_role_canary_output_path(output_path)?;
            }
            if output_path.exists() {
                rows_written = count_nonempty_lines(output_path)?;
                if live_native_role_canary_mode && rows_written > 0 {
                    return Err(Phase51CaptureError::new(
                        "phase51 live native-role canary capture requires absent or empty future output",
                    ));
                }
                if rows_written > config.max_rows {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 forward-refresh capture output already exceeds max_rows: {rows_written} > {}",
                        config.max_rows
                    )));
                }
            }
            let pfill_observation_output_path =
                source_owner_pfill_observation_output_path(output_path);
            validate_output_path(&pfill_observation_output_path)?;
            if pfill_observation_output_path.exists() {
                source_owner_pfill_observation_rows_written =
                    count_nonempty_lines(&pfill_observation_output_path)?;
                if live_native_role_canary_mode && source_owner_pfill_observation_rows_written > 0 {
                    return Err(Phase51CaptureError::new(
                        "phase51 live native-role canary capture requires absent or empty source-owner pfill observation output",
                    ));
                }
                if source_owner_pfill_observation_rows_written > config.max_rows {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 source-owner pfill observation output already exceeds max_rows: {source_owner_pfill_observation_rows_written} > {}",
                        config.max_rows
                    )));
                }
            }
        }

        Ok(Self {
            config: config.clone(),
            rows_written,
            source_owner_pfill_observation_rows_written,
            live_native_role_canary_mode,
        })
    }

    pub fn live_native_role_canary_mode(&self) -> bool {
        self.live_native_role_canary_mode
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    pub fn source_owner_pfill_observation_rows_written(&self) -> usize {
        self.source_owner_pfill_observation_rows_written
    }

    pub fn capture_fill(
        &mut self,
        fill: &Fill,
    ) -> Phase51CaptureResult<Phase51ForwardRefreshCaptureAudit> {
        let mut audit = phase51_capture_audit_from_fill(self.config.enabled, fill);
        if !self.config.enabled {
            return Ok(audit);
        }

        let Some(runtime_target_key) = fill.phase51_target_key.as_ref() else {
            return Ok(audit);
        };
        let target_key = target_key_from_runtime(runtime_target_key);
        if !target_key.is_complete() {
            return Ok(audit);
        }
        audit.canonical_group_id = Some(target_key.canonical_group_id.clone());
        audit.order_key = Some(target_key.order_key.clone());

        if let Some(runtime_native_role) = fill.phase51_native_role.as_ref() {
            audit.native_role_source = Some(native_role_audit_source(runtime_native_role));
            let emitted = self.capture_native_role(
                Some(&target_key),
                &fill.venue_id,
                Some(native_role_from_runtime(runtime_native_role)),
            )?;
            if emitted.is_some() {
                audit.target_type = Some("native_role".to_string());
                audit.sanitized_row_emitted = true;
            }
        }

        if !self.live_native_role_canary_mode {
            if let Some(runtime_lighter_pressure) = fill.phase51_lighter_native_limit.as_ref() {
                audit.lighter_pressure_status = runtime_lighter_pressure
                    .native_limit_event_time_status
                    .clone();
                let emitted = self.capture_lighter_native_limit(
                    Some(&target_key),
                    Some(lighter_pressure_from_runtime(runtime_lighter_pressure)),
                )?;
                if emitted.is_some() {
                    audit.target_type = Some(match audit.target_type.as_deref() {
                        Some("native_role") => "native_role,lighter_native_limit".to_string(),
                        _ => "lighter_native_limit".to_string(),
                    });
                    audit.sanitized_row_emitted = true;
                }
            }
        }

        Ok(audit)
    }

    pub fn capture_source_owner_pfill_observation(
        &mut self,
        target_key: Option<&Phase51CaptureTargetKey>,
        venue_id: &str,
        native_role: &Phase51ForwardRefreshNativeRole,
        observation: &Phase51ForwardRefreshSourceOwnerPfillObservation,
    ) -> Phase51CaptureResult<Option<Value>> {
        if !self.config.enabled {
            return Ok(None);
        }
        let Some(target_key) = target_key else {
            return Ok(None);
        };
        if !target_key.is_complete() {
            return Ok(None);
        }
        let venue_id = canonical_venue_id(venue_id);
        if venue_id != "lighter" {
            return Ok(None);
        }
        let native_role = native_role_from_runtime(native_role);
        let Some(native_payload) = native_role.to_payload(&venue_id)? else {
            return Ok(None);
        };
        let Some(observation_payload) = source_owner_pfill_observation_payload(observation) else {
            return Ok(None);
        };

        let mut row = base_row("source_owner_pfill_observation", &venue_id, target_key);
        row.insert(
            "phase51_bridge_kind".to_string(),
            json!("source_owner_pfill_observation"),
        );
        row.insert("compatibility_view_only".to_string(), json!(false));
        row.insert(
            "source_owner_native_role_compatibility_only".to_string(),
            json!(false),
        );
        row.insert(
            "source_link_status".to_string(),
            json!("DIRECT_TARGET_LINKABLE"),
        );
        row.insert("source_link_inference_allowed".to_string(), json!(false));
        row.insert(
            "time_price_size_inference_allowed".to_string(),
            json!(false),
        );
        row.insert("role_inference_allowed".to_string(), json!(false));
        row.insert("missing_pressure_values_inferred".to_string(), json!(false));
        row.insert("blocker_cleared".to_string(), json!(false));
        row.insert("clears_phase51_blockers".to_string(), json!(false));
        row.insert(
            "native_role_source".to_string(),
            json!(native_role_audit_source_runtime(&native_role)),
        );
        row.insert(
            "native_role_exact_source_available".to_string(),
            json!(true),
        );
        for (key, value) in native_payload {
            row.insert(key, value);
        }
        for (key, value) in observation_payload {
            row.insert(key, value);
        }
        self.write_source_owner_pfill_observation_row(Value::Object(row))
    }

    pub fn capture_source_owner_fill(
        &mut self,
        fill: &Phase51ForwardRefreshSourceOwnerFill,
    ) -> Phase51CaptureResult<Phase51ForwardRefreshCaptureAudit> {
        let mut audit = phase51_capture_audit_from_source_owner_fill(self.config.enabled, fill);
        if !self.config.enabled {
            return Ok(audit);
        }

        let Some(runtime_target_key) = fill.phase51_target_key.as_ref() else {
            return Ok(audit);
        };
        let target_key = target_key_from_runtime(runtime_target_key);
        if !target_key.is_complete() {
            return Ok(audit);
        }
        audit.canonical_group_id = Some(target_key.canonical_group_id.clone());
        audit.order_key = Some(target_key.order_key.clone());

        let mut native_role_emitted = false;
        if let Some(runtime_native_role) = fill.phase51_native_role.as_ref() {
            audit.native_role_source = Some(native_role_audit_source(runtime_native_role));
            let emitted = self.capture_native_role(
                Some(&target_key),
                &fill.venue_id,
                Some(native_role_from_runtime(runtime_native_role)),
            )?;
            if emitted.is_some() {
                native_role_emitted = true;
                audit.target_type = Some("native_role".to_string());
                audit.sanitized_row_emitted = true;
            }
        }

        if native_role_emitted {
            if let (Some(runtime_native_role), Some(observation)) = (
                fill.phase51_native_role.as_ref(),
                fill.phase51_source_owner_pfill_observation.as_ref(),
            ) {
                self.capture_source_owner_pfill_observation(
                    Some(&target_key),
                    &fill.venue_id,
                    runtime_native_role,
                    observation,
                )?;
            }
        }

        if !self.live_native_role_canary_mode {
            if let Some(runtime_lighter_pressure) = fill.phase51_lighter_native_limit.as_ref() {
                audit.lighter_pressure_status = runtime_lighter_pressure
                    .native_limit_event_time_status
                    .clone();
                let emitted = self.capture_lighter_native_limit(
                    Some(&target_key),
                    Some(lighter_pressure_from_runtime(runtime_lighter_pressure)),
                )?;
                if emitted.is_some() {
                    audit.target_type = Some(match audit.target_type.as_deref() {
                        Some("native_role") => "native_role,lighter_native_limit".to_string(),
                        _ => "lighter_native_limit".to_string(),
                    });
                    audit.sanitized_row_emitted = true;
                }
            }
        }

        Ok(audit)
    }

    pub fn capture_native_role(
        &mut self,
        target_key: Option<&Phase51CaptureTargetKey>,
        venue_id: &str,
        native_role: Option<Phase51VenueNativeRole>,
    ) -> Phase51CaptureResult<Option<Value>> {
        if !self.config.enabled {
            return Ok(None);
        }
        let Some(target_key) = target_key else {
            return Ok(None);
        };
        if !target_key.is_complete() {
            return Ok(None);
        }
        let Some(native_role) = native_role else {
            return Ok(None);
        };
        let venue_id = canonical_venue_id(venue_id);
        if self.live_native_role_canary_mode && venue_id != "lighter" {
            return Ok(None);
        }
        let Some(payload) = native_role.to_payload(&venue_id)? else {
            return Ok(None);
        };

        let mut row = base_row("native_role", &venue_id, target_key);
        for (key, value) in payload {
            row.insert(key, value);
        }
        self.write_row(Value::Object(row))
    }

    pub fn capture_lighter_native_limit(
        &mut self,
        target_key: Option<&Phase51CaptureTargetKey>,
        pressure: Option<Phase51LighterNativeLimitPressure>,
    ) -> Phase51CaptureResult<Option<Value>> {
        if !self.config.enabled || self.live_native_role_canary_mode {
            return Ok(None);
        }
        let Some(target_key) = target_key else {
            return Ok(None);
        };
        if !target_key.is_complete() {
            return Ok(None);
        }
        let Some(pressure) = pressure else {
            return Ok(None);
        };
        if !pressure.is_complete() {
            return Ok(None);
        }

        let mut row = base_row("lighter_native_limit", "lighter", target_key);
        pressure.add_payload(&mut row);
        self.write_row(Value::Object(row))
    }

    fn write_row(&mut self, row: Value) -> Phase51CaptureResult<Option<Value>> {
        enforce_output_safety(&row)?;
        if self.rows_written >= self.config.max_rows {
            return Err(Phase51CaptureError::new(format!(
                "phase51 forward-refresh capture max_rows reached: {}",
                self.config.max_rows
            )));
        }

        let output_path = PathBuf::from(&self.config.output_path);
        append_row_to_path(&output_path, &row, "phase51 capture output")?;
        self.rows_written += 1;
        Ok(Some(row))
    }

    fn write_source_owner_pfill_observation_row(
        &mut self,
        row: Value,
    ) -> Phase51CaptureResult<Option<Value>> {
        enforce_output_safety(&row)?;
        if self.source_owner_pfill_observation_rows_written >= self.config.max_rows {
            return Err(Phase51CaptureError::new(format!(
                "phase51 source-owner pfill observation max_rows reached: {}",
                self.config.max_rows
            )));
        }

        let output_path =
            source_owner_pfill_observation_output_path(Path::new(&self.config.output_path));
        append_row_to_path(
            &output_path,
            &row,
            "phase51 source-owner pfill observation output",
        )?;
        self.source_owner_pfill_observation_rows_written += 1;
        Ok(Some(row))
    }
}

fn phase51_capture_audit_from_fill(
    enabled: bool,
    fill: &Fill,
) -> Phase51ForwardRefreshCaptureAudit {
    Phase51ForwardRefreshCaptureAudit {
        enabled,
        target_type: None,
        venue_id: Some(canonical_venue_id(&fill.venue_id)),
        canonical_group_id: None,
        order_key: None,
        native_role_source: fill
            .phase51_native_role
            .as_ref()
            .map(native_role_audit_source),
        lighter_pressure_status: fill
            .phase51_lighter_native_limit
            .as_ref()
            .and_then(|pressure| pressure.native_limit_event_time_status.clone()),
        sanitized_row_emitted: false,
        no_live_flag: true,
        approved_for_live: false,
        approved_for_canary: false,
        approved_for_capital_escalation: false,
    }
}

fn phase51_capture_audit_from_source_owner_fill(
    enabled: bool,
    fill: &Phase51ForwardRefreshSourceOwnerFill,
) -> Phase51ForwardRefreshCaptureAudit {
    Phase51ForwardRefreshCaptureAudit {
        enabled,
        target_type: None,
        venue_id: Some(canonical_venue_id(&fill.venue_id)),
        canonical_group_id: None,
        order_key: None,
        native_role_source: fill
            .phase51_native_role
            .as_ref()
            .map(native_role_audit_source),
        lighter_pressure_status: fill
            .phase51_lighter_native_limit
            .as_ref()
            .and_then(|pressure| pressure.native_limit_event_time_status.clone()),
        sanitized_row_emitted: false,
        no_live_flag: true,
        approved_for_live: false,
        approved_for_canary: false,
        approved_for_capital_escalation: false,
    }
}

fn target_key_from_runtime(target_key: &Phase51ForwardRefreshTargetKey) -> Phase51CaptureTargetKey {
    Phase51CaptureTargetKey::new(&target_key.canonical_group_id, &target_key.order_key)
}

fn native_role_from_runtime(
    native_role: &Phase51ForwardRefreshNativeRole,
) -> Phase51VenueNativeRole {
    match native_role {
        Phase51ForwardRefreshNativeRole::Aster {
            maker,
            last_filled_qty,
        } => Phase51VenueNativeRole::Aster {
            maker: *maker,
            last_filled_qty: last_filled_qty.clone(),
        },
        Phase51ForwardRefreshNativeRole::Extended { is_taker } => {
            Phase51VenueNativeRole::Extended {
                is_taker: *is_taker,
            }
        }
        Phase51ForwardRefreshNativeRole::Hyperliquid { crossed } => {
            Phase51VenueNativeRole::Hyperliquid { crossed: *crossed }
        }
        Phase51ForwardRefreshNativeRole::Lighter {
            account_index,
            is_maker_ask,
            ask_account_id,
            bid_account_id,
        } => Phase51VenueNativeRole::Lighter {
            account_index: *account_index,
            is_maker_ask: *is_maker_ask,
            ask_account_id: *ask_account_id,
            bid_account_id: *bid_account_id,
        },
        Phase51ForwardRefreshNativeRole::Paradex { liquidity } => Phase51VenueNativeRole::Paradex {
            liquidity: liquidity.clone(),
        },
    }
}

fn native_role_audit_source(native_role: &Phase51ForwardRefreshNativeRole) -> String {
    match native_role {
        Phase51ForwardRefreshNativeRole::Aster { .. } => "aster.ORDER_TRADE_UPDATE",
        Phase51ForwardRefreshNativeRole::Extended { .. } => "extended.isTaker",
        Phase51ForwardRefreshNativeRole::Hyperliquid { .. } => "hyperliquid.crossed",
        Phase51ForwardRefreshNativeRole::Lighter { .. } => {
            "lighter.account_index.is_maker_ask.ask_account_id.bid_account_id"
        }
        Phase51ForwardRefreshNativeRole::Paradex { .. } => "paradex.liquidity",
    }
    .to_string()
}

fn native_role_audit_source_runtime(native_role: &Phase51VenueNativeRole) -> String {
    match native_role {
        Phase51VenueNativeRole::Aster { .. } => "aster.ORDER_TRADE_UPDATE",
        Phase51VenueNativeRole::Extended { .. } => "extended.isTaker",
        Phase51VenueNativeRole::Hyperliquid { .. } => "hyperliquid.crossed",
        Phase51VenueNativeRole::Lighter { .. } => {
            "lighter.account_index.is_maker_ask.ask_account_id.bid_account_id"
        }
        Phase51VenueNativeRole::Paradex { .. } => "paradex.liquidity",
    }
    .to_string()
}

fn source_owner_pfill_observation_payload(
    observation: &Phase51ForwardRefreshSourceOwnerPfillObservation,
) -> Option<Map<String, Value>> {
    if observation.source_event_type != "LIGHTER_TRADES_JSON" {
        return None;
    }
    if !observation.price.is_finite() || observation.price <= 0.0 {
        return None;
    }
    if !observation.size.is_finite() || observation.size <= 0.0 {
        return None;
    }
    if observation.event_time_ms <= 0 {
        return None;
    }
    if observation.fill_count != 1
        || observation.outcome_status != "OBSERVED_FILLED"
        || (observation.p_fill_outcome - 1.0).abs() > f64::EPSILON
    {
        return None;
    }

    let mut payload = Map::new();
    payload.insert(
        "source_event_type".to_string(),
        json!(observation.source_event_type),
    );
    payload.insert("side".to_string(), json!(side_text(observation.side)));
    payload.insert("price".to_string(), json!(observation.price));
    payload.insert("size".to_string(), json!(observation.size));
    payload.insert(
        "event_time_ms".to_string(),
        json!(observation.event_time_ms),
    );
    payload.insert(
        "first_fill_time_ms".to_string(),
        json!(observation.event_time_ms),
    );
    payload.insert(
        "last_fill_time_ms".to_string(),
        json!(observation.event_time_ms),
    );
    payload.insert("fill_count".to_string(), json!(observation.fill_count));
    payload.insert(
        "outcome_status".to_string(),
        json!(observation.outcome_status),
    );
    payload.insert(
        "p_fill_outcome".to_string(),
        json!(observation.p_fill_outcome),
    );
    payload.insert("terminal_event_count".to_string(), json!(1));
    payload.insert(
        "terminal_action_first".to_string(),
        json!("source_owner_fill"),
    );
    payload.insert(
        "observed_side_source".to_string(),
        json!("LIGHTER_TRADES_JSON.account_side"),
    );
    payload.insert(
        "observed_price_size_source".to_string(),
        json!("LIGHTER_TRADES_JSON.price_size"),
    );
    payload.insert(
        "observed_outcome_source".to_string(),
        json!("LIGHTER_TRADES_JSON.trade_event"),
    );
    if let (Some(order_source_tick), Some(fill_source_tick), Some(observed_horizon_source_ticks)) = (
        observation.order_source_tick,
        observation.fill_source_tick,
        observation.observed_horizon_source_ticks(),
    ) {
        payload.insert("order_source_tick".to_string(), json!(order_source_tick));
        payload.insert("fill_source_tick".to_string(), json!(fill_source_tick));
        payload.insert(
            "observed_horizon_source_ticks".to_string(),
            json!(observed_horizon_source_ticks),
        );
        payload.insert(
            "observed_horizon_source_ticks_source".to_string(),
            json!("runtime_source_tick"),
        );
    }
    Some(payload)
}

fn side_text(side: crate::types::Side) -> &'static str {
    match side {
        crate::types::Side::Buy => "Buy",
        crate::types::Side::Sell => "Sell",
    }
}

fn source_owner_pfill_observation_output_path(output_path: &Path) -> PathBuf {
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("phase51_forward_refresh");
    let stem = file_name.strip_suffix(".jsonl").unwrap_or(file_name);
    let mut path = output_path.to_path_buf();
    path.set_file_name(format!("{stem}.source_owner_pfill_observation.jsonl"));
    path
}

fn append_row_to_path(path: &Path, row: &Value, label: &str) -> Phase51CaptureResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            Phase51CaptureError::new(format!(
                "failed to create {label} directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| {
            Phase51CaptureError::new(format!("failed to open {label} {}: {err}", path.display()))
        })?;
    serde_json::to_writer(&mut file, row)
        .map_err(|err| Phase51CaptureError::new(format!("failed to serialize {label}: {err}")))?;
    file.write_all(b"\n")
        .map_err(|err| Phase51CaptureError::new(format!("failed to append {label} newline: {err}")))
}

fn lighter_pressure_from_runtime(
    pressure: &Phase51ForwardRefreshLighterNativeLimit,
) -> Phase51LighterNativeLimitPressure {
    Phase51LighterNativeLimitPressure {
        active_order_headroom_account: pressure.active_order_headroom_account,
        active_order_sendtx_utilization_account: pressure.active_order_sendtx_utilization_account,
        rest_open_orders_count: pressure.rest_open_orders_count,
        rest_open_orders_cap: pressure.rest_open_orders_cap,
        weighted_open_order_slots_used: pressure.weighted_open_order_slots_used,
        weighted_open_order_slots_cap: pressure.weighted_open_order_slots_cap,
        native_limit_event_time_status: pressure.native_limit_event_time_status.clone(),
    }
}

fn base_row(
    target_type: &str,
    venue_id: &str,
    target_key: &Phase51CaptureTargetKey,
) -> Map<String, Value> {
    let mut row = Map::new();
    row.insert("target_type".to_string(), json!(target_type));
    row.insert("venue_id".to_string(), json!(venue_id));
    row.insert(
        "canonical_group_id".to_string(),
        json!(target_key.canonical_group_id.trim()),
    );
    row.insert("order_key".to_string(), json!(target_key.order_key.trim()));
    row.insert("no_live_flag".to_string(), json!(true));
    row.insert("approved_for_live".to_string(), json!(false));
    row.insert("approved_for_canary".to_string(), json!(false));
    row.insert("approved_for_model_training".to_string(), json!(false));
    row.insert("approved_for_capital_escalation".to_string(), json!(false));
    row.insert("admissible_for_financial_claim".to_string(), json!(false));
    row.insert("admissible_for_ev_admission".to_string(), json!(false));
    row.insert("live_orders_allowed".to_string(), json!(false));
    row.insert("capital_change_allowed".to_string(), json!(false));
    row.insert("risk_limit_relaxation_allowed".to_string(), json!(false));
    row
}

fn canonical_venue_id(venue_id: &str) -> String {
    venue_id.trim().to_ascii_lowercase()
}

fn validate_live_native_role_canary_context(
    config: &Phase51ForwardRefreshCaptureConfig,
    live_context: Option<&Phase51LiveNativeRoleCanaryContext>,
) -> Phase51CaptureResult<()> {
    let Some(live_context) = live_context else {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires explicit runtime context",
        ));
    };
    if !live_context.canary_enabled {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires canary mode",
        ));
    }
    if !live_context.native_role_strict_canary_enabled {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires strict native-role canary mode",
        ));
    }
    if !live_context.native_role_one_sided_canary_enabled {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires one-sided native-role canary mode",
        ));
    }
    let venue_ids: Vec<String> = live_context
        .venue_ids
        .iter()
        .map(|venue| canonical_venue_id(venue))
        .filter(|venue| !venue.is_empty())
        .collect();
    if venue_ids.as_slice() != ["lighter"] {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires Lighter-only venue selection",
        ));
    }
    if !live_context.strict_maker_only_observation_enabled {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires strict maker-only Lighter observation mode",
        ));
    }
    if !live_context.canary_enforce_post_only {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires canary post-only enforcement",
        ));
    }
    if live_context.canary_enforce_reduce_only {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires non-reduce-only canary enforcement",
        ));
    }
    if live_context.canary_max_open_orders != Some(1) {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires max_open_orders=1",
        ));
    }
    if !live_context.replacements_disabled {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires replacements disabled",
        ));
    }
    if !live_context.stop_after_first_row {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires stop-after-first-row",
        ));
    }
    if config.max_rows != 1 {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires max_rows=1",
        ));
    }
    Ok(())
}

fn validate_live_native_role_canary_output_path(path: &Path) -> Phase51CaptureResult<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if file_name == "forward_refresh.remaining.jsonl" {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture must not write forward_refresh.remaining.jsonl",
        ));
    }
    if !file_name.ends_with(".jsonl") || !file_name.contains("future_native_role") {
        return Err(Phase51CaptureError::new(
            "phase51 live native-role canary capture requires a future_native_role .jsonl output path",
        ));
    }
    Ok(())
}

fn validate_output_path(path: &Path) -> Phase51CaptureResult<()> {
    if path.as_os_str().is_empty() {
        return Err(Phase51CaptureError::new(
            "phase51 forward-refresh capture output_path is empty",
        ));
    }
    for component in path.components() {
        if let Component::Normal(part) = component {
            let part = part.to_string_lossy().to_ascii_lowercase();
            if part == ".env" || part.ends_with(".env") {
                return Err(Phase51CaptureError::new(
                    "phase51 forward-refresh capture output_path must not reference .env content",
                ));
            }
        }
    }
    reject_symlink_path(path)?;
    Ok(())
}

fn reject_symlink_path(path: &Path) -> Phase51CaptureResult<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Err(Phase51CaptureError::new(format!(
                    "phase51 forward-refresh capture rejects symlink path component: {}",
                    current.display()
                )));
            }
        }
    }
    Ok(())
}

fn count_nonempty_lines(path: &Path) -> Phase51CaptureResult<usize> {
    let file = File::open(path).map_err(|err| {
        Phase51CaptureError::new(format!(
            "failed to open existing phase51 capture output {}: {err}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line.map_err(|err| {
            Phase51CaptureError::new(format!(
                "failed to read existing phase51 capture output {}: {err}",
                path.display()
            ))
        })?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

fn enforce_output_safety(row: &Value) -> Phase51CaptureResult<()> {
    let no_live_flag = row
        .get("no_live_flag")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !no_live_flag {
        return Err(Phase51CaptureError::new(
            "phase51 capture row rejected: no_live_flag must be true",
        ));
    }
    enforce_json_safety(row)
}

fn enforce_json_safety(value: &Value) -> Phase51CaptureResult<()> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let normalized_key = key.to_ascii_lowercase();
                if FORBIDDEN_FIELD_NAMES.iter().any(|field| field == key)
                    || FORBIDDEN_KEY_FRAGMENTS
                        .iter()
                        .any(|fragment| normalized_key.contains(fragment))
                {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 capture row rejected forbidden field: {key}",
                    )));
                }
                if UNSAFE_TRUE_FLAGS.iter().any(|flag| flag == key)
                    && child.as_bool().unwrap_or(false)
                {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 capture row rejected unsafe true flag: {key}",
                    )));
                }
                enforce_json_safety(child)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                enforce_json_safety(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}
