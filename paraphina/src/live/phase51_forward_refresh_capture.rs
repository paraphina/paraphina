use serde_json::{json, Map, Value};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};

use crate::config::Phase51ForwardRefreshCaptureConfig;

use super::types::{
    Fill, Phase51ForwardRefreshCaptureAudit, Phase51ForwardRefreshLighterNativeLimit,
    Phase51ForwardRefreshNativeRole, Phase51ForwardRefreshTargetKey,
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
}

impl Phase51ForwardRefreshCapture {
    pub fn from_config(
        config: &Phase51ForwardRefreshCaptureConfig,
        execution_mode: Phase51CaptureExecutionMode,
    ) -> Phase51CaptureResult<Self> {
        if config.allow_live {
            return Err(Phase51CaptureError::new(
                "phase51 forward-refresh capture is fail-closed: allow_live=true is prohibited",
            ));
        }

        let mut rows_written = 0usize;
        if config.enabled {
            if !execution_mode.is_non_live_shadow_safe() {
                return Err(Phase51CaptureError::new(format!(
                    "phase51 forward-refresh capture requires non-live/shadow-safe mode; got {execution_mode:?}",
                )));
            }
            if !config.append_only {
                return Err(Phase51CaptureError::new(
                    "phase51 forward-refresh capture requires append_only=true",
                ));
            }
            let output_path = Path::new(&config.output_path);
            validate_output_path(output_path)?;
            if output_path.exists() {
                rows_written = count_nonempty_lines(output_path)?;
                if rows_written > config.max_rows {
                    return Err(Phase51CaptureError::new(format!(
                        "phase51 forward-refresh capture output already exceeds max_rows: {rows_written} > {}",
                        config.max_rows
                    )));
                }
            }
        }

        Ok(Self {
            config: config.clone(),
            rows_written,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
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
        let Some(payload) = native_role.to_payload(venue_id)? else {
            return Ok(None);
        };

        let venue_id = canonical_venue_id(venue_id);
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
        if !self.config.enabled {
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
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                Phase51CaptureError::new(format!(
                    "failed to create phase51 capture output directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .map_err(|err| {
                Phase51CaptureError::new(format!(
                    "failed to open phase51 capture output {}: {err}",
                    output_path.display()
                ))
            })?;
        serde_json::to_writer(&mut file, &row).map_err(|err| {
            Phase51CaptureError::new(format!("failed to serialize phase51 capture row: {err}"))
        })?;
        file.write_all(b"\n").map_err(|err| {
            Phase51CaptureError::new(format!("failed to append phase51 capture newline: {err}"))
        })?;
        self.rows_written += 1;
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
