use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct LighterSignerClient {
    base_url: String,
    http: reqwest::Client,
}

impl LighterSignerClient {
    pub fn new(base_url: String) -> Self {
        let base_url = normalize_signer_base_url(&base_url);
        Self {
            base_url,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
                .pool_idle_timeout(std::time::Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("lighter signer http client build"),
        }
    }

    pub async fn sign_create_order(&self, req: SignCreateOrderRequest) -> anyhow::Result<SignedTx> {
        self.send_sign_request(&req).await
    }

    pub async fn sign_cancel_order(&self, req: SignCancelOrderRequest) -> anyhow::Result<SignedTx> {
        self.send_sign_request(&req).await
    }

    pub async fn sign_cancel_all(&self, req: SignCancelAllRequest) -> anyhow::Result<SignedTx> {
        self.send_sign_request(&req).await
    }

    pub async fn sign_modify_order(&self, req: SignModifyOrderRequest) -> anyhow::Result<SignedTx> {
        self.send_sign_request(&req).await
    }

    async fn send_sign_request<T: Serialize + ?Sized>(&self, req: &T) -> anyhow::Result<SignedTx> {
        let url = format!("{}/sign", self.base_url);
        let resp = self.http.post(url).json(req).send().await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("signer error status={} reason=non_success", status);
        }
        let parsed: SignerResponse = serde_json::from_str(&body)?;
        Ok(SignedTx {
            tx_type: parsed.tx_type,
            tx_info: parsed.tx_info,
            tx_hash: parsed.tx_hash,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SignedTx {
    pub tx_type: u32,
    pub tx_info: Value,
    pub tx_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SignCreateOrderRequest {
    pub op: String,
    pub account_index: u64,
    pub api_key_index: u64,
    pub nonce: u64,
    pub market_index: u64,
    pub client_order_index: u64,
    pub base_amount: i64,
    pub price: i64,
    pub is_ask: u8,
    pub order_type: String,
    pub time_in_force: String,
    pub post_only: u8,
    pub reduce_only: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_price: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_expiry: Option<u64>,
    pub expired_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SignCancelOrderRequest {
    pub op: String,
    pub account_index: u64,
    pub api_key_index: u64,
    pub nonce: u64,
    pub market_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_order_index: Option<u64>,
    pub expired_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SignCancelAllRequest {
    pub op: String,
    pub account_index: u64,
    pub api_key_index: u64,
    pub nonce: u64,
    pub cancel_all_time_in_force: u8,
    pub cancel_all_time: u64,
    pub expired_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SignModifyOrderRequest {
    pub op: String,
    pub account_index: u64,
    pub api_key_index: u64,
    pub nonce: u64,
    pub market_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_order_index: Option<u64>,
    pub base_amount: i64,
    pub price: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_price: Option<i64>,
    pub expired_at: u64,
}

#[derive(Debug, Deserialize)]
struct SignerResponse {
    tx_type: u32,
    tx_info: Value,
    #[serde(default)]
    tx_hash: Option<String>,
}

fn normalize_signer_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    trimmed
        .strip_suffix("/sign")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{normalize_signer_base_url, SignModifyOrderRequest};

    #[test]
    fn normalize_signer_base_url_accepts_base_or_sign_path() {
        assert_eq!(
            normalize_signer_base_url("http://127.0.0.1:9001"),
            "http://127.0.0.1:9001"
        );
        assert_eq!(
            normalize_signer_base_url("http://127.0.0.1:9001/"),
            "http://127.0.0.1:9001"
        );
        assert_eq!(
            normalize_signer_base_url("http://127.0.0.1:9001/sign"),
            "http://127.0.0.1:9001"
        );
        assert_eq!(
            normalize_signer_base_url("http://127.0.0.1:9001/sign/"),
            "http://127.0.0.1:9001"
        );
    }

    #[test]
    fn modify_request_serializes_order_identity() {
        let payload = serde_json::to_value(SignModifyOrderRequest {
            op: "modify_order".to_string(),
            account_index: 123,
            api_key_index: 1,
            nonce: 456,
            market_index: 7,
            order_index: None,
            client_order_index: Some(42),
            base_amount: 1234,
            price: 220055,
            trigger_price: None,
            expired_at: 999_999,
        })
        .expect("serialize modify request");
        assert_eq!(
            payload.get("op").and_then(|v| v.as_str()),
            Some("modify_order")
        );
        assert_eq!(
            payload.get("client_order_index").and_then(|v| v.as_u64()),
            Some(42)
        );
        assert!(payload.get("order_index").is_none());
    }
}
