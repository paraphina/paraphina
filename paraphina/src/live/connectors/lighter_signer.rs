use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct LighterSignerClient {
    base_url: String,
    http: reqwest::Client,
}

impl LighterSignerClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
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

    async fn send_sign_request<T: Serialize + ?Sized>(&self, req: &T) -> anyhow::Result<SignedTx> {
        let url = format!("{}/sign", self.base_url);
        let resp = self.http.post(url).json(req).send().await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("signer error status={} body={}", status, body);
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

#[derive(Debug, Deserialize)]
struct SignerResponse {
    tx_type: u32,
    tx_info: Value,
    #[serde(default)]
    tx_hash: Option<String>,
}
