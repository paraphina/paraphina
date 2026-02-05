# Extended Funding Investigation

## TL;DR

Extended funding was failing in shadow mode because:
1. The default `EXTENDED_REST_URL` pointed to `https://api.extended.exchange` which returns HTTP 404 for the funding endpoint
2. The default `EXTENDED_FUNDING_PATH` was `/fapi/v1/premiumIndex` which doesn't exist on the working API
3. The correct working API is the Starknet-specific endpoint: `https://api.starknet.extended.exchange`
4. The correct funding path is `/api/v1/info/markets/{market}/stats`
5. The `fundingRate` field in the response is hourly (not 8h), requiring normalization
6. The `nextFundingRate` field is misnamed—it's actually the next funding timestamp in milliseconds

## Root Cause Analysis

### Symptom
In shadow mode telemetry, Extended (venue[1]) showed:
- `venue_funding_status=Unknown`
- `venue_funding_rate_8h=null`
- Log errors: "Extended funding polling error: error decoding response body"

### Investigation Steps

1. **Verified API failure** with `curl`:
   ```bash
   curl -s https://api.extended.exchange/fapi/v1/premiumIndex?symbol=ETH-USD
   # Returns HTTP 404 "Page not found"
   ```

2. **Found working configuration** in `docs/RUNBOOK.md`:
   ```
   EXTENDED_REST_URL=https://api.starknet.extended.exchange
   EXTENDED_FUNDING_PATH=/api/v1/info/markets/ETH-USD/stats
   ```

3. **Verified working endpoint**:
   ```bash
   curl -s "https://api.starknet.extended.exchange/api/v1/info/markets/ETH-USD/stats"
   ```
   Returns:
   ```json
   {
     "status": "OK",
     "data": {
       "fundingRate": "0.000013",
       "nextFundingRate": 1770314400000,
       "markPrice": "1963.0832622375",
       ...
     }
   }
   ```

4. **Discovered field semantics**:
   - `fundingRate`: **hourly** rate (not 8h)
   - `nextFundingRate`: **timestamp in ms** (not a rate!)

## Fix Applied

### File: `paraphina/src/live/connectors/extended.rs`

1. **Updated default `rest_url`** (line ~117):
   ```rust
   // Default to Starknet Extended API (api.extended.exchange returns 404)
   let rest_url = std::env::var("EXTENDED_REST_URL")
       .unwrap_or_else(|_| "https://api.starknet.extended.exchange".to_string());
   ```

2. **Updated `fetch_public_funding`** (line ~1081):
   - Changed default path to `/api/v1/info/markets/{market}/stats`
   - Market symbol is now part of path, not a query parameter
   - Added better error logging (HTTP status, URL, body snippet)

3. **Updated `parse_public_funding`** (line ~1095):
   - Extract `next_funding_ms` from `nextFundingRate` field
   - Default `interval_sec = 3600` (1 hour) when `fundingRate` present
   - Normalize: `rate_8h = rate_native * 8` (since native is hourly)
   - Set `settlement_price_kind = Mark` (Extended uses mark price)

### New Test

Added deterministic test: `parse_public_funding_market_stats_fixture`
- Fixture: `tests/fixtures/extended/public_market_stats.json`
- Validates rate parsing, normalization, timestamp extraction

## Validation

```bash
# Build
cargo build -p paraphina --bin paraphina_live --features "live,live_extended,live_hyperliquid,live_aster,live_lighter,live_paradex"

# Test
cargo test -p paraphina --features "live,live_extended" "parse_public_funding_market_stats_fixture"

# Shadow validation (run for ~30s, check telemetry)
RUST_LOG=info ./target/debug/paraphina_live --trade-mode shadow --ticks 300
# Expect: venue[1] funding_status=Healthy, funding_rate_8h=non-null
```

## Notes

- The Extended API has two base URLs:
  - `api.extended.exchange` - appears deprecated/non-functional for perpetual endpoints
  - `api.starknet.extended.exchange` - active Starknet deployment with working perpetual/funding data
- Environment variables `EXTENDED_REST_URL` and `EXTENDED_FUNDING_PATH` can still be used to override if the API changes
