# Burn-in Report

## Metadata

- Timestamp (UTC): {{timestamp_utc}}
- Mode: {{mode}}
- Connector: {{connector}}
- Features: {{features}}
- Out dir: {{out_dir}}
- Ticks: {{ticks}}
- Fixture dir: {{fixture_dir}}

## Run Results

- Run exit code: {{run_exit_code}}
- Telemetry contract: {{telemetry_contract}}
- Summary file: {{summary_present}}

## Summary Stats

- Ticks run: {{ticks_run}}
- Kill switch: {{kill_switch}}
- Execution mode: {{execution_mode}}
- FV available rate: {{fv_available_rate}}
- Fills count: {{fills_count}}
- Drift events: {{drift_count}}
- Cancel acks: {{cancel_count}}

## Venue Rollups

{{venue_rollups}}

## Gates

- Gate: telemetry contract valid → {{gate_telemetry}}
- Gate: kill switch not triggered → {{gate_kill_switch}}
- Gate: execution mode matches → {{gate_execution_mode}}

## Notes

{{notes}}
