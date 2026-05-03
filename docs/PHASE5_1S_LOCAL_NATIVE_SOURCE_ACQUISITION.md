# Phase 5.1s Local Native Source Acquisition

Status: `HOLD`, non-live source staging only.

## Objective

Phase 5.1s is the repo-owned local-source safety gate in front of Phase 5.1r.
It accepts an explicit manifest of local `.json` / `.jsonl` native snapshots,
rejects network paths, `.env` files, symlinks, unsafe authorization flags, and
secret-shaped fields, then emits one redacted JSONL source file for Phase 5.1r.

This gate does not authorize live orders, canary, capital escalation,
risk-limit relaxation, EV admission, model training, or financial claims.

## Tool

Command:

```bash
python3 tools/phase51s_local_native_source_acquisition.py \
  --manifest configs/phase51s_local_native_source_manifest.example.json \
  --run-id <phase51s_run_id>
```

Outputs:

- `phase51s_local_native_source_acquisition_summary.json`
- `local_native_source.jsonl`
- `local_source_labels.jsonl`
- `phase51s_manifest.json`

`local_native_source.jsonl` is the only output intended for Phase 5.1r:

```bash
python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/<canonical_pfill_run> \
  --source-json runs/phase51s_local_native_source_acquisition/<phase51s_run_id>/local_native_source.jsonl \
  --run-id <phase51r_run_id>
```

If a forward capture produces a separate redacted mapping from
`phase51s_source_record_sha256`, `source_record_sha256`, or
`redacted_source_record_sha256` to observed `canonical_group_id` / `order_key`,
pass that mapping to Phase 5.1r with repeated `--source-link-jsonl` arguments.
Phase 5.1r validates the sidecar against observed P-fill labels and rejects raw
IDs, duplicate hashes, ambiguous mappings, or unsafe flags. The sidecar is only
join evidence; it does not infer maker/taker role or Lighter native-limit
pressure.

## Safety Contract

Phase 5.1s rejects:

- `http://`, `https://`, or other URI-like source paths.
- `.env` manifests or source files.
- Symlinked manifests or source files.
- Any manifest/source field that looks like an API key, private key, secret,
  password, passphrase, token, JWT, bearer authorization, or session credential.
- Any unsafe true authorization flag.

Phase 5.1s strips raw venue identifier fields before output, including
order/client/fill/trade IDs and common venue-native ID aliases. It preserves
only non-secret fields needed by Phase 5.1r, such as `canonical_group_id`,
`order_key`, venue, native maker/taker role fields, and native limit-pressure
fields. Phase 5.1r may also join by the redacted source-record hash when a
validated source-link sidecar is supplied.

## Boundary

Phase 5.1s does not clear Phase 5.1 matrix blockers by itself. It only stages
source rows. Blocker reduction requires the downstream Phase 5.1r -> 5.1q ->
5.1n -> 5.1h -> 5.1i chain to prove that staged rows join to canonical labels
and contain complete explicit venue-native evidence.

The current required source coverage remains:

- All five venues: explicit venue-native maker/taker role fields for filled
  rows that still lack observed role evidence.
- Lighter: event-time active-order, sendTx, and REST/weighted-request pressure
  fields for native-limit pressure labels.

## Current Board Verdict

`PROMOTE` only for local non-live source staging and downstream evidence reruns.

`HOLD` remains for live orders, canary, capital escalation, risk-limit
relaxation, model training, EV admission, financial claims, and 24/7 readiness.
