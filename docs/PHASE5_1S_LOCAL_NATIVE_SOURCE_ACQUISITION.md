# Phase 5.1s Local Native Source Acquisition

Status: `HOLD`, non-live source staging only.

## Objective

Phase 5.1s is the repo-owned local-source safety gate in front of Phase 5.1r.
It accepts an explicit manifest of local `.json` / `.jsonl` native snapshots,
rejects network paths, `.env` files, symlinks, unsafe authorization flags, and
secret-shaped fields, then emits redacted JSONL source files for Phase 5.1r.
The manifest may also include local redacted source-link sidecars that bind
source-record hashes to observed `canonical_group_id` / `order_key` values.

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
- `local_source_link_sidecar.jsonl`
- `local_source_link_labels.jsonl`
- `phase51s_manifest.json`

`local_native_source.jsonl` and `local_source_link_sidecar.jsonl` are the only
outputs intended for Phase 5.1r:

```bash
python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/<canonical_pfill_run> \
  --source-json runs/phase51s_local_native_source_acquisition/<phase51s_run_id>/local_native_source.jsonl \
  --source-link-jsonl runs/phase51s_local_native_source_acquisition/<phase51s_run_id>/local_source_link_sidecar.jsonl \
  --run-id <phase51r_run_id>
```

If a forward capture produces a separate redacted mapping from
`phase51s_source_record_sha256`, `source_record_sha256`, or
`redacted_source_record_sha256` to observed `canonical_group_id` / `order_key`,
place it under the manifest `source_links` list. Phase 5.1s stages the mapping
as `local_source_link_sidecar.jsonl`; Phase 5.1r then validates it against
observed P-fill labels and rejects raw IDs, duplicate hashes, ambiguous
mappings, or unsafe flags. The sidecar is only join evidence; it does not infer
maker/taker role or Lighter native-limit pressure.

Phase 5.1t (`tools/phase51t_source_link_sidecar_builder.py`) is the repo-owned
HOLD-only helper for building these sidecars from quarantined local snapshots.
It matches only by existing redacted order/client identifier hashes and emits
`source_links.sanitized.jsonl` for this manifest `source_links` input.

## Safety Contract

Phase 5.1s rejects:

- `http://`, `https://`, or other URI-like source paths.
- `.env` manifests or source files.
- Symlinked manifests or source files.
- Any manifest/source field that looks like an API key, private key, secret,
  password, passphrase, token, JWT, bearer authorization, or session credential.
- Any unsafe true authorization flag.
- Source-link sidecar rows containing unsupported fields, raw venue
  identifiers, duplicate source hashes, non-string source-hash/join fields, or
  missing source-hash/join fields.

Phase 5.1s strips raw venue identifier fields before output, including
order/client/fill/trade IDs and common venue-native ID aliases. It preserves
only non-secret fields needed by Phase 5.1r, such as `canonical_group_id`,
`order_key`, venue, native maker/taker role fields, and native limit-pressure
fields. Source-link sidecar output is stricter: it may contain only a
source-record hash, `canonical_group_id` or `order_key`, and false safety
authorization flags. Phase 5.1r may then join by the redacted source-record
hash when the staged source row lacks direct join fields.

## Boundary

Phase 5.1s does not clear Phase 5.1 matrix blockers by itself. It only stages
source rows and optional source-link sidecars. Blocker reduction requires the
downstream Phase 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i chain to prove that
staged rows join to canonical labels and contain complete explicit
venue-native evidence.

A manifest that stages only source-link sidecars, with no source rows, remains
`phase51s_local_native_source_acquisition_incomplete_source_links_only` because
sidecars alone cannot provide venue-native role or native-limit evidence.

The current required source coverage remains:

- All five venues: explicit venue-native maker/taker role fields for filled
  rows that still lack observed role evidence.
- Lighter: event-time active-order, sendTx, and REST/weighted-request pressure
  fields for native-limit pressure labels.

## Current Board Verdict

`PROMOTE` only for local non-live source staging and downstream evidence reruns.

`HOLD` remains for live orders, canary, capital escalation, risk-limit
relaxation, model training, EV admission, financial claims, and 24/7 readiness.
