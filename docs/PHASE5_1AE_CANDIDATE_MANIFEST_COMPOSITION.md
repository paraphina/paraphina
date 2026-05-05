# Phase 5.1ae Candidate Manifest Composition

Phase 5.1ae is a non-live evidence gate for composing Phase 5.1v candidate
manifests. It exists to remove manual manifest stitching after source-link
materialization.

## Scope

- Tool: `tools/phase51ae_candidate_manifest_compose.py`
- Input: local Phase 5.1v candidate manifest(s), optional local source
  artifacts, optional local source-link artifacts.
- Output: `candidate_manifest.composed.json` for Phase 5.1v validation.
- Status: `HOLD`.

Phase 5.1ae does not infer source links, create native-role evidence, observe
Lighter native-limit pressure, authorize live/canary behavior, authorize capital
changes, relax risk limits, or support economic claims.

## Safety Contract

The composer rejects:

- network paths;
- env files;
- symlinks;
- missing or non-file artifacts;
- unsupported file suffixes;
- secret-shaped fields;
- raw identifier fields;
- unsafe true authorization flags;
- baseline commit mismatches;
- unsupported manifest/source/source-link fields;
- conflicting duplicate source or source-link paths.

Source specs use:

```text
SOURCE_ID=VENUE_ID=PATH
```

Source-link specs use:

```text
SOURCE_LINK_ID=PATH
```

## Current Reference Run

```text
composition:
runs/phase51ae_candidate_manifest_compose/PHASE51AE-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z

Phase 5.1v validation:
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z
```

Result:

```text
native-role targets ready: 73 / 287
native-role targets missing: 214 / 287
Lighter native-limit targets ready: 0 / 3132
Lighter native-limit targets missing: 3132 / 3132
clears_phase51_blockers: false
```

## Resume Pattern

After a validated redacted mapping exists for the current-target wide request
pack:

```bash
python3 tools/phase51ad_source_link_sidecar_materialize.py \
  --request-pack runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z \
  --mapping <validated_redacted_mapping.jsonl> \
  --output-root runs/phase51ad_source_link_sidecar_materialize \
  --run-id <phase51ad_run_id>
```

Then compose all currently ready manifests:

```bash
python3 tools/phase51ae_candidate_manifest_compose.py \
  --candidate-manifest runs/phase51ad_source_link_sidecar_materialize/<phase51ad_run_id>/candidate_manifest_with_materialized_sidecar.json \
  --source phase51x_hyperliquid_forward_native_role_snapshot=hyperliquid=runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z/hyperliquid_forward_native_role_snapshot.jsonl \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z \
  --output-root runs/phase51ae_candidate_manifest_compose \
  --run-id <phase51ae_run_id>
```

If a Phase 5.1ab Lighter native-limit pressure manifest exists, include it with
an additional `--candidate-manifest`.

Finally rerun Phase 5.1v against:

```text
runs/phase51ae_candidate_manifest_compose/<phase51ae_run_id>/candidate_manifest.composed.json
```

Only Phase 5.1v target-ready counts can reduce the Phase 5.1 blocker.
