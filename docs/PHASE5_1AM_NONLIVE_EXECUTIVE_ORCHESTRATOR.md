# Phase 5.1am Non-Live Executive Orchestrator

Status: `HOLD` for live/canary/capital/risk. Continue using Phase 5.1am only
for structured non-live workflow control.

Tool:

```text
tools/phase51am_nonlive_executive_orchestrator.py
```

Test:

```text
tests/test_phase51am_nonlive_executive_orchestrator.py
```

## Purpose

Phase 5.1am is the executable control point for autonomous Phase 5.1/V2
orchestration. It reads the latest Phase 5.1ak blocker state plus any supplied
source-owner artifacts, classifies the next admissible route, and emits an
auditable route ledger, subagent work packets, a workflow optimization ledger,
source-owner intake status/template files, a subagent prompt pack, and a
source-owner request.

It does not collect private source data, run venue APIs, use credentials, place
orders, cancel orders, call `sendTx`, call `sendTxBatch`, train models, admit
EV, or infer missing source truth.

## Route Priority

Phase 5.1am selects the first ready route in this order:

1. Real Phase 5.1al forward-refresh pack.
2. Validated redacted Phase 5.1ad source-link mapping.
3. Materially new directly target-linkable Phase 5.1aj private/native rows.
4. Complete sanitized Phase 5.1ab Lighter event-time pressure rows.

If no route is ready, Phase 5.1am keeps `gate_status=HOLD` but sets
`autonomous_continuation_status=source_owner_and_subagent_work_packets_emitted`.
That is the required autonomous continuation behavior: the blocker is not
cleared, but the next work is structured and dispatchable.

## Continuous Optimization

Each run emits `workflow_optimization_ledger.jsonl`. These records tell the
executive orchestrator how to keep optimizing the board while work is in
progress:

- Re-run Phase 5.1am after every blocker task or new source-owner artifact.
- Compare each run against the previous Phase 5.1am summary and flag unchanged
  no-route loops before more local mining is dispatched.
- Resize the board around the selected route, using the smallest active team
  plus an independent auditor.
- Suppress duplicate retrospective mining unless a material-change reason
  exists.
- Preserve route priority when multiple routes are ready.
- Audit after each packet completion or downstream gate, then reclassify.

## Artifacts

Each run writes:

```text
runs/phase51am_nonlive_executive_orchestrator/<RUN_ID>/
  phase51am_nonlive_executive_orchestrator_summary.json
  phase51am_route_decision_ledger.jsonl
  subagent_work_packets.jsonl
  subagent_prompt_pack/
    index.json
    01_<packet>.md
  workflow_optimization_ledger.jsonl
  source_owner_intake_status.json
  source_owner_intake_manifest.template.json
  source_owner_request.md
  evidence_pack/artifact_index.json
  manifest.json
```

The summary and manifest carry the standard Phase 5.1 safety envelope:
`gate_status=HOLD`, `clears_phase51_blockers=false`, `no_live_flag=true`, and
all live/canary/capital/risk/model/EV/financial authorization flags false.

## CLI

Default current-state classification:

```text
python3 tools/phase51am_nonlive_executive_orchestrator.py \
  --run-id <RUN_ID>
```

Explicit artifact intake:

```text
python3 tools/phase51am_nonlive_executive_orchestrator.py \
  --phase51al-summary <path> \
  --validated-mapping <path> \
  --phase51aj-source-json venue=<path> \
  --phase51ab-pressure-jsonl <path> \
  --run-id <RUN_ID>
```

Structured source-owner intake:

```text
python3 tools/phase51am_nonlive_executive_orchestrator.py \
  --source-owner-intake-manifest <path> \
  --run-id <RUN_ID>
```

The intake manifest is a local JSON file with these fields:

```json
{
  "schema_version": 1,
  "material_change_reason": "describe_the_new_source_owner_truth_or_material_change",
  "phase51al_summaries": [],
  "validated_mappings": [],
  "phase51aj_source_json": [],
  "phase51ab_pressure_jsonls": [],
  "no_live_flag": true,
  "approved_for_live": false,
  "live_orders_allowed": false
}
```

`phase51aj_source_json` uses the same `venue=/local/path` strings accepted by
Phase 5.1aj. Other artifact fields are local paths. Network paths, `.env`
paths, symlinks, unsafe flags, secret-shaped fields, and raw identifier fields
fail closed.

Only supply the artifact arguments that exist. Phase 5.1am validates local path
safety and fails closed on network paths, `.env` paths, symlinks, unsafe true
flags, secret-shaped fields, and raw identifier fields.

For deterministic resume audits, an explicit previous run may be supplied:

```text
python3 tools/phase51am_nonlive_executive_orchestrator.py \
  --previous-phase51am-summary <path> \
  --run-id <RUN_ID>
```

## Acceptance

Accepted:

- The tool emits a route ledger for all four admissible route families.
- The tool emits a workflow optimization ledger on every run.
- The tool renders each work packet into a prompt file under
  `subagent_prompt_pack/`.
- The tool emits a source-owner intake template/status file on every run.
- The tool can route source-owner artifacts from a structured intake manifest.
- The tool records `phase51am_delta` when a previous Phase 5.1am summary is
  available.
- Fixture-only Phase 5.1al packs do not select the forward-refresh route.
- Real non-fixture Phase 5.1al packs with required artifacts select
  `forward_refresh` for Phase 5.1ak validation.
- No-route HOLD emits subagent/source-owner work packets instead of ending at
  an unstructured halt.

Rejected:

- Treating Phase 5.1am as evidence that clears the blocker.
- Treating generated work packets as authorization for live/canary/capital/risk
  action.
- Repeating exhausted retrospective mining without a materially new artifact,
  source surface, target window, account activity, auth material, or source
  semantics.
