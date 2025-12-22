# Evidence Pack v1 Specification

## Purpose

The Evidence Pack defines an **audit boundary** for `sim_eval` outputs, providing:

1. **Provenance** — tracing outputs back to source code, configuration, and environment
2. **Integrity** — cryptographic hashes for all artifacts enabling verification

This specification establishes a reproducible, machine-verifiable bundle that accompanies every simulation evaluation run.

---

## Directory Layout

All evidence pack files reside under `<output_root>/evidence_pack/`:

```
<output_root>/
└── evidence_pack/
    ├── manifest.json      # Structured metadata + artifact listing
    ├── suite.yaml         # Copy of input suite configuration
    └── SHA256SUMS         # Checksums for verification
```

---

## Determinism Requirements

### Relative Paths Only

- All paths in `manifest.json` MUST be relative to `<output_root>`.
- Absolute paths are forbidden; they break portability and reproducibility.

### Stable Ordering

- All lists in `manifest.json` (e.g., `artifacts`) MUST be sorted by `path` (lexicographic, ascending).
- This ensures byte-identical manifests across runs with identical inputs.

### Atomic Writes

- All evidence pack files MUST be written atomically (temp file + rename).
- This prevents partial/corrupt files if a run is interrupted.

---

## manifest.json Specification

### Required Fields

All keys are required. Values may be `null` when best-effort retrieval fails.

```json
{
  "evidence_pack_schema_version": "v1",
  "generated_at_unix_ms": 1703260800000,
  "paraphina_version": "0.1.0",
  "repository": {
    "git_commit": "a1b2c3d4e5f6...",
    "cargo_lock_sha256": "sha256:...",
    "sim_output_schema_sha256": "sha256:..."
  },
  "suite": {
    "source_path": "suites/stress_test.yaml",
    "copied_to": "evidence_pack/suite.yaml",
    "sha256": "sha256:..."
  },
  "artifacts": [
    { "path": "evidence_pack/suite.yaml", "sha256": "sha256:..." },
    { "path": "results/run_001.json", "sha256": "sha256:..." },
    { "path": "results/run_002.json", "sha256": "sha256:..." }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `evidence_pack_schema_version` | string | Always `"v1"` for this spec |
| `generated_at_unix_ms` | integer | Unix timestamp (milliseconds) when pack was generated |
| `paraphina_version` | string | Paraphina version from `Cargo.toml` |
| `repository.git_commit` | string\|null | Git commit SHA (null if not in git repo or dirty) |
| `repository.cargo_lock_sha256` | string\|null | SHA256 of `Cargo.lock` (null if missing) |
| `repository.sim_output_schema_sha256` | string\|null | SHA256 of sim output schema definition (null if N/A) |
| `suite.source_path` | string | Original path to suite YAML (relative to repo root) |
| `suite.copied_to` | string | Always `"evidence_pack/suite.yaml"` |
| `suite.sha256` | string | SHA256 of the suite file |
| `artifacts` | array | List of `{ path, sha256 }` objects for all output files |

### Artifacts Array

- Each entry is `{ "path": "<relative_path>", "sha256": "<hash>" }`.
- Paths are relative to `<output_root>`.
- Sorted lexicographically by `path`.
- **MUST NOT include `evidence_pack/manifest.json`** (self-referential hash is undefined).

---

## SHA256SUMS Contract

The `evidence_pack/SHA256SUMS` file enables standard verification via `sha256sum -c`.

### Contents

The file contains SHA256 lines for:

1. `evidence_pack/manifest.json`
2. `evidence_pack/suite.yaml`
3. Each `artifacts[].path` from the manifest

### Format

Standard `sha256sum` output format:

```
<hash>  evidence_pack/manifest.json
<hash>  evidence_pack/suite.yaml
<hash>  results/run_001.json
<hash>  results/run_002.json
```

### Verification

From `<output_root>`, verify all checksums with:

```bash
(cd <output_root> && sha256sum -c evidence_pack/SHA256SUMS)
```

All lines should report `OK`. Any mismatch indicates tampering or corruption.

---

## Non-Goals (Explicit)

The following are explicitly **out of scope** for Evidence Pack v1:

### No Signing/Attestation

- Cryptographic signatures (GPG, Sigstore, etc.) are reserved for a future version.
- V1 provides integrity (hashes) but not non-repudiation (signatures).

### No Full Environment Capture

- System dependencies, OS version, CPU architecture, etc. are not captured.
- Full reproducibility requires additional tooling (e.g., Nix, Docker image hashes).
- Reserved for future versions.

---

## Implementation Notes

### Generation Workflow

1. Run simulation suite, collecting output artifacts.
2. Copy suite YAML to `evidence_pack/suite.yaml`.
3. Compute SHA256 for all artifacts.
4. Build `manifest.json` with sorted artifacts list.
5. Write `manifest.json` atomically.
6. Generate `SHA256SUMS` including `manifest.json`.
7. Write `SHA256SUMS` atomically.

### Error Handling

- If `git_commit` cannot be determined (dirty tree, not a repo), set to `null`.
- If `Cargo.lock` is missing, set `cargo_lock_sha256` to `null`.
- Missing optional metadata should not fail pack generation.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2024-12 | Initial specification |
