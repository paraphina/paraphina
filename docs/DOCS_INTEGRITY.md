# Docs Integrity Gate

The Docs Integrity Gate is an institutional-grade mechanism that mechanically prevents
whitepaper drift by enforcing three invariants:

1. **Implemented References Validation** — All `Implemented:` annotations in the whitepaper
   must point to files that actually exist in the repository.
2. **Canonical Spec Hash Lock** — The canonical specification appendix (Part II) is locked
   behind a committed SHA256 baseline, preventing unintended modifications.
3. **Roadmap/Whitepaper Alignment** — Required STATUS markers must be present and match
   between `ROADMAP.md` and `docs/WHITEPAPER.md` Part I.

---

## What Is Checked

### 1. Implemented References

The tool parses `docs/WHITEPAPER.md` for lines containing `Implemented:` and extracts
backticked tokens. A token is treated as a file reference if it:

- Contains `/` (forward slash)
- Ends with a known extension: `.rs`, `.py`, `.md`, `.yml`, `.yaml`, `.toml`

**Symbol annotations** are supported in Rust-style format:
- `path/to/file.rs::SymbolName`
- `path/to/file.rs::Module::function_name`

For symbol annotations, the tool validates:
1. The file exists
2. The **last** identifier (e.g., `function_name`) appears somewhere in the file content
   (simple substring check)

**Error format:**
```
docs/WHITEPAPER.md:<line> missing file: <path>
docs/WHITEPAPER.md:<line> missing symbol: <symbol> in <path>
```

### 2. Canonical Spec Hash Lock

The canonical specification (Part II of the whitepaper) is protected by:

1. A marker line in `docs/WHITEPAPER.md`:
   ```
   <!-- CANONICAL_SPEC_V1_BEGIN -->
   ```
2. A baseline SHA256 hash stored in `docs/CANONICAL_SPEC_V1_SHA256.txt`

The tool computes the SHA256 of all content **after** the marker (normalized to LF line
endings) and compares it against the committed baseline.

If the hashes don't match, the check fails with instructions to either:
- Revert the spec change, or
- Intentionally update the baseline (see below)

### 3. Roadmap/Whitepaper Alignment

The tool enforces that critical status claims are consistent between `ROADMAP.md` and
`docs/WHITEPAPER.md` (Part I only, before the canonical marker).

**Required STATUS markers:**

Both files must contain HTML comment markers for these keys:

| Key | Description |
|-----|-------------|
| `MILESTONE_F` | Status of the global hedge allocator (Milestone F) |
| `CEM` | Status of Cross-Entropy Method adversarial search implementation |

**Marker format:**

```
<!-- STATUS: KEY = VALUE -->
```

- Keys and values are case-insensitive (normalized to uppercase)
- Valid values depend on context: `COMPLETE`, `PARTIAL`, `IMPLEMENTED`, `PLANNED`
- Whitespace around `=` is flexible

**Example markers:**

In `ROADMAP.md`:
```markdown
### Milestone F — Global hedge allocator
<!-- STATUS: MILESTONE_F = COMPLETE -->
```

In `docs/WHITEPAPER.md` Part I:
```markdown
## Hedging (current implementation)
<!-- STATUS: MILESTONE_F = COMPLETE -->
```

**Error format:**

```
ROADMAP.md:<line> missing STATUS marker: MILESTONE_F
docs/WHITEPAPER.md:<line> STATUS mismatch for CEM: ROADMAP=IMPLEMENTED WHITEPAPER=PLANNED
```

**Important:** Only Part I of `WHITEPAPER.md` (before `<!-- CANONICAL_SPEC_V1_BEGIN -->`) is
scanned for status markers. Part II is hash-locked and cannot be modified.

---

## How to Run Locally

### Check mode (default)

```bash
python3 tools/check_docs_integrity.py
```

Exit codes:
- `0` — OK: all checks passed
- `1` — Internal error (script bug, I/O error)
- `2` — Integrity violation (missing file, missing symbol, hash mismatch, or status mismatch)

### Update canonical hash

If you **intentionally** modify the canonical spec (Part II), update the baseline:

```bash
python3 tools/check_docs_integrity.py --update-canonical-hash
```

This will:
1. Verify all `Implemented:` references first (must pass)
2. Compute the new canonical hash
3. Write it to `docs/CANONICAL_SPEC_V1_SHA256.txt`
4. Exit with code `0` on success

**Important:** Only run `--update-canonical-hash` when you have intentionally modified
the canonical specification. This action should be reviewed carefully in PRs.

---

## CI Enforcement

The Docs Integrity Gate runs automatically on all pull requests and pushes to `main`
via the GitHub Actions workflow:

**Workflow name:** `docs-integrity`

**File:** `.github/workflows/docs_integrity.yml`

The workflow:
1. Runs the Unicode controls scanner
2. Runs the Docs Integrity Gate
3. Runs all unit tests

If any check fails, the PR cannot be merged.

---

## Troubleshooting

### "missing file: path/to/file.rs"

The referenced file doesn't exist. Either:
- Fix the path in the `Implemented:` annotation
- Create the missing file
- Remove the annotation if the feature is no longer implemented

### "missing symbol: Symbol in path/to/file.rs"

The symbol was not found in the file. Either:
- Fix the symbol name in the annotation
- Add the symbol to the file
- Remove the symbol annotation if it's no longer relevant

### "Canonical spec hash mismatch!"

The canonical specification has been modified. If this was:
- **Unintentional:** Revert the changes to Part II
- **Intentional:** Run `--update-canonical-hash` and include the updated baseline in your PR

### "Baseline hash file missing"

The baseline hash file `docs/CANONICAL_SPEC_V1_SHA256.txt` doesn't exist. Run:

```bash
python3 tools/check_docs_integrity.py --update-canonical-hash
```

### "missing STATUS marker: KEY"

A required STATUS marker is missing from one of the files. Add the marker:

```markdown
<!-- STATUS: KEY = VALUE -->
```

Required markers (both files must have):
- `MILESTONE_F` — Hedge allocator milestone status (COMPLETE/PARTIAL)
- `CEM` — CEM implementation status (IMPLEMENTED/PARTIAL/PLANNED)

### "STATUS mismatch for KEY: ROADMAP=X WHITEPAPER=Y"

The STATUS marker values don't match between files. Either:
- Update `ROADMAP.md` to use value `Y`
- Update `docs/WHITEPAPER.md` Part I to use value `X`
- Investigate which value is correct based on implementation evidence

---

## Design Rationale

This gate exists to:

1. **Prevent documentation rot** — References to code must stay synchronized
2. **Protect the spec** — The canonical specification is a contract; changes must be deliberate
3. **Prevent status drift** — Milestone/feature status must be consistent across documents
4. **Enable CI enforcement** — Failures are actionable with exact file/line/message
5. **Maintain determinism** — All output is sorted and reproducible

