# Repository Integrity Gate

This document describes the repository's integrity checks designed to prevent
Trojan-Source style attacks and ensure code review integrity.

## Overview

The repository includes an automated scanner that detects forbidden Unicode
control characters that could be used to hide malicious code or manipulate
how code appears during review.

## Why These Characters Are Banned

### Trojan Source Attacks

The [Trojan Source](https://trojansource.codes/) vulnerability (CVE-2021-42574)
demonstrates how Unicode bidirectional (Bidi) control characters can be used to
make code appear different to human reviewers than what the compiler sees.

For example, an attacker could use RIGHT-TO-LEFT OVERRIDE (U+202E) to visually
reorder code, making malicious logic appear as comments or benign code.

### Review Integrity

Zero-width and invisible characters can:

- Hide malicious identifiers that look identical to legitimate ones
- Insert invisible text that affects program behavior
- Bypass code review by making dangerous code invisible

### Banned Character Categories

| Category | Codepoints | Reason |
|----------|------------|--------|
| Bidi Controls | U+202A-U+202E, U+2066-U+2069 | Can reverse text display direction |
| Direction Marks | U+200E, U+200F, U+061C | Can affect text ordering |
| Zero-Width | U+200B, U+200C, U+200D, U+FEFF | Invisible characters that affect parsing |
| Soft Hyphen | U+00AD | Invisible in most contexts |

## Running the Scanner Locally

To check the repository for forbidden Unicode control characters:

```bash
python3 tools/check_unicode_controls.py
```

### Exit Codes

- **0**: Clean - no forbidden characters found
- **1**: Internal error (e.g., git not available)
- **2**: Forbidden characters detected

### Example Output (Clean)

```
OK: no forbidden Unicode controls found in 423 tracked files.
```

### Example Output (Violation Detected)

```
FAILED: Found 1 forbidden Unicode control character(s):

src/example.py:15:23 U+202E RIGHT-TO-LEFT OVERRIDE (Cf)
  # Comment with⟦U+202E⟧hidden text
```

The scanner shows:
- File path, line number, and column
- The Unicode codepoint and its official name
- The line content with the forbidden character replaced by a visible placeholder

## CI Enforcement

The Unicode control character scanner runs automatically in CI as part of the
Phase AB smoke workflow:

**Location**: `.github/workflows/phase_ab_smoke.yml`

The scanner runs early in the `phase-ab-smoke` job, after checkout and Python
setup but before any build steps. If any forbidden characters are detected,
the workflow fails immediately.

```yaml
- name: Check for forbidden Unicode control characters
  run: python3 tools/check_unicode_controls.py
```

## Fixing Violations

If the scanner detects forbidden characters:

1. **Identify the file and location** from the scanner output
2. **Open the file in a hex editor** or Unicode-aware editor to see the actual bytes
3. **Remove the forbidden character(s)**
4. **Re-run the scanner** to verify the fix

### Common Scenarios

- **Copied from web**: Text copied from websites may contain invisible formatting
- **IDEs with RTL support**: Some editors insert Bidi marks when editing mixed-direction text
- **BOM in middle of file**: Usually from concatenating files or copy-paste

## Technical Details

### Scanner Implementation

The scanner (`tools/check_unicode_controls.py`):

- Uses only Python standard library (no dependencies)
- Scans git-tracked files via `git ls-files -z`
- Skips binary files (detected by presence of null bytes)
- Skips the `runs/` directory (runtime artifacts)
- Produces deterministic, sorted output

### Allowed Unicode

The scanner does NOT flag:

- Normal printable Unicode (accented characters, CJK, etc.)
- Emoji
- Standard whitespace (spaces, tabs, newlines)
- BOMs at the very start of a file (though mid-file BOMs are flagged)

Only the specific control characters listed above are forbidden.

## References

- [Trojan Source: Invisible Vulnerabilities](https://trojansource.codes/)
- [CVE-2021-42574](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-42574)
- [Unicode Bidirectional Algorithm](https://unicode.org/reports/tr9/)
- [GitHub's hidden character warnings](https://github.blog/changelog/2021-10-31-warning-about-bidirectional-unicode-text/)

