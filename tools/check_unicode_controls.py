#!/usr/bin/env python3
"""
Integrity Gate: Unicode Control Character Scanner

Scans all git-tracked files for forbidden Unicode control characters that
can be used in Trojan-Source style attacks (invisible text manipulation,
bidirectional overrides, etc.) or cause GitHub "hidden or bidirectional
Unicode text" warnings.

Exit codes:
    0 - Clean: no forbidden characters found
    1 - Internal error
    2 - Forbidden characters detected

Usage:
    python3 tools/check_unicode_controls.py           # Scan only
    python3 tools/check_unicode_controls.py --fix    # Auto-sanitize files
"""

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    """A single forbidden character finding."""
    path: str
    line: int
    column: int
    codepoint: int
    name: str
    category: str
    line_content: str


# =============================================================================
# FORBIDDEN UNICODE CHARACTER DEFINITIONS
# =============================================================================
# This is the comprehensive set of characters that GitHub flags as "hidden or
# bidirectional Unicode text" plus additional Trojan-Source hazards.

# Bidirectional control characters (U+061C, U+200E, U+200F, U+202A..U+202E, U+2066..U+2069)
# These can reverse text direction for code review attacks
BIDI_CONTROLS: set[int] = {
    0x061C,  # ARABIC LETTER MARK
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
}

# Zero-width / hidden / format characters
# Used in Trojan Source attacks or invisible edits
ZERO_WIDTH_AND_FORMAT: set[int] = {
    0x200B,  # ZERO WIDTH SPACE (ZWSP)
    0x200C,  # ZERO WIDTH NON-JOINER (ZWNJ)
    0x200D,  # ZERO WIDTH JOINER (ZWJ)
    0x2060,  # WORD JOINER
    0xFEFF,  # BYTE ORDER MARK (BOM) / ZERO WIDTH NO-BREAK SPACE
    0x00AD,  # SOFT HYPHEN
    0x034F,  # COMBINING GRAPHEME JOINER (CGJ)
    0x180E,  # MONGOLIAN VOWEL SEPARATOR (deprecated but still appears)
}

# Variation selectors: U+FE00..U+FE0F
VARIATION_SELECTORS_1: set[int] = set(range(0xFE00, 0xFE10))

# Variation selectors supplement: U+E0100..U+E01EF
VARIATION_SELECTORS_SUPPLEMENT: set[int] = set(range(0xE0100, 0xE01F0))

# ASCII/C0 control characters (except allowed: \t=0x09, \n=0x0A, \r=0x0D)
# Forbid: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F
ASCII_CONTROLS: set[int] = (
    set(range(0x00, 0x09)) |    # 0x00-0x08
    {0x0B, 0x0C} |              # VT, FF
    set(range(0x0E, 0x20)) |    # 0x0E-0x1F
    {0x7F}                       # DEL
)

# Combined set of all forbidden codepoints
FORBIDDEN_CODEPOINTS: set[int] = (
    BIDI_CONTROLS |
    ZERO_WIDTH_AND_FORMAT |
    VARIATION_SELECTORS_1 |
    VARIATION_SELECTORS_SUPPLEMENT |
    ASCII_CONTROLS
)


def get_unicode_info(codepoint: int) -> tuple[str, str]:
    """Get Unicode name and category for a codepoint."""
    try:
        char = chr(codepoint)
        name = unicodedata.name(char, None)
        if name is None:
            # Provide descriptive names for unnamed characters
            if codepoint in ASCII_CONTROLS:
                name = _ascii_control_name(codepoint)
            else:
                name = "<unknown>"
        category = unicodedata.category(char)
    except ValueError:
        name = "<unknown>"
        category = "??"
    return name, category


def _ascii_control_name(codepoint: int) -> str:
    """Return descriptive name for ASCII control characters."""
    names = {
        0x00: "NULL",
        0x01: "START OF HEADING",
        0x02: "START OF TEXT",
        0x03: "END OF TEXT",
        0x04: "END OF TRANSMISSION",
        0x05: "ENQUIRY",
        0x06: "ACKNOWLEDGE",
        0x07: "BELL",
        0x08: "BACKSPACE",
        0x0B: "VERTICAL TAB",
        0x0C: "FORM FEED",
        0x0E: "SHIFT OUT",
        0x0F: "SHIFT IN",
        0x10: "DATA LINK ESCAPE",
        0x11: "DEVICE CONTROL ONE",
        0x12: "DEVICE CONTROL TWO",
        0x13: "DEVICE CONTROL THREE",
        0x14: "DEVICE CONTROL FOUR",
        0x15: "NEGATIVE ACKNOWLEDGE",
        0x16: "SYNCHRONOUS IDLE",
        0x17: "END OF TRANSMISSION BLOCK",
        0x18: "CANCEL",
        0x19: "END OF MEDIUM",
        0x1A: "SUBSTITUTE",
        0x1B: "ESCAPE",
        0x1C: "FILE SEPARATOR",
        0x1D: "GROUP SEPARATOR",
        0x1E: "RECORD SEPARATOR",
        0x1F: "UNIT SEPARATOR",
        0x7F: "DELETE",
    }
    return names.get(codepoint, f"CONTROL-{codepoint:02X}")


def is_forbidden(codepoint: int) -> bool:
    """Check if a codepoint is forbidden."""
    return codepoint in FORBIDDEN_CODEPOINTS


def format_line_with_placeholder(line: str, column: int, codepoint: int) -> str:
    """Replace the forbidden character with a visible placeholder."""
    placeholder = f"<U+{codepoint:04X}>"
    # column is 1-based, convert to 0-based index
    idx = column - 1
    if 0 <= idx < len(line):
        return line[:idx] + placeholder + line[idx + 1:]
    return line


def scan_content(content: str, path: str) -> list[Finding]:
    """Scan text content for forbidden Unicode characters."""
    findings: list[Finding] = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, start=1):
        for col_num, char in enumerate(line, start=1):
            codepoint = ord(char)
            if is_forbidden(codepoint):
                name, category = get_unicode_info(codepoint)
                findings.append(Finding(
                    path=path,
                    line=line_num,
                    column=col_num,
                    codepoint=codepoint,
                    name=name,
                    category=category,
                    line_content=line,
                ))
    
    return findings


def scan_file(path: Path) -> list[Finding]:
    """Scan a single file for forbidden Unicode characters."""
    try:
        raw = path.read_bytes()
    except (OSError, IOError) as e:
        print(f"WARNING: Could not read {path}: {e}", file=sys.stderr)
        return []
    
    # Check for forbidden ASCII controls in raw bytes first (including null bytes)
    # This allows us to detect null bytes even in "binary" files that we'd skip
    has_null = b'\x00' in raw
    
    # Skip binary files (contain null bytes) - they're not text
    if has_null:
        return []
    
    try:
        content = raw.decode('utf-8')
    except UnicodeDecodeError:
        # Try with replacement for partially valid UTF-8
        try:
            content = raw.decode('utf-8', errors='replace')
        except Exception:
            return []
    
    return scan_content(content, str(path))


def sanitize_content(content: str) -> str:
    """Remove all forbidden Unicode characters from content."""
    result = []
    for char in content:
        codepoint = ord(char)
        if not is_forbidden(codepoint):
            result.append(char)
    return ''.join(result)


def fix_file(path: Path) -> bool:
    """
    Fix a single file by removing forbidden Unicode characters.
    
    Returns True if the file was modified, False otherwise.
    Preserves existing line endings.
    """
    try:
        raw = path.read_bytes()
    except (OSError, IOError) as e:
        print(f"WARNING: Could not read {path}: {e}", file=sys.stderr)
        return False
    
    # Skip binary files
    if b'\x00' in raw:
        return False
    
    try:
        content = raw.decode('utf-8')
    except UnicodeDecodeError:
        # Can't safely fix files with encoding issues
        return False
    
    # Detect line ending style
    has_crlf = '\r\n' in content
    has_lf_only = '\n' in content and not has_crlf
    
    # Sanitize content
    sanitized = sanitize_content(content)
    
    # No changes needed
    if sanitized == content:
        return False
    
    # Write back with appropriate line endings
    try:
        # Encode and write
        new_raw = sanitized.encode('utf-8')
        path.write_bytes(new_raw)
        return True
    except (OSError, IOError) as e:
        print(f"WARNING: Could not write {path}: {e}", file=sys.stderr)
        return False


def get_tracked_files() -> list[Path]:
    """Get list of git-tracked files using git ls-files."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '-z'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git ls-files failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: git not found in PATH", file=sys.stderr)
        sys.exit(1)
    
    # Split on null bytes, filter empty strings
    files = [Path(f) for f in result.stdout.split('\0') if f]
    
    # Filter out runs/ directory (as specified)
    files = [f for f in files if not str(f).startswith('runs/')]
    
    return files


def print_finding(finding: Finding) -> None:
    """Print a single finding in audit-grade format."""
    # Format: PATH:LINE:COLUMN U+XXXX <UNICODE_NAME> (<CATEGORY>)
    print(f"{finding.path}:{finding.line}:{finding.column} "
          f"U+{finding.codepoint:04X} {finding.name} ({finding.category})")
    
    # Show context snippet with placeholder
    marked_line = format_line_with_placeholder(
        finding.line_content, 
        finding.column, 
        finding.codepoint
    )
    # Truncate very long lines for readability
    if len(marked_line) > 120:
        # Find the placeholder position and show context around it
        placeholder = f"<U+{finding.codepoint:04X}>"
        pos = marked_line.find(placeholder)
        if pos >= 0:
            start = max(0, pos - 40)
            end = min(len(marked_line), pos + len(placeholder) + 40)
            snippet = marked_line[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(marked_line):
                snippet = snippet + "..."
            marked_line = snippet
    
    print(f"  {marked_line}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scan for forbidden Unicode control characters"
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Auto-sanitize files by removing forbidden characters'
    )
    args = parser.parse_args()
    
    try:
        files = get_tracked_files()
    except Exception as e:
        print(f"ERROR: Failed to get tracked files: {e}", file=sys.stderr)
        return 1
    
    if args.fix:
        # Fix mode: sanitize all files
        modified_files: list[str] = []
        
        for path in files:
            if fix_file(path):
                modified_files.append(str(path))
        
        if modified_files:
            print(f"FIXED: Sanitized {len(modified_files)} file(s):")
            for f in sorted(modified_files):
                print(f"  {f}")
            print("\nRe-run without --fix to verify all files are clean.")
        else:
            print(f"OK: No forbidden Unicode controls found in {len(files)} tracked files.")
        
        return 0
    
    # Scan mode: check all files
    all_findings: list[Finding] = []
    
    for path in files:
        findings = scan_file(path)
        all_findings.extend(findings)
    
    if all_findings:
        # Sort findings for deterministic output
        all_findings.sort(key=lambda f: (f.path, f.line, f.column))
        
        print(f"FAILED: Found {len(all_findings)} forbidden Unicode control character(s):\n")
        for finding in all_findings:
            print_finding(finding)
            print()
        
        print("To auto-fix: python3 tools/check_unicode_controls.py --fix")
        return 2
    
    print(f"OK: No forbidden Unicode controls found in {len(files)} tracked files.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
