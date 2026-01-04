"""
Unit tests for the Unicode control character scanner.

Tests the scanner's ability to detect forbidden Unicode control characters
(Trojan-Source style attacks) while allowing legitimate Unicode.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add tools directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from check_unicode_controls import (
    FORBIDDEN_CODEPOINTS,
    Finding,
    format_line_with_placeholder,
    get_unicode_info,
    scan_content,
    scan_file,
)


class TestForbiddenCodepoints(unittest.TestCase):
    """Test that all expected forbidden codepoints are defined."""
    
    def test_bidi_controls_are_forbidden(self):
        """Bidi control characters should be in the forbidden set."""
        bidi_controls = [
            0x202A,  # LEFT-TO-RIGHT EMBEDDING
            0x202B,  # RIGHT-TO-LEFT EMBEDDING
            0x202C,  # POP DIRECTIONAL FORMATTING
            0x202D,  # LEFT-TO-RIGHT OVERRIDE
            0x202E,  # RIGHT-TO-LEFT OVERRIDE
            0x2066,  # LEFT-TO-LEFT ISOLATE
            0x2067,  # RIGHT-TO-LEFT ISOLATE
            0x2068,  # FIRST STRONG ISOLATE
            0x2069,  # POP DIRECTIONAL ISOLATE
        ]
        for cp in bidi_controls:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS, 
                         f"U+{cp:04X} should be forbidden")
    
    def test_direction_marks_are_forbidden(self):
        """Direction mark characters should be in the forbidden set."""
        direction_marks = [
            0x200E,  # LEFT-TO-RIGHT MARK
            0x200F,  # RIGHT-TO-LEFT MARK
            0x061C,  # ARABIC LETTER MARK
        ]
        for cp in direction_marks:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
    
    def test_zero_width_chars_are_forbidden(self):
        """Zero-width/invisible characters should be in the forbidden set."""
        zero_width = [
            0x200B,  # ZERO WIDTH SPACE
            0x200C,  # ZERO WIDTH NON-JOINER
            0x200D,  # ZERO WIDTH JOINER
            0xFEFF,  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
        ]
        for cp in zero_width:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
    
    def test_soft_hyphen_is_forbidden(self):
        """Soft hyphen should be in the forbidden set."""
        self.assertIn(0x00AD, FORBIDDEN_CODEPOINTS,
                     "U+00AD (SOFT HYPHEN) should be forbidden")


class TestScanContent(unittest.TestCase):
    """Test scanning text content for forbidden characters."""
    
    def test_clean_ascii_content(self):
        """Clean ASCII content should produce no findings."""
        content = "def hello():\n    print('Hello, World!')\n"
        findings = scan_content(content, "test.py")
        self.assertEqual(len(findings), 0)
    
    def test_clean_unicode_content(self):
        """Normal Unicode characters should NOT be flagged."""
        # These are all legitimate Unicode that should pass
        content = "# ✓ Check mark is OK\n# 你好 Chinese is OK\n# Ñ Accented is OK\n"
        findings = scan_content(content, "test.py")
        self.assertEqual(len(findings), 0,
                        "Normal Unicode characters should not be flagged")
    
    def test_emoji_not_flagged(self):
        """Emoji should not be flagged as forbidden."""
        content = "message = '🎉 Success! 🚀'\n"
        findings = scan_content(content, "test.py")
        self.assertEqual(len(findings), 0,
                        "Emoji should not be flagged")
    
    def test_detect_bidi_override(self):
        """U+202E (RIGHT-TO-LEFT OVERRIDE) should be detected."""
        # Insert RLO character
        rlo = chr(0x202E)
        content = f"# Comment with{rlo}hidden text\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x202E)
        self.assertEqual(findings[0].line, 1)
        self.assertEqual(findings[0].column, 15)  # Position of RLO
        self.assertIn("RIGHT-TO-LEFT OVERRIDE", findings[0].name)
    
    def test_detect_zero_width_space(self):
        """U+200B (ZERO WIDTH SPACE) should be detected."""
        zws = chr(0x200B)
        content = f"variable{zws}name = 42\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x200B)
    
    def test_detect_soft_hyphen(self):
        """U+00AD (SOFT HYPHEN) should be detected."""
        shy = chr(0x00AD)
        content = f"function{shy}name()\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x00AD)
    
    def test_detect_multiple_violations(self):
        """Multiple forbidden characters should all be detected."""
        rlo = chr(0x202E)
        zws = chr(0x200B)
        content = f"line1{rlo}bad\nline2{zws}also bad\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 2)
        codepoints = {f.codepoint for f in findings}
        self.assertIn(0x202E, codepoints)
        self.assertIn(0x200B, codepoints)
    
    def test_correct_line_and_column(self):
        """Line and column numbers should be accurate."""
        rlo = chr(0x202E)
        content = f"line1\nABC{rlo}DEF\nline3\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line, 2)
        self.assertEqual(findings[0].column, 4)  # After "ABC"
    
    def test_detect_bom_not_at_start(self):
        """BOM (U+FEFF) should be detected when not at file start."""
        bom = chr(0xFEFF)
        content = f"normal start\nwith{bom}bom in middle\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0xFEFF)


class TestScanFile(unittest.TestCase):
    """Test scanning actual files."""
    
    def test_scan_clean_file(self):
        """Scanning a clean file should produce no findings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                          delete=False) as f:
            f.write("def clean():\n    return True\n")
            f.flush()
            path = Path(f.name)
        
        try:
            findings = scan_file(path)
            self.assertEqual(len(findings), 0)
        finally:
            path.unlink()
    
    def test_scan_file_with_forbidden_char(self):
        """Scanning a file with forbidden character should detect it."""
        rlo = chr(0x202E)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            f.write(f"# Hidden{rlo}text\ndef func(): pass\n")
            f.flush()
            path = Path(f.name)
        
        try:
            findings = scan_file(path)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].codepoint, 0x202E)
        finally:
            path.unlink()
    
    def test_skip_binary_file(self):
        """Binary files (containing null bytes) should be skipped."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin',
                                          delete=False) as f:
            # Write binary content with null bytes
            f.write(b'\x00\x01\x02\x03binary content')
            f.flush()
            path = Path(f.name)
        
        try:
            findings = scan_file(path)
            self.assertEqual(len(findings), 0,
                           "Binary files should be skipped")
        finally:
            path.unlink()


class TestFormatLineWithPlaceholder(unittest.TestCase):
    """Test the line formatting with placeholder."""
    
    def test_replace_character_with_placeholder(self):
        """The forbidden character should be replaced with placeholder."""
        rlo = chr(0x202E)
        line = f"ABC{rlo}DEF"
        result = format_line_with_placeholder(line, 4, 0x202E)
        
        self.assertEqual(result, "ABC⟦U+202E⟧DEF")
        self.assertNotIn(rlo, result)
    
    def test_placeholder_at_start(self):
        """Placeholder should work at the start of a line."""
        zws = chr(0x200B)
        line = f"{zws}start"
        result = format_line_with_placeholder(line, 1, 0x200B)
        
        self.assertEqual(result, "⟦U+200B⟧start")
    
    def test_placeholder_at_end(self):
        """Placeholder should work at the end of a line."""
        shy = chr(0x00AD)
        line = f"end{shy}"
        result = format_line_with_placeholder(line, 4, 0x00AD)
        
        self.assertEqual(result, "end⟦U+00AD⟧")


class TestGetUnicodeInfo(unittest.TestCase):
    """Test Unicode name and category retrieval."""
    
    def test_get_info_for_rlo(self):
        """Should get correct info for RLO character."""
        name, category = get_unicode_info(0x202E)
        self.assertIn("RIGHT-TO-LEFT OVERRIDE", name)
        self.assertEqual(category, "Cf")  # Format character
    
    def test_get_info_for_soft_hyphen(self):
        """Should get correct info for soft hyphen."""
        name, category = get_unicode_info(0x00AD)
        self.assertIn("SOFT HYPHEN", name)
        self.assertEqual(category, "Cf")  # Format character


class TestFindingNamedTuple(unittest.TestCase):
    """Test the Finding named tuple."""
    
    def test_finding_has_all_fields(self):
        """Finding should have all required fields."""
        finding = Finding(
            path="test.py",
            line=10,
            column=5,
            codepoint=0x202E,
            name="RIGHT-TO-LEFT OVERRIDE",
            category="Cf",
            line_content="test line",
        )
        
        self.assertEqual(finding.path, "test.py")
        self.assertEqual(finding.line, 10)
        self.assertEqual(finding.column, 5)
        self.assertEqual(finding.codepoint, 0x202E)
        self.assertEqual(finding.name, "RIGHT-TO-LEFT OVERRIDE")
        self.assertEqual(finding.category, "Cf")
        self.assertEqual(finding.line_content, "test line")


if __name__ == '__main__':
    unittest.main()

