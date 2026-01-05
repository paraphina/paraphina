"""
Unit tests for the Unicode control character scanner.

Tests the scanner's ability to detect forbidden Unicode control characters
(Trojan-Source style attacks, GitHub hidden-unicode warnings) while allowing
legitimate Unicode.

Tests are hermetic: use temp dirs/files; don't depend on git state.
"""

import tempfile
import unittest
from pathlib import Path

# Import from tools package
from tools.check_unicode_controls import (
    ASCII_CONTROLS,
    BIDI_CONTROLS,
    FORBIDDEN_CODEPOINTS,
    VARIATION_SELECTORS_1,
    VARIATION_SELECTORS_SUPPLEMENT,
    ZERO_WIDTH_AND_FORMAT,
    Finding,
    fix_file,
    format_line_with_placeholder,
    get_unicode_info,
    is_forbidden,
    sanitize_content,
    scan_content,
    scan_file,
)


class TestForbiddenCodepoints(unittest.TestCase):
    """Test that all expected forbidden codepoints are defined."""
    
    def test_bidi_controls_are_forbidden(self):
        """All bidi control characters should be in the forbidden set."""
        bidi_controls = [
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
        ]
        for cp in bidi_controls:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS, 
                         f"U+{cp:04X} should be forbidden")
            self.assertIn(cp, BIDI_CONTROLS,
                         f"U+{cp:04X} should be in BIDI_CONTROLS")
    
    def test_zero_width_chars_are_forbidden(self):
        """Zero-width/invisible characters should be in the forbidden set."""
        zero_width = [
            0x200B,  # ZERO WIDTH SPACE (ZWSP)
            0x200C,  # ZERO WIDTH NON-JOINER (ZWNJ)
            0x200D,  # ZERO WIDTH JOINER (ZWJ)
            0x2060,  # WORD JOINER
            0xFEFF,  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
            0x00AD,  # SOFT HYPHEN
            0x034F,  # COMBINING GRAPHEME JOINER (CGJ)
            0x180E,  # MONGOLIAN VOWEL SEPARATOR
        ]
        for cp in zero_width:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
            self.assertIn(cp, ZERO_WIDTH_AND_FORMAT,
                         f"U+{cp:04X} should be in ZERO_WIDTH_AND_FORMAT")
    
    def test_variation_selectors_are_forbidden(self):
        """Variation selectors U+FE00..U+FE0F should be forbidden."""
        for cp in range(0xFE00, 0xFE10):
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
            self.assertIn(cp, VARIATION_SELECTORS_1,
                         f"U+{cp:04X} should be in VARIATION_SELECTORS_1")
    
    def test_variation_selectors_supplement_are_forbidden(self):
        """Variation selectors supplement U+E0100..U+E01EF should be forbidden."""
        # Test a sample - don't test all 240
        test_cps = [0xE0100, 0xE0110, 0xE01EF]
        for cp in test_cps:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
            self.assertIn(cp, VARIATION_SELECTORS_SUPPLEMENT,
                         f"U+{cp:04X} should be in VARIATION_SELECTORS_SUPPLEMENT")
    
    def test_ascii_controls_are_forbidden(self):
        """ASCII control characters (except tab/newline/CR) should be forbidden."""
        # Forbidden ASCII controls
        forbidden_ascii = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x0B, 0x0C,  # VT, FF
            0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
            0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
            0x7F,  # DEL
        ]
        for cp in forbidden_ascii:
            self.assertIn(cp, FORBIDDEN_CODEPOINTS,
                         f"U+{cp:04X} should be forbidden")
            self.assertIn(cp, ASCII_CONTROLS,
                         f"U+{cp:04X} should be in ASCII_CONTROLS")
    
    def test_allowed_ascii_controls_not_forbidden(self):
        """Tab, newline, and carriage return should NOT be forbidden."""
        allowed = [0x09, 0x0A, 0x0D]  # TAB, LF, CR
        for cp in allowed:
            self.assertNotIn(cp, FORBIDDEN_CODEPOINTS,
                            f"U+{cp:04X} should NOT be forbidden")


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
    
    def test_em_dash_not_flagged(self):
        """Em dash (U+2014) and other typography should not be flagged."""
        content = "# This — that (em dash is allowed)\n"
        findings = scan_content(content, "test.py")
        self.assertEqual(len(findings), 0,
                        "Em dash should not be flagged")
    
    def test_detect_bidi_override(self):
        """U+202E (RIGHT-TO-LEFT OVERRIDE) should be detected."""
        rlo = chr(0x202E)
        content = f"# Comment with{rlo}hidden text\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x202E)
        self.assertEqual(findings[0].line, 1)
        self.assertEqual(findings[0].column, 15)
        self.assertIn("RIGHT-TO-LEFT OVERRIDE", findings[0].name)
    
    def test_detect_zero_width_space(self):
        """U+200B (ZERO WIDTH SPACE) should be detected."""
        zws = chr(0x200B)
        content = f"variable{zws}name = 42\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x200B)
    
    def test_detect_word_joiner(self):
        """U+2060 (WORD JOINER) should be detected."""
        wj = chr(0x2060)
        content = f"function{wj}name()\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x2060)
    
    def test_detect_combining_grapheme_joiner(self):
        """U+034F (COMBINING GRAPHEME JOINER) should be detected."""
        cgj = chr(0x034F)
        content = f"test{cgj}string\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x034F)
    
    def test_detect_variation_selector(self):
        """U+FE0F (VARIATION SELECTOR-16) should be detected."""
        vs16 = chr(0xFE0F)
        content = f"emoji{vs16}text\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0xFE0F)
    
    def test_detect_ascii_control_bell(self):
        """ASCII control BELL (0x07) should be detected."""
        bell = chr(0x07)
        content = f"alert{bell}sound\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x07)
        self.assertIn("BELL", findings[0].name)
    
    def test_detect_ascii_control_escape(self):
        """ASCII control ESCAPE (0x1B) should be detected."""
        esc = chr(0x1B)
        content = f"color{esc}[31m\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x1B)
        self.assertIn("ESCAPE", findings[0].name)
    
    def test_detect_soft_hyphen(self):
        """U+00AD (SOFT HYPHEN) should be detected."""
        shy = chr(0x00AD)
        content = f"function{shy}name()\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x00AD)
    
    def test_detect_mongolian_vowel_separator(self):
        """U+180E (MONGOLIAN VOWEL SEPARATOR) should be detected."""
        mvs = chr(0x180E)
        content = f"text{mvs}more\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].codepoint, 0x180E)
    
    def test_detect_multiple_violations(self):
        """Multiple forbidden characters should all be detected."""
        rlo = chr(0x202E)
        zws = chr(0x200B)
        cgj = chr(0x034F)
        content = f"line1{rlo}bad\nline2{zws}also bad\nline3{cgj}more\n"
        findings = scan_content(content, "test.py")
        
        self.assertEqual(len(findings), 3)
        codepoints = {f.codepoint for f in findings}
        self.assertIn(0x202E, codepoints)
        self.assertIn(0x200B, codepoints)
        self.assertIn(0x034F, codepoints)
    
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
    
    def test_tabs_and_newlines_allowed(self):
        """Tab and newline characters should NOT be flagged."""
        content = "def func():\n\treturn True\n"
        findings = scan_content(content, "test.py")
        self.assertEqual(len(findings), 0,
                        "Tabs and newlines should be allowed")


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


class TestSanitizeContent(unittest.TestCase):
    """Test content sanitization."""
    
    def test_sanitize_removes_bidi_controls(self):
        """Sanitize should remove bidi control characters."""
        rlo = chr(0x202E)
        content = f"text{rlo}more"
        result = sanitize_content(content)
        self.assertEqual(result, "textmore")
        self.assertNotIn(rlo, result)
    
    def test_sanitize_removes_zero_width(self):
        """Sanitize should remove zero-width characters."""
        zws = chr(0x200B)
        content = f"hello{zws}world"
        result = sanitize_content(content)
        self.assertEqual(result, "helloworld")
    
    def test_sanitize_removes_variation_selectors(self):
        """Sanitize should remove variation selectors."""
        vs16 = chr(0xFE0F)
        content = f"emoji{vs16}text"
        result = sanitize_content(content)
        self.assertEqual(result, "emojitext")
    
    def test_sanitize_removes_ascii_controls(self):
        """Sanitize should remove forbidden ASCII controls."""
        bell = chr(0x07)
        content = f"alert{bell}sound"
        result = sanitize_content(content)
        self.assertEqual(result, "alertsound")
    
    def test_sanitize_preserves_allowed_chars(self):
        """Sanitize should preserve allowed characters."""
        content = "Hello, World! 你好 ✓ —\n\tindented"
        result = sanitize_content(content)
        self.assertEqual(result, content)
    
    def test_sanitize_preserves_newlines_tabs(self):
        """Sanitize should preserve newlines and tabs."""
        content = "line1\n\tline2\r\nline3"
        result = sanitize_content(content)
        self.assertEqual(result, content)


class TestFixFile(unittest.TestCase):
    """Test the --fix mode file fixing functionality."""
    
    def test_fix_removes_forbidden_chars(self):
        """Fix should remove forbidden characters from file."""
        rlo = chr(0x202E)
        zws = chr(0x200B)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            f.write(f"text{rlo}hidden{zws}chars\n")
            f.flush()
            path = Path(f.name)
        
        try:
            # Fix the file
            modified = fix_file(path)
            self.assertTrue(modified, "File should be modified")
            
            # Read back and verify
            content = path.read_text(encoding='utf-8')
            self.assertEqual(content, "texthiddenchars\n")
            self.assertNotIn(rlo, content)
            self.assertNotIn(zws, content)
            
            # Subsequent scan should pass
            findings = scan_file(path)
            self.assertEqual(len(findings), 0,
                           "Fixed file should have no findings")
        finally:
            path.unlink()
    
    def test_fix_clean_file_not_modified(self):
        """Fix should not modify already-clean files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            original = "def clean():\n    return True\n"
            f.write(original)
            f.flush()
            path = Path(f.name)
        
        try:
            modified = fix_file(path)
            self.assertFalse(modified, "Clean file should not be modified")
            
            # Content should be unchanged
            content = path.read_text(encoding='utf-8')
            self.assertEqual(content, original)
        finally:
            path.unlink()
    
    def test_fix_preserves_crlf_line_endings(self):
        """Fix should preserve CRLF line endings."""
        rlo = chr(0x202E)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py',
                                          delete=False) as f:
            content = f"line1{rlo}bad\r\nline2\r\n"
            f.write(content.encode('utf-8'))
            f.flush()
            path = Path(f.name)
        
        try:
            modified = fix_file(path)
            self.assertTrue(modified)
            
            # Read back raw bytes to verify CRLF preserved
            # (read_text does universal newline translation, so use bytes)
            raw = path.read_bytes()
            self.assertEqual(raw, b"line1bad\r\nline2\r\n")
            self.assertIn(b'\r\n', raw)
        finally:
            path.unlink()
    
    def test_fix_skips_binary_files(self):
        """Fix should skip binary files."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin',
                                          delete=False) as f:
            original = b'\x00\x01\x02binary'
            f.write(original)
            f.flush()
            path = Path(f.name)
        
        try:
            modified = fix_file(path)
            self.assertFalse(modified, "Binary file should not be modified")
            
            # Content should be unchanged
            content = path.read_bytes()
            self.assertEqual(content, original)
        finally:
            path.unlink()
    
    def test_fix_multiple_forbidden_chars(self):
        """Fix should remove all forbidden characters in one pass."""
        chars = [
            chr(0x202E),  # bidi
            chr(0x200B),  # zero-width
            chr(0x034F),  # CGJ
            chr(0xFE0F),  # variation selector
            chr(0x07),    # ASCII control
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            content = "a" + "".join(chars) + "b\n"
            f.write(content)
            f.flush()
            path = Path(f.name)
        
        try:
            modified = fix_file(path)
            self.assertTrue(modified)
            
            result = path.read_text(encoding='utf-8')
            self.assertEqual(result, "ab\n")
            
            # Verify all forbidden chars removed
            for c in chars:
                self.assertNotIn(c, result)
            
            # Scan should pass
            findings = scan_file(path)
            self.assertEqual(len(findings), 0)
        finally:
            path.unlink()


class TestFormatLineWithPlaceholder(unittest.TestCase):
    """Test the line formatting with placeholder."""
    
    def test_replace_character_with_placeholder(self):
        """The forbidden character should be replaced with placeholder."""
        rlo = chr(0x202E)
        line = f"ABC{rlo}DEF"
        result = format_line_with_placeholder(line, 4, 0x202E)
        
        self.assertEqual(result, "ABC<U+202E>DEF")
        self.assertNotIn(rlo, result)
    
    def test_placeholder_at_start(self):
        """Placeholder should work at the start of a line."""
        zws = chr(0x200B)
        line = f"{zws}start"
        result = format_line_with_placeholder(line, 1, 0x200B)
        
        self.assertEqual(result, "<U+200B>start")
    
    def test_placeholder_at_end(self):
        """Placeholder should work at the end of a line."""
        shy = chr(0x00AD)
        line = f"end{shy}"
        result = format_line_with_placeholder(line, 4, 0x00AD)
        
        self.assertEqual(result, "end<U+00AD>")


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
    
    def test_get_info_for_ascii_control(self):
        """Should get descriptive name for ASCII controls."""
        name, category = get_unicode_info(0x07)
        self.assertIn("BELL", name)
        self.assertEqual(category, "Cc")  # Control character
    
    def test_get_info_for_cgj(self):
        """Should get correct info for CGJ."""
        name, category = get_unicode_info(0x034F)
        self.assertIn("COMBINING GRAPHEME JOINER", name)
        self.assertEqual(category, "Mn")  # Nonspacing mark


class TestIsForbidden(unittest.TestCase):
    """Test the is_forbidden helper function."""
    
    def test_forbidden_returns_true(self):
        """is_forbidden should return True for forbidden codepoints."""
        forbidden_samples = [0x202E, 0x200B, 0x034F, 0xFE0F, 0x07]
        for cp in forbidden_samples:
            self.assertTrue(is_forbidden(cp), f"U+{cp:04X} should be forbidden")
    
    def test_allowed_returns_false(self):
        """is_forbidden should return False for allowed codepoints."""
        allowed_samples = [
            ord('A'), ord('z'), ord('0'),  # ASCII
            ord('é'), ord('中'), ord('✓'),  # Unicode
            0x09, 0x0A, 0x0D,  # Tab, LF, CR
            ord('—'),  # Em dash
        ]
        for cp in allowed_samples:
            self.assertFalse(is_forbidden(cp), 
                           f"U+{cp:04X} should NOT be forbidden")


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
