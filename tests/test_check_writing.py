"""Unit tests for check_writing.py"""

import json
import tempfile
from pathlib import Path

import pytest

# Import the module under test
import check_writing as cw


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_markers():
    """Sample markers for testing."""
    return [
        {"item": "comprehensive", "type": "word", "log_odds": 3.5, "opus_rate": 0.01, "human_rate": 0.001},
        {"item": "robust", "type": "word", "log_odds": 3.0, "opus_rate": 0.008, "human_rate": 0.001},
        {"item": "fundamentally", "type": "word", "log_odds": 2.8, "opus_rate": 0.005, "human_rate": 0.0008},
        {"item": "typically", "type": "word", "log_odds": 2.2, "opus_rate": 0.02, "human_rate": 0.005},
        {"item": "in this", "type": "bigram", "log_odds": 1.8, "opus_rate": 0.01, "human_rate": 0.003},
        {"item": "python", "type": "word", "log_odds": 2.0, "opus_rate": 0.01, "human_rate": 0.002},
    ]


@pytest.fixture
def neutral_text():
    """Human-like text without AI markers."""
    return """
    The old bookshop on Market Street had been there for forty years.
    Mrs. Chen opened it the same year her daughter was born, filling
    the shelves with paperbacks and hardcovers from estate sales
    across the county. The smell of yellowed pages mixed with the
    jasmine tea she kept brewing behind the counter.
    """


@pytest.fixture
def ai_text():
    """Text with AI markers."""
    return """
    This comprehensive guide provides a robust framework for understanding
    the fundamentals of software development. Typically, developers need
    to consider multiple factors when building applications. In this document,
    we'll explore the various aspects that fundamentally shape how we
    approach comprehensive system design.
    """


@pytest.fixture
def technical_text():
    """Technical documentation with legitimate tech terms."""
    return """
    The Python API provides authentication via OAuth tokens. Users can
    configure rate limiting through the configuration file. The REST API
    endpoints handle JSON responses for all requests.
    """


# =============================================================================
# Test check_text function
# =============================================================================

class TestCheckText:
    """Tests for the check_text function."""

    def test_neutral_text_has_few_findings(self, sample_markers, neutral_text):
        """Neutral human-like text should have few or no findings."""
        findings = cw.check_text(neutral_text, sample_markers, technical=True)

        assert findings["stats"]["high_severity"] == 0
        assert findings["stats"]["patterns_found"] <= 2

    def test_ai_text_has_findings(self, sample_markers, ai_text):
        """AI-like text should have multiple findings."""
        findings = cw.check_text(ai_text, sample_markers, technical=True)

        assert findings["stats"]["patterns_found"] > 0
        # Should find "comprehensive", "robust", "fundamentally", "typically"
        all_patterns = [f["pattern"].lower() for f in findings["high"] + findings["medium"]]
        assert "comprehensive" in all_patterns
        assert "robust" in all_patterns

    def test_word_count(self, sample_markers, neutral_text):
        """Word count should be calculated correctly."""
        findings = cw.check_text(neutral_text, sample_markers)

        # Approximate word count
        assert findings["stats"]["total_words"] > 50
        assert findings["stats"]["total_words"] < 200

    def test_technical_mode_excludes_tech_terms(self, sample_markers, technical_text):
        """Technical mode should exclude technical terms like 'python'."""
        findings_tech = cw.check_text(technical_text, sample_markers, technical=True)
        findings_strict = cw.check_text(technical_text, sample_markers, technical=False)

        # Technical mode should have fewer findings
        assert findings_tech["stats"]["patterns_found"] <= findings_strict["stats"]["patterns_found"]

        # "python" should not be flagged in technical mode
        tech_patterns = [f["pattern"].lower() for f in findings_tech["high"] + findings_tech["medium"]]
        assert "python" not in tech_patterns

    def test_empty_text(self, sample_markers):
        """Empty text should return zero counts."""
        findings = cw.check_text("", sample_markers)

        assert findings["stats"]["total_words"] == 0
        assert findings["stats"]["patterns_found"] == 0


# =============================================================================
# Test calculate_score function
# =============================================================================

class TestCalculateScore:
    """Tests for the calculate_score function."""

    def test_perfect_score_for_clean_text(self, sample_markers, neutral_text):
        """Clean text should score high."""
        findings = cw.check_text(neutral_text, sample_markers)
        score = cw.calculate_score(findings)

        assert score >= 80
        assert score <= 100

    def test_low_score_for_ai_text(self, sample_markers, ai_text):
        """AI-heavy text should score low."""
        findings = cw.check_text(ai_text, sample_markers)
        score = cw.calculate_score(findings)

        assert score < 80

    def test_score_bounds(self, sample_markers):
        """Score should always be between 0 and 100."""
        # Test with extremely AI-heavy text
        heavy_ai_text = "comprehensive " * 100
        findings = cw.check_text(heavy_ai_text, sample_markers)
        score = cw.calculate_score(findings)

        assert score >= 0
        assert score <= 100


# =============================================================================
# Test format functions
# =============================================================================

class TestFormatFunctions:
    """Tests for output formatting functions."""

    def test_format_text_contains_score(self, sample_markers, neutral_text):
        """Text format should contain score."""
        findings = cw.check_text(neutral_text, sample_markers)
        score = cw.calculate_score(findings)
        output = cw.format_text(findings, score, "test.md")

        assert f"{score}/100" in output
        assert "test.md" in output

    def test_format_json_is_valid(self, sample_markers, neutral_text):
        """JSON format should be valid JSON."""
        findings = cw.check_text(neutral_text, sample_markers)
        score = cw.calculate_score(findings)
        output = cw.format_json(findings, score, "test.md")

        data = json.loads(output)
        assert data["score"] == score
        assert data["filename"] == "test.md"
        assert "stats" in data

    def test_format_html_contains_markup(self, sample_markers, neutral_text):
        """HTML format should contain proper markup."""
        findings = cw.check_text(neutral_text, sample_markers)
        score = cw.calculate_score(findings)
        output = cw.format_html(findings, score, "test.md")

        assert "<html" in output.lower()  # May have attributes like lang="en"
        assert "</html>" in output.lower()
        assert "test.md" in output


# =============================================================================
# Test config functions
# =============================================================================

class TestConfigFunctions:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Default config should have expected values."""
        config = cw.load_config()

        assert config["min_score"] == 60
        assert config["technical"] == True
        assert isinstance(config["exclude"], list)
        assert isinstance(config["ignore_patterns"], list)

    def test_load_config_from_file(self):
        """Config should load from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("min_score: 75\ntechnical: false\n")
            f.flush()

            config = cw.load_config(Path(f.name))

            assert config["min_score"] == 75
            assert config["technical"] == False

    def test_invalid_yaml_returns_default(self):
        """Invalid YAML should return default config with warning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [[[\n")
            f.flush()

            config = cw.load_config(Path(f.name))

            # Should return default config
            assert config["min_score"] == 60


# =============================================================================
# Test file exclusion
# =============================================================================

class TestFileExclusion:
    """Tests for file exclusion patterns."""

    def test_exclude_exact_match(self):
        """Exact filename should be excluded."""
        patterns = ["CHANGELOG.md", "vendor/**"]

        assert cw.should_exclude_file("CHANGELOG.md", patterns) == True
        assert cw.should_exclude_file("README.md", patterns) == False

    def test_exclude_glob_pattern(self):
        """Glob patterns should work."""
        patterns = ["vendor/**", "*.min.js"]

        assert cw.should_exclude_file("vendor/lib/file.js", patterns) == True
        assert cw.should_exclude_file("app.min.js", patterns) == True
        assert cw.should_exclude_file("app.js", patterns) == False

    def test_empty_patterns(self):
        """Empty patterns should exclude nothing."""
        assert cw.should_exclude_file("anything.md", []) == False


# =============================================================================
# Test structural analysis
# =============================================================================

class TestStructuralAnalysis:
    """Tests for structural analysis functions."""

    def test_analyze_structure_basic(self):
        """Basic structural analysis."""
        text = """
        This is the first paragraph with several sentences. It has some content.

        This is the second paragraph. It also has content. And more sentences here.
        """

        result = cw.analyze_structure(text)

        assert result["para_count"] == 2
        assert result["sentence_count"] > 0
        assert result["avg_para_words"] > 0

    def test_analyze_structure_list_items(self):
        """List items should be counted."""
        text = """
        Here is a list:
        - Item one
        - Item two
        - Item three

        1. Numbered one
        2. Numbered two
        """

        result = cw.analyze_structure(text)

        assert result["list_items"] >= 5

    def test_analyze_structure_empty(self):
        """Empty text should return zeros."""
        result = cw.analyze_structure("")

        assert result["para_count"] == 0
        assert result["sentence_count"] == 0


# =============================================================================
# Test severity functions
# =============================================================================

class TestSeverityFunctions:
    """Tests for severity classification."""

    def test_get_severity_high(self):
        """High log_odds should return high severity."""
        assert cw.get_severity(3.0) == "high"
        assert cw.get_severity(2.5) == "high"

    def test_get_severity_medium(self):
        """Medium log_odds should return medium severity."""
        assert cw.get_severity(2.0) == "medium"
        assert cw.get_severity(1.5) == "medium"

    def test_get_severity_low(self):
        """Low log_odds should return low severity."""
        assert cw.get_severity(1.0) == "low"
        assert cw.get_severity(0.5) == "low"

    def test_get_grade(self):
        """Grade descriptions should match score ranges."""
        assert "Excellent" in cw.get_grade(95)
        assert "Good" in cw.get_grade(80)
        assert "Fair" in cw.get_grade(65)
        assert "Needs work" in cw.get_grade(50)
        assert "High AI signal" in cw.get_grade(30)


# =============================================================================
# Test utility functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_read_file_txt(self):
        """Should read text files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content here.")
            f.flush()

            content = cw.read_file(Path(f.name))
            assert content == "Test content here."

    def test_read_file_md(self):
        """Should read markdown files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Heading\n\nParagraph text.")
            f.flush()

            content = cw.read_file(Path(f.name))
            assert "# Heading" in content
            assert "Paragraph text" in content


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_text(self, sample_markers):
        """Very short text should be handled gracefully."""
        findings = cw.check_text("Hi.", sample_markers)
        score = cw.calculate_score(findings)

        assert score >= 0
        assert score <= 100

    def test_text_with_unicode(self, sample_markers):
        """Unicode text should be handled correctly."""
        text = "This is comprehensive analysis with émojis 🎉 and unicode: café résumé"
        findings = cw.check_text(text, sample_markers)

        # Should find "comprehensive"
        patterns = [f["pattern"].lower() for f in findings["high"] + findings["medium"]]
        assert "comprehensive" in patterns

    def test_text_with_special_characters(self, sample_markers):
        """Special characters should not break parsing."""
        text = "A comprehensive look at regex: [a-z]+ and paths: C:\\Users\\test"
        findings = cw.check_text(text, sample_markers)

        # Should still work
        assert "stats" in findings

    def test_duplicate_pattern_handling(self, sample_markers):
        """Duplicate patterns should not cause issues."""
        # Add duplicate marker
        markers_with_dup = sample_markers + [
            {"item": "comprehensive", "type": "phrase_llm_favorite", "log_odds": 3.2,
             "opus_rate": 0.01, "human_rate": 0.001}
        ]

        text = "This is a comprehensive comprehensive document."
        findings = cw.check_text(text, markers_with_dup)

        # Should count pattern only once (or handle gracefully)
        comp_findings = [f for f in findings["high"] + findings["medium"]
                        if f["pattern"].lower() == "comprehensive"]
        # Either deduplicated or both counted - just ensure no crash
        assert len(comp_findings) >= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
