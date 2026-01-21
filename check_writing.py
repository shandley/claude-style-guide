#!/usr/bin/env python3
"""
Writing Checker - Scan your documents for LLM-isms.

Usage:
    python check_writing.py document.md
    python check_writing.py document.docx
    python check_writing.py document.md --verbose
    python check_writing.py document.md --format json
    python check_writing.py document.md --format html > report.html
    python check_writing.py document.md --interactive
    python check_writing.py *.md --format text
    cat document.md | python check_writing.py --stdin

Supported formats: .txt, .md, .docx
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx"}


def find_markers_file() -> Path:
    """Find markers.json in various locations."""
    # 1. Local development path (relative to this file)
    local_path = Path(__file__).parent / "results" / "markers.json"
    if local_path.exists():
        return local_path

    # 2. Installed package data location
    try:
        import importlib.resources as pkg_resources
        # Python 3.9+ with files()
        if hasattr(pkg_resources, 'files'):
            pkg_path = pkg_resources.files('prose_check_data') / 'markers.json'
            if pkg_path.is_file():
                return Path(str(pkg_path))
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    # 3. User data directory
    user_data = Path.home() / ".prose-check" / "markers.json"
    if user_data.exists():
        return user_data

    # 4. System data directory (Unix)
    system_data = Path("/usr/local/share/prose-check/markers.json")
    if system_data.exists():
        return system_data

    # Return the local path (will fail gracefully in load_markers)
    return local_path


# Path to markers file
MARKERS_PATH = find_markers_file()

# Severity thresholds (log-odds)
HIGH_SEVERITY = 2.5
MEDIUM_SEVERITY = 1.5

# Structural thresholds (based on analysis)
# AI avg: 16.2 words/para, Human avg: ~200 words/para
MIN_HEALTHY_PARA_LENGTH = 40  # Below this is AI-like
# AI avg: 9.5 list items/doc, Human avg: ~0
MAX_HEALTHY_LIST_ITEMS_PER_100_WORDS = 1.0

# Markdown syntax patterns to ignore (not writing style issues)
MARKDOWN_PATTERNS = {
    "##", "###", "####", "#####",
    "```", "``", "`",
    "**", "__", "*", "_",
    "---", "***", "___",
    "- [", "- [ ]", "- [x]",
    "|", "|-",
    ">",
}

# Training data artifacts (character names, etc.) - always exclude
TRAINING_ARTIFACTS = {
    "jenna", "marcus", "margaret", "chen", "sarah", "mike", "david",
    "alex", "tom", "lisa", "john", "james", "emma", "emily",
    "**marcus**", "**jenna**", "**margaret:**", "**the",
}

# Technical terms - exclude when in technical context
TECHNICAL_TERMS = {
    # Languages & frameworks
    "python", "javascript", "typescript", "java", "rust", "go", "ruby",
    "react", "vue", "angular", "node", "django", "flask", "rails",
    "bash", "shell", "powershell", "perl", "php", "scala", "kotlin",
    # Infrastructure & tools
    "docker", "kubernetes", "aws", "azure", "gcp", "linux", "unix",
    "git", "github", "gitlab", "npm", "yarn", "pip", "cargo",
    "terraform", "ansible", "jenkins", "circleci", "travis",
    # Concepts
    "api", "apis", "rest", "graphql", "sql", "nosql", "json", "xml",
    "html", "css", "http", "https", "url", "uri", "oauth", "jwt",
    "microservices", "monolith", "serverless", "devops", "ci", "cd",
    "frontend", "backend", "fullstack", "database", "cache", "cdn",
    "tdd", "bdd", "agile", "scrum", "sprint", "kanban",
    # AI/ML specific
    "ai", "ml", "llm", "gpt", "nlp", "neural", "model", "training",
    "pytorch", "tensorflow", "keras", "sklearn", "pandas", "numpy",
    # Documentation terms
    "readme", "changelog", "documentation", "docs", "wiki",
    "technical", "specification", "requirements", "architecture",
}

# Generic programming terms - exclude when in technical context
PROGRAMMING_TERMS = {
    "code", "function", "method", "class", "variable", "parameter",
    "argument", "return", "loop", "array", "object", "string",
    "integer", "boolean", "null", "undefined", "import", "export",
    "module", "package", "library", "framework", "dependency",
    "error", "errors", "exception", "bug", "bugs", "debug",
    "test", "tests", "testing", "unit", "integration",
    "deploy", "deployment", "build", "compile", "runtime",
    "server", "client", "request", "response", "endpoint",
    "query", "mutation", "schema", "type", "interface",
    "pattern", "patterns", "architecture", "design",
    "config", "configuration", "environment", "production", "staging",
    "log", "logs", "logging", "monitor", "monitoring",
    "async", "await", "promise", "callback", "event",
    # Database terms
    "postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle",
    "table", "column", "row", "index", "migration", "seed",
    # Auth terms
    "authentication", "authorization", "token", "tokens", "session",
    "credentials", "password", "username", "login", "logout",
    "permission", "permissions", "role", "roles", "user", "users",
    # Common technical phrases (as single words)
    "timestamp", "timeout", "expiration", "pagination", "validation",
    "serialization", "deserialization", "middleware", "handler",
    "controller", "service", "repository", "factory", "singleton",
    "requests", "responses", "headers", "payload", "body",
    # Additional technical terms
    "context", "contexts", "conventions", "convention", "document", "documents",
    "documentation", "limiting", "limits", "limit", "specified", "specify",
    "parameters", "values", "value", "options", "option", "settings",
    "format", "formats", "output", "input", "inputs", "outputs",
    "process", "processes", "processing", "handle", "handling",
}

# Technical bigrams/trigrams to exclude
TECHNICAL_PHRASES = {
    "error handling", "rate limiting", "rate limit", "rest api", "api endpoint",
    "api endpoints", "user id", "user ids", "access token", "access tokens",
    "http status", "status code", "status codes", "query string", "request body",
    "response body", "json response", "json responses", "the api", "the backend",
    "the frontend", "the database", "the server", "the client", "this document",
    "this api", "this endpoint", "this method", "this function", "this class",
    "error code", "error codes", "error message", "error messages",
    "best practices", "use case", "use cases", "end user", "end users",
    "to users", "for users", "by users", "the user", "a user", "each user",
    "data model", "data models", "file format", "return value", "return values",
    "input data", "output data", "default value", "default values",
}

# Categories to check
CATEGORIES = {
    "phrase_hedging": "Hedging",
    "phrase_transition": "Transitions",
    "phrase_filler": "Fillers",
    "phrase_structure": "Structure phrases",
    "phrase_conclusion": "Conclusions",
    "phrase_emphasis": "Emphasis",
    "phrase_llm_favorite": "LLM favorites",
    "word": "Overused words",
    "bigram": "Bigrams",
    "trigram": "Trigrams",
    "sentence_starter": "Sentence starters",
}

# Hedging words that AI overuses (with ratios from analysis)
HEDGING_WORDS = {
    "typically": 9.6,
    "often": 4.9,
    "sometimes": 4.2,
    "potentially": 3.4,
    "usually": 3.4,
    "rather": 3.3,
    "probably": 2.2,
}

# Formulaic sentence starters AI overuses
FORMULAIC_STARTERS = [
    "this document",
    "this guide",
    "this article",
    "this section",
    "comprehensive",
    "in this",
    "introduction",
    "let's",
]

# Human alternatives for common patterns
ALTERNATIVES = {
    "comprehensive": "complete, full, thorough",
    "utilize": "use",
    "leverage": "use, apply",
    "facilitate": "help, enable",
    "robust": "strong, solid",
    "nuanced": "subtle, detailed",
    "paradigm": "model, approach",
    "in essence": "basically (or delete)",
    "fundamentally": "basically (or delete)",
    "essentially": "basically (or delete)",
    "furthermore": "also, and",
    "moreover": "also, and",
    "additionally": "also, and",
    "in order to": "to",
    "due to the fact that": "because",
    "it's important to note": "(delete)",
    "it's worth noting": "(delete)",
    "that being said": "but, however",
}

# Default configuration
DEFAULT_CONFIG = {
    "min_score": 60,
    "exclude": [],
    "ignore_patterns": [],
    "technical": True,  # Exclude technical terms (for tech docs)
}


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load config from .prose-check.yaml if exists."""
    config = DEFAULT_CONFIG.copy()

    # Try to find config file
    if config_path is None:
        # Look in current directory and parent directories
        search_paths = [
            Path.cwd() / ".prose-check.yaml",
            Path.cwd() / ".prose-check.yml",
            Path(__file__).parent / ".prose-check.yaml",
            Path(__file__).parent / ".prose-check.yml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        if not YAML_AVAILABLE:
            print("Warning: PyYAML not installed, config file ignored", file=sys.stderr)
            return config

        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Warning: Invalid config file {config_path}: {e}", file=sys.stderr)
            return config

        # Merge with defaults
        if "min_score" in file_config:
            config["min_score"] = file_config["min_score"]
        if "exclude" in file_config:
            config["exclude"] = file_config["exclude"]
        if "ignore_patterns" in file_config:
            config["ignore_patterns"] = file_config["ignore_patterns"]
        if "technical" in file_config:
            config["technical"] = file_config["technical"]

    return config


def should_exclude_file(filepath: str, exclude_patterns: list) -> bool:
    """Check if file matches any exclude pattern."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        # Also check basename
        if fnmatch.fnmatch(Path(filepath).name, pattern):
            return True
    return False


def load_markers() -> dict:
    """Load markers from analysis results."""
    if not MARKERS_PATH.exists():
        print(f"Error: Markers file not found at {MARKERS_PATH}", file=sys.stderr)
        print("Run 'python run_pipeline.py analyze' first.", file=sys.stderr)
        sys.exit(1)

    with open(MARKERS_PATH) as f:
        return json.load(f)


def read_docx(path: Path) -> str:
    """Extract text from a .docx file using textutil (macOS) or python-docx."""
    # Try textutil first (macOS built-in)
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        pass  # textutil not available, try python-docx
    except subprocess.CalledProcessError as e:
        print(f"Error converting docx: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Fall back to python-docx if available
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        print("Error: Cannot read .docx files.", file=sys.stderr)
        print("Install python-docx: pip install python-docx", file=sys.stderr)
        sys.exit(1)


def read_file(path: Path) -> str:
    """Read text from a file, handling different formats."""
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported file type: {suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    if suffix == ".docx":
        return read_docx(path)
    else:
        return path.read_text()


def get_severity(log_odds: float) -> str:
    """Get severity level from log-odds."""
    if log_odds >= HIGH_SEVERITY:
        return "high"
    elif log_odds >= MEDIUM_SEVERITY:
        return "medium"
    return "low"


def check_text(text: str, markers: list, verbose: bool = False, technical: bool = True) -> dict:
    """
    Check text for LLM patterns.

    Args:
        text: Text to analyze
        markers: List of marker patterns from markers.json
        verbose: Include low-severity findings
        technical: Exclude technical/programming terms (default True)

    Returns dict with findings.
    """
    findings = {
        "high": [],
        "medium": [],
        "low": [],
        "by_category": defaultdict(list),
        "stats": {
            "total_chars": len(text),
            "total_words": len(text.split()),
            "patterns_found": 0,
            "high_severity": 0,
            "medium_severity": 0,
        }
    }

    # Build exclusion set based on context
    excluded_patterns = set(MARKDOWN_PATTERNS) | set(TRAINING_ARTIFACTS)
    if technical:
        excluded_patterns |= TECHNICAL_TERMS | PROGRAMMING_TERMS | TECHNICAL_PHRASES

    # Track seen patterns to avoid duplicates (keep highest log_odds)
    seen_patterns = {}  # pattern_lower -> (severity, index_in_list, log_odds)

    # Check each marker
    for marker in markers:
        item = marker["item"]
        marker_type = marker["type"]
        log_odds = marker["log_odds"]

        # Skip excluded patterns (markdown, training artifacts, tech terms)
        item_lower = item.lower().strip()
        if item in excluded_patterns or item_lower in excluded_patterns:
            continue

        # Skip low-ratio items unless verbose
        if log_odds < MEDIUM_SEVERITY and not verbose:
            continue

        # Count occurrences
        if marker_type == "sentence_starter":
            # Match at start of sentences
            pattern = re.compile(r'(?:^|[.!?]\s+)' + re.escape(item), re.IGNORECASE | re.MULTILINE)
            matches = pattern.findall(text)
            count = len(matches)
        else:
            # Simple word/phrase match
            pattern = re.compile(r'\b' + re.escape(item) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(text))
            count = len(matches)

        if count > 0:
            severity = get_severity(log_odds)
            ratio = marker["opus_rate"] / marker["human_rate"] if marker["human_rate"] > 0 else float('inf')

            finding = {
                "pattern": item,
                "type": marker_type,
                "count": count,
                "severity": severity,
                "ratio": ratio,
                "log_odds": log_odds,
                "alternative": ALTERNATIVES.get(item.lower(), None),
            }

            # Find example location
            if matches and not isinstance(matches[0], str):
                match = matches[0]
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].replace('\n', ' ')
                finding["context"] = f"...{context}..."

            # Deduplicate: only add if pattern not seen or this has higher log_odds
            pattern_key = item.lower()
            if pattern_key in seen_patterns:
                prev_severity, prev_idx, prev_log_odds = seen_patterns[pattern_key]
                if log_odds > prev_log_odds:
                    # Replace the previous finding with this one
                    findings[prev_severity][prev_idx] = None  # Mark for removal
                    if prev_severity == "high":
                        findings["stats"]["high_severity"] -= 1
                    elif prev_severity == "medium":
                        findings["stats"]["medium_severity"] -= 1
                    findings["stats"]["patterns_found"] -= 1
                else:
                    # Keep previous, skip this one
                    continue

            # Add the finding
            seen_patterns[pattern_key] = (severity, len(findings[severity]), log_odds)
            findings[severity].append(finding)
            findings["by_category"][marker_type].append(finding)
            findings["stats"]["patterns_found"] += 1

            if severity == "high":
                findings["stats"]["high_severity"] += 1
            elif severity == "medium":
                findings["stats"]["medium_severity"] += 1

    # Check punctuation
    em_dash_count = text.count("—") + text.count("--")
    if em_dash_count > 0:
        em_dash_per_1k = em_dash_count / len(text) * 1000
        # Human average is ~0.28 per 1k
        if em_dash_per_1k > 1.0:  # More than 3.5x human average
            findings["high"].append({
                "pattern": "em dash (—)",
                "type": "punctuation",
                "count": em_dash_count,
                "severity": "high",
                "ratio": em_dash_per_1k / 0.28,
                "alternative": "Use commas or periods instead",
                "context": f"{em_dash_per_1k:.1f} per 1k chars (human avg: 0.28)"
            })
            findings["stats"]["high_severity"] += 1
            findings["stats"]["patterns_found"] += 1

    # Check structural patterns
    structural = analyze_structure(text)
    findings["stats"]["structural"] = structural

    # Paragraph fragmentation check
    # In technical mode, use lower threshold (short paragraphs are normal in tech docs)
    para_threshold = 20 if technical else MIN_HEALTHY_PARA_LENGTH
    if structural["avg_para_words"] > 0 and structural["avg_para_words"] < para_threshold:
        # In technical mode, always treat as medium severity
        if technical:
            severity = "medium"
        else:
            severity = "high" if structural["avg_para_words"] < 25 else "medium"
        findings[severity].append({
            "pattern": "Short paragraphs",
            "type": "structure",
            "count": structural["para_count"],
            "severity": severity,
            "ratio": para_threshold / structural["avg_para_words"],
            "alternative": "Combine related ideas into longer paragraphs",
            "context": f"Avg {structural['avg_para_words']:.0f} words/para (aim for {para_threshold}+)"
        })
        if severity == "high":
            findings["stats"]["high_severity"] += 1
        else:
            findings["stats"]["medium_severity"] += 1
        findings["stats"]["patterns_found"] += 1

    # List overuse check
    total_words = findings["stats"]["total_words"]
    if total_words > 0:
        list_density = structural["list_items"] / total_words * 100
        if list_density > MAX_HEALTHY_LIST_ITEMS_PER_100_WORDS:
            findings["medium"].append({
                "pattern": "Bullet point overuse",
                "type": "structure",
                "count": structural["list_items"],
                "severity": "medium",
                "ratio": list_density / MAX_HEALTHY_LIST_ITEMS_PER_100_WORDS,
                "alternative": "Convert lists to prose paragraphs",
                "context": f"{structural['list_items']} list items in {total_words} words"
            })
            findings["stats"]["medium_severity"] += 1
            findings["stats"]["patterns_found"] += 1

    # Hedging word check
    hedging_count = 0
    hedging_details = []
    for word, ratio in HEDGING_WORDS.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        count = len(pattern.findall(text))
        if count > 0:
            hedging_count += count
            hedging_details.append(f"{word}({count})")

    if total_words > 0:
        hedging_per_1k = hedging_count / total_words * 1000
        # Flag if hedging rate is significantly above human average (4.8 per 1k)
        if hedging_per_1k > 8.0:  # ~1.7x human average
            findings["medium"].append({
                "pattern": "Hedging language overuse",
                "type": "hedging",
                "count": hedging_count,
                "severity": "medium",
                "ratio": hedging_per_1k / 4.8,
                "alternative": "Be more direct; reduce typically/often/sometimes",
                "context": f"{hedging_per_1k:.1f} per 1k words: {', '.join(hedging_details[:5])}"
            })
            findings["stats"]["medium_severity"] += 1
            findings["stats"]["patterns_found"] += 1

    # Formulaic sentence starter check
    sentences = re.split(r'[.!?]+', text)
    formulaic_count = 0
    formulaic_examples = []
    for sentence in sentences:
        sentence_clean = sentence.strip().lower()
        for starter in FORMULAIC_STARTERS:
            if sentence_clean.startswith(starter):
                formulaic_count += 1
                if len(formulaic_examples) < 3:
                    formulaic_examples.append(sentence.strip()[:50] + "...")
                break

    if formulaic_count >= 2:
        findings["medium"].append({
            "pattern": "Formulaic sentence starters",
            "type": "sentence_starter",
            "count": formulaic_count,
            "severity": "medium",
            "ratio": formulaic_count,
            "alternative": "Vary sentence openings; avoid 'This document/guide/article'",
            "context": "; ".join(formulaic_examples[:2])
        })
        findings["stats"]["medium_severity"] += 1
        findings["stats"]["patterns_found"] += 1

    # Clean up None values from deduplication
    for severity in ["high", "medium", "low"]:
        findings[severity] = [f for f in findings[severity] if f is not None]

    return findings


def analyze_structure(text: str) -> dict:
    """Analyze paragraph and list structure."""
    # Split into paragraphs (by blank lines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Calculate paragraph stats
    para_lengths = [len(p.split()) for p in paragraphs if len(p.split()) > 0]
    avg_para_words = sum(para_lengths) / len(para_lengths) if para_lengths else 0

    # Count list items (bullets and numbered)
    list_items = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
    list_items += len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))

    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]

    avg_sentence_words = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    # Categorize sentences
    short_sentences = sum(1 for length in sentence_lengths if length <= 10)
    medium_sentences = sum(1 for length in sentence_lengths if 10 < length <= 25)
    long_sentences = sum(1 for length in sentence_lengths if length > 25)
    total_sentences = len(sentence_lengths)

    return {
        "para_count": len(para_lengths),
        "avg_para_words": avg_para_words,
        "para_lengths": para_lengths,
        "list_items": list_items,
        "sentence_count": total_sentences,
        "avg_sentence_words": avg_sentence_words,
        "pct_short_sentences": (short_sentences / total_sentences * 100) if total_sentences else 0,
        "pct_medium_sentences": (medium_sentences / total_sentences * 100) if total_sentences else 0,
        "pct_long_sentences": (long_sentences / total_sentences * 100) if total_sentences else 0,
    }


def calculate_score(findings: dict) -> int:
    """Calculate a 0-100 score (100 = most human-like)."""
    high_count = findings["stats"]["high_severity"]
    medium_count = findings["stats"]["medium_severity"]
    total_words = findings["stats"]["total_words"]

    if total_words == 0:
        return 100

    # Penalize based on pattern density
    high_penalty = high_count * 10
    medium_penalty = medium_count * 3

    # Normalize by document length (per 100 words)
    penalty = (high_penalty + medium_penalty) / (total_words / 100)

    score = max(0, min(100, 100 - penalty))
    return int(score)


def get_grade(score: int) -> str:
    """Get grade description from score."""
    if score >= 90:
        return "Excellent - Very human-like"
    elif score >= 75:
        return "Good - Minor issues"
    elif score >= 60:
        return "Fair - Some LLM patterns"
    elif score >= 40:
        return "Needs work - Notable LLM patterns"
    else:
        return "High AI signal - Many LLM patterns"


def format_text(findings: dict, score: int, filename: str, verbose: bool = False, technical: bool = True) -> str:
    """Format findings as plain text."""
    lines = []
    stats = findings["stats"]

    lines.append("=" * 60)
    lines.append(f"Writing Analysis: {filename}")
    lines.append("=" * 60)
    lines.append(f"Words: {stats['total_words']:,}")
    lines.append(f"Patterns found: {stats['patterns_found']}")
    lines.append(f"  High severity: {stats['high_severity']}")
    lines.append(f"  Medium severity: {stats['medium_severity']}")
    lines.append("")

    # Score
    grade = get_grade(score)
    lines.append(f"Score: {score}/100 ({grade})")
    lines.append("")

    # Structural metrics
    para_threshold = 20 if technical else MIN_HEALTHY_PARA_LENGTH
    if "structural" in stats:
        struct = stats["structural"]
        lines.append("-" * 60)
        lines.append("STRUCTURE ANALYSIS")
        lines.append("-" * 60)
        lines.append(f"  Paragraphs: {struct['para_count']} (avg {struct['avg_para_words']:.0f} words each)")
        if struct['avg_para_words'] < para_threshold:
            lines.append(f"    WARNING: Short paragraphs suggest AI (aim for {para_threshold}+ words)")
        lines.append(f"  Sentences: {struct['sentence_count']} (avg {struct['avg_sentence_words']:.0f} words each)")
        lines.append(f"    Short (1-10): {struct['pct_short_sentences']:.0f}%  Medium (11-25): {struct['pct_medium_sentences']:.0f}%  Long (26+): {struct['pct_long_sentences']:.0f}%")
        if struct['list_items'] > 0:
            lines.append(f"  List items: {struct['list_items']}")
        lines.append("")

    # High severity findings
    if findings["high"]:
        lines.append("-" * 60)
        lines.append("HIGH SEVERITY (strongly suggests AI)")
        lines.append("-" * 60)
        for f in sorted(findings["high"], key=lambda x: -x.get("ratio", 0))[:15]:
            alt = f" -> {f['alternative']}" if f.get("alternative") else ""
            lines.append(f"  [{f['count']}x] \"{f['pattern']}\"{alt}")
            if f.get("context") and verbose:
                lines.append(f"       {f['context']}")
        lines.append("")

    # Medium severity findings
    if findings["medium"] and verbose:
        lines.append("-" * 60)
        lines.append("MEDIUM SEVERITY (moderately AI-like)")
        lines.append("-" * 60)
        for f in sorted(findings["medium"], key=lambda x: -x.get("ratio", 0))[:10]:
            alt = f" -> {f['alternative']}" if f.get("alternative") else ""
            lines.append(f"  [{f['count']}x] \"{f['pattern']}\"{alt}")
        lines.append("")

    # Summary by category
    if verbose and findings["by_category"]:
        lines.append("-" * 60)
        lines.append("BY CATEGORY")
        lines.append("-" * 60)
        for cat_type, cat_findings in sorted(findings["by_category"].items()):
            cat_name = CATEGORIES.get(cat_type, cat_type)
            total = sum(f["count"] for f in cat_findings)
            lines.append(f"  {cat_name}: {total} occurrences")
        lines.append("")

    # Suggestions
    if findings["high"]:
        lines.append("-" * 60)
        lines.append("SUGGESTIONS")
        lines.append("-" * 60)
        suggestions = set()
        for f in findings["high"][:5]:
            if f.get("alternative"):
                suggestions.add(f"Replace \"{f['pattern']}\" with: {f['alternative']}")
        for s in list(suggestions)[:5]:
            lines.append(f"  - {s}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_json(findings: dict, score: int, filename: str) -> str:
    """Format findings as JSON."""
    output = {
        "filename": filename,
        "score": score,
        "grade": get_grade(score),
        "stats": findings["stats"],
        "high_severity": findings["high"],
        "medium_severity": findings["medium"],
    }
    return json.dumps(output, indent=2)


def format_json_batch(results: list) -> str:
    """Format multiple file results as JSON."""
    output = {
        "files": results,
        "summary": {
            "total_files": len(results),
            "passing": sum(1 for r in results if r["score"] >= 60),
            "failing": sum(1 for r in results if r["score"] < 60),
            "average_score": sum(r["score"] for r in results) / len(results) if results else 0,
        }
    }
    return json.dumps(output, indent=2)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Writing Analysis Report</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .report { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; margin-bottom: 20px; font-size: 1.8em; }
        h2 { color: #34495e; margin: 25px 0 15px; font-size: 1.3em; border-bottom: 2px solid #eee; padding-bottom: 8px; }
        .score-gauge {
            display: flex;
            align-items: center;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .score-number {
            font-size: 3em;
            font-weight: bold;
        }
        .score-excellent { color: #27ae60; }
        .score-good { color: #2ecc71; }
        .score-fair { color: #f39c12; }
        .score-poor { color: #e67e22; }
        .score-bad { color: #e74c3c; }
        .score-bar {
            flex: 1;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; font-size: 0.9em; }
        .findings-section { margin: 20px 0; }
        .finding {
            display: flex;
            align-items: flex-start;
            gap: 15px;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            border-left: 4px solid;
        }
        .finding-high { background: #fdf2f2; border-color: #e74c3c; }
        .finding-medium { background: #fef6e7; border-color: #f39c12; }
        .finding-count {
            background: #fff;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        }
        .finding-high .finding-count { color: #e74c3c; }
        .finding-medium .finding-count { color: #f39c12; }
        .finding-content { flex: 1; }
        .finding-pattern { font-weight: 600; }
        .finding-alt { color: #27ae60; font-size: 0.9em; }
        .finding-context { color: #7f8c8d; font-size: 0.85em; margin-top: 4px; font-style: italic; }
        .collapsible { cursor: pointer; user-select: none; }
        .collapsible:hover { background: #f0f0f0; }
        .collapsible::before { content: '▼ '; font-size: 0.8em; }
        .collapsible.collapsed::before { content: '▶ '; }
        .content { display: block; }
        .content.hidden { display: none; }
        .suggestion {
            background: #e8f6e9;
            padding: 10px 15px;
            border-radius: 6px;
            margin: 8px 0;
            border-left: 4px solid #27ae60;
        }
        footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #7f8c8d; font-size: 0.85em; text-align: center; }
    </style>
</head>
<body>
    <div class="report">
        <h1>Writing Analysis Report</h1>
        <p><strong>File:</strong> {{ filename }}</p>

        <div class="score-gauge">
            <div class="score-number {{ score_class }}">{{ score }}</div>
            <div>
                <div style="margin-bottom: 8px; font-weight: 600;">{{ grade }}</div>
                <div class="score-bar">
                    <div class="score-fill {{ score_class }}" style="width: {{ score }}%; background: currentColor;"></div>
                </div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{{ stats.total_words }}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ stats.patterns_found }}</div>
                <div class="stat-label">Patterns Found</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #e74c3c;">{{ stats.high_severity }}</div>
                <div class="stat-label">High Severity</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #f39c12;">{{ stats.medium_severity }}</div>
                <div class="stat-label">Medium Severity</div>
            </div>
        </div>

        {% if structural %}
        <h2>Structure Analysis</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{{ structural.para_count }}</div>
                <div class="stat-label">Paragraphs</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ structural.avg_para_words|int }}</div>
                <div class="stat-label">Avg Words/Para</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ structural.sentence_count }}</div>
                <div class="stat-label">Sentences</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ structural.list_items }}</div>
                <div class="stat-label">List Items</div>
            </div>
        </div>
        {% endif %}

        {% if high_findings %}
        <h2 class="collapsible" onclick="toggleSection(this)">High Severity Findings ({{ high_findings|length }})</h2>
        <div class="content findings-section">
            {% for f in high_findings %}
            <div class="finding finding-high">
                <div class="finding-count">{{ f.count }}x</div>
                <div class="finding-content">
                    <div class="finding-pattern">"{{ f.pattern }}"</div>
                    {% if f.alternative %}<div class="finding-alt">→ {{ f.alternative }}</div>{% endif %}
                    {% if f.context %}<div class="finding-context">{{ f.context }}</div>{% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if medium_findings %}
        <h2 class="collapsible" onclick="toggleSection(this)">Medium Severity Findings ({{ medium_findings|length }})</h2>
        <div class="content findings-section">
            {% for f in medium_findings %}
            <div class="finding finding-medium">
                <div class="finding-count">{{ f.count }}x</div>
                <div class="finding-content">
                    <div class="finding-pattern">"{{ f.pattern }}"</div>
                    {% if f.alternative %}<div class="finding-alt">→ {{ f.alternative }}</div>{% endif %}
                    {% if f.context %}<div class="finding-context">{{ f.context }}</div>{% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if suggestions %}
        <h2>Suggestions</h2>
        {% for s in suggestions %}
        <div class="suggestion">{{ s }}</div>
        {% endfor %}
        {% endif %}

        <footer>
            Generated by Writing Checker | Score: {{ score }}/100
        </footer>
    </div>

    <script>
        function toggleSection(el) {
            el.classList.toggle('collapsed');
            el.nextElementSibling.classList.toggle('hidden');
        }
    </script>
</body>
</html>"""


def format_html(findings: dict, score: int, filename: str) -> str:
    """Format findings as HTML."""
    if not JINJA2_AVAILABLE:
        # Fallback: generate simple HTML without jinja2
        return generate_simple_html(findings, score, filename)

    template = Template(HTML_TEMPLATE)

    # Determine score class
    if score >= 90:
        score_class = "score-excellent"
    elif score >= 75:
        score_class = "score-good"
    elif score >= 60:
        score_class = "score-fair"
    elif score >= 40:
        score_class = "score-poor"
    else:
        score_class = "score-bad"

    # Build suggestions
    suggestions = []
    for f in findings["high"][:5]:
        if f.get("alternative"):
            suggestions.append(f'Replace "{f["pattern"]}" with: {f["alternative"]}')

    return template.render(
        filename=filename,
        score=score,
        grade=get_grade(score),
        score_class=score_class,
        stats=findings["stats"],
        structural=findings["stats"].get("structural"),
        high_findings=sorted(findings["high"], key=lambda x: -x.get("ratio", 0))[:15],
        medium_findings=sorted(findings["medium"], key=lambda x: -x.get("ratio", 0))[:10],
        suggestions=suggestions,
    )


def generate_simple_html(findings: dict, score: int, filename: str) -> str:
    """Generate simple HTML without jinja2."""
    stats = findings["stats"]
    grade = get_grade(score)

    html = f"""<!DOCTYPE html>
<html><head><title>Writing Analysis: {filename}</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px}}
.high{{color:#e74c3c}}.medium{{color:#f39c12}}</style></head>
<body><h1>Writing Analysis: {filename}</h1>
<p><strong>Score:</strong> {score}/100 ({grade})</p>
<p><strong>Words:</strong> {stats['total_words']} | <strong>Patterns:</strong> {stats['patterns_found']} |
<span class="high">High: {stats['high_severity']}</span> |
<span class="medium">Medium: {stats['medium_severity']}</span></p>
<h2 class="high">High Severity</h2><ul>"""

    for f in sorted(findings["high"], key=lambda x: -x.get("ratio", 0))[:15]:
        alt = f' → {f["alternative"]}' if f.get("alternative") else ""
        html += f'<li><strong>{f["count"]}x</strong> "{f["pattern"]}"{alt}</li>'

    html += "</ul><h2 class='medium'>Medium Severity</h2><ul>"

    for f in sorted(findings["medium"], key=lambda x: -x.get("ratio", 0))[:10]:
        alt = f' → {f["alternative"]}' if f.get("alternative") else ""
        html += f'<li><strong>{f["count"]}x</strong> "{f["pattern"]}"{alt}</li>'

    html += "</ul></body></html>"
    return html


def print_report(findings: dict, filename: str, verbose: bool = False, technical: bool = True):
    """Print human-readable report (legacy wrapper)."""
    score = calculate_score(findings)
    print(format_text(findings, score, filename, verbose, technical))


def print_json(findings: dict, filename: str):
    """Print JSON output (legacy wrapper)."""
    score = calculate_score(findings)
    print(format_json(findings, score, filename))


def highlight_match(text: str, pattern: str, context_chars: int = 60) -> str:
    """Highlight pattern match in context."""
    match = re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE)
    if not match:
        return text[:context_chars * 2]

    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)

    before = text[start:match.start()]
    matched = text[match.start():match.end()]
    after = text[match.end():end]

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""

    return f"{prefix}{before}\033[1;31m{matched}\033[0m{after}{suffix}"


def apply_replacement(text: str, pattern: str, replacement: str, occurrence: int = 0) -> str:
    """Replace a specific occurrence of pattern with replacement."""
    regex = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
    matches = list(regex.finditer(text))

    if occurrence >= len(matches):
        return text

    match = matches[occurrence]
    return text[:match.start()] + replacement + text[match.end():]


def interactive_mode(text: str, findings: dict, filepath: str) -> str:
    """Process findings interactively, return modified text."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print(f"File: {filepath}")
    print("Commands: [a]ccept suggestion, [s]kip, [e]dit manually, [q]uit")
    print("=" * 60 + "\n")

    modified_text = text
    changes_made = 0
    total_findings = len(findings["high"]) + len(findings["medium"])

    # Combine and sort findings by severity and ratio
    all_findings = []
    for f in findings["high"]:
        f["_severity_order"] = 0
        all_findings.append(f)
    for f in findings["medium"]:
        f["_severity_order"] = 1
        all_findings.append(f)

    all_findings.sort(key=lambda x: (x["_severity_order"], -x.get("ratio", 0)))

    processed = 0
    for finding in all_findings:
        pattern = finding["pattern"]
        severity = finding["severity"]
        alternative = finding.get("alternative")

        # Skip structural/aggregate findings that can't be fixed with replacement
        if finding["type"] in ("structure", "hedging", "punctuation"):
            continue

        # Check if pattern still exists in modified text
        if not re.search(r'\b' + re.escape(pattern) + r'\b', modified_text, re.IGNORECASE):
            continue

        processed += 1
        severity_color = "\033[1;31m" if severity == "high" else "\033[1;33m"

        print(f"\n[{processed}/{total_findings}] {severity_color}{severity.upper()}\033[0m: \"{pattern}\"")

        # Show context
        context = highlight_match(modified_text, pattern)
        print(f"Context: {context}")

        if alternative:
            # Parse alternatives (comma-separated)
            alts = [a.strip() for a in alternative.split(",")]
            print(f"Suggestions: {', '.join(alts)}")

        while True:
            try:
                choice = input("\n[a]ccept / [s]kip / [e]dit / [q]uit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice == "q":
                print("\nQuitting interactive mode...")
                if changes_made > 0:
                    return modified_text, changes_made
                return text, 0

            elif choice == "s":
                print("Skipped.")
                break

            elif choice == "a":
                if not alternative:
                    print("No suggestion available. Use [e]dit to provide replacement.")
                    continue

                # Use first alternative
                alts = [a.strip() for a in alternative.split(",")]
                replacement = alts[0]

                # Handle special cases in suggestions
                if replacement.startswith("(") and replacement.endswith(")"):
                    # Things like "(delete)" mean remove it
                    replacement = ""
                elif "(or delete)" in replacement.lower():
                    # "basically (or delete)" -> "basically"
                    replacement = re.sub(r'\s*\(or delete\)\s*', '', replacement, flags=re.IGNORECASE).strip()
                elif replacement.endswith(")") and "(" in replacement:
                    # Strip trailing parenthetical notes like "word (note)"
                    replacement = re.sub(r'\s*\([^)]+\)\s*$', '', replacement).strip()

                old_text = modified_text
                modified_text = apply_replacement(modified_text, pattern, replacement)

                if modified_text != old_text:
                    changes_made += 1
                    if replacement:
                        print(f"\033[31m- {pattern}\033[0m")
                        print(f"\033[32m+ {replacement}\033[0m")
                    else:
                        print(f"\033[31m- {pattern}\033[0m (deleted)")
                break

            elif choice == "e":
                try:
                    replacement = input("Enter replacement text: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("Cancelled.")
                    continue

                old_text = modified_text
                modified_text = apply_replacement(modified_text, pattern, replacement)

                if modified_text != old_text:
                    changes_made += 1
                    print(f"\033[31m- {pattern}\033[0m")
                    print(f"\033[32m+ {replacement}\033[0m")
                break

            else:
                print("Invalid choice. Use a/s/e/q")

    # Show summary
    print("\n" + "=" * 60)
    print(f"Changes made: {changes_made}")

    if changes_made > 0:
        # Recalculate score (use default technical=True for interactive mode)
        data = load_markers()
        markers = data.get("markers", [])
        new_findings = check_text(modified_text, markers, technical=True)
        new_score = calculate_score(new_findings)
        old_score = calculate_score(findings)

        print(f"Score: {old_score} -> {new_score}")

        while True:
            try:
                save = input(f"\nSave changes to {filepath}? [y/n] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                save = "n"

            if save == "y":
                return modified_text, changes_made
            elif save == "n":
                print("Changes discarded.")
                return text, 0
            else:
                print("Please enter y or n")
    else:
        print("No changes made.")

    return text, 0


def process_single_file(
    filepath: str,
    markers: list,
    verbose: bool,
    output_format: str,
    config: dict,
    interactive: bool = False,
) -> dict:
    """Process a single file and return results."""
    path = Path(filepath)

    if not path.exists():
        return {"filename": filepath, "error": f"File not found: {filepath}"}

    # Check exclusions
    if should_exclude_file(filepath, config.get("exclude", [])):
        return {"filename": filepath, "excluded": True}

    try:
        text = read_file(path)
    except Exception as e:
        return {"filename": filepath, "error": str(e)}

    technical = config.get("technical", True)
    findings = check_text(text, markers, verbose=verbose, technical=technical)

    # Filter out ignored patterns
    ignore_patterns = config.get("ignore_patterns", [])
    if ignore_patterns:
        for severity in ["high", "medium", "low"]:
            findings[severity] = [
                f for f in findings[severity]
                if f["pattern"].lower() not in [p.lower() for p in ignore_patterns]
            ]
        # Recalculate stats
        findings["stats"]["high_severity"] = len(findings["high"])
        findings["stats"]["medium_severity"] = len(findings["medium"])
        findings["stats"]["patterns_found"] = (
            findings["stats"]["high_severity"] + findings["stats"]["medium_severity"]
        )

    score = calculate_score(findings)

    # Interactive mode
    if interactive:
        modified_text, changes = interactive_mode(text, findings, filepath)
        if changes > 0:
            path.write_text(modified_text)
            print(f"Saved {changes} changes to {filepath}")
            # Re-analyze
            findings = check_text(modified_text, markers, verbose=verbose, technical=technical)
            score = calculate_score(findings)

    return {
        "filename": filepath,
        "score": score,
        "grade": get_grade(score),
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check your writing for LLM patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_writing.py document.md
  python check_writing.py document.md --verbose
  python check_writing.py document.md --format json
  python check_writing.py document.md --format html > report.html
  python check_writing.py document.md --interactive
  python check_writing.py *.md docs/*.md --format text
  cat document.md | python check_writing.py --stdin
        """
    )
    parser.add_argument("files", nargs="*", help="Files to check")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Review findings interactively and apply fixes"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config file (default: .prose-check.yaml)"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum score to pass (overrides config)"
    )
    parser.add_argument(
        "--no-technical",
        action="store_true",
        help="Don't exclude technical terms (stricter checking for prose)"
    )
    # Legacy support for --json
    parser.add_argument("--json", "-j", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Load config
    config = load_config(args.config)
    if args.min_score is not None:
        config["min_score"] = args.min_score
    if args.no_technical:
        config["technical"] = False

    min_score = config["min_score"]

    # Validate interactive mode constraints
    if args.interactive:
        if args.stdin:
            print("Error: --interactive requires file input (not stdin)", file=sys.stderr)
            sys.exit(1)
        if args.format != "text":
            print("Warning: --interactive ignores --format, using interactive display", file=sys.stderr)

    # Load markers
    data = load_markers()
    markers = data.get("markers", [])

    # Handle stdin
    if args.stdin:
        text = sys.stdin.read()
        technical = config.get("technical", True)
        findings = check_text(text, markers, verbose=args.verbose, technical=technical)
        score = calculate_score(findings)

        if args.format == "json":
            print(format_json(findings, score, "<stdin>"))
        elif args.format == "html":
            print(format_html(findings, score, "<stdin>"))
        else:
            print(format_text(findings, score, "<stdin>", args.verbose, technical))

        sys.exit(0 if score >= min_score else 1)

    # Handle file(s)
    if not args.files:
        parser.print_help()
        sys.exit(1)

    # Process files
    results = []
    any_failed = False

    for filepath in args.files:
        result = process_single_file(
            filepath=filepath,
            markers=markers,
            verbose=args.verbose,
            output_format=args.format,
            config=config,
            interactive=args.interactive,
        )
        results.append(result)

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            any_failed = True
        elif result.get("excluded"):
            if args.verbose:
                print(f"Excluded: {filepath}", file=sys.stderr)
        elif result.get("score", 100) < min_score:
            any_failed = True

    # Filter out excluded/error results for output
    valid_results = [r for r in results if "findings" in r]

    if not valid_results:
        print("No valid files to check.", file=sys.stderr)
        sys.exit(1)

    # Output based on format
    if args.interactive:
        # Interactive mode handles its own output
        pass
    elif args.format == "json":
        if len(valid_results) == 1:
            r = valid_results[0]
            print(format_json(r["findings"], r["score"], r["filename"]))
        else:
            # Batch JSON output
            batch_output = []
            for r in valid_results:
                batch_output.append({
                    "filename": r["filename"],
                    "score": r["score"],
                    "grade": r["grade"],
                    "stats": r["findings"]["stats"],
                    "high_severity": r["findings"]["high"],
                    "medium_severity": r["findings"]["medium"],
                })
            print(format_json_batch(batch_output))
    elif args.format == "html":
        if len(valid_results) == 1:
            r = valid_results[0]
            print(format_html(r["findings"], r["score"], r["filename"]))
        else:
            # For multiple files in HTML, generate a summary page
            print("<!DOCTYPE html><html><head><title>Batch Report</title>")
            print("<style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px}")
            print(".pass{color:#27ae60}.fail{color:#e74c3c}</style></head><body>")
            print("<h1>Batch Writing Analysis</h1>")
            print(f"<p>Files: {len(valid_results)} | ")
            passing = sum(1 for r in valid_results if r["score"] >= min_score)
            print(f'<span class="pass">Passing: {passing}</span> | ')
            print(f'<span class="fail">Failing: {len(valid_results) - passing}</span></p>')
            print("<table border='1' cellpadding='8' cellspacing='0'>")
            print("<tr><th>File</th><th>Score</th><th>Grade</th><th>High</th><th>Medium</th></tr>")
            for r in valid_results:
                status_class = "pass" if r["score"] >= min_score else "fail"
                stats = r["findings"]["stats"]
                print(f"<tr class='{status_class}'>")
                print(f"<td>{r['filename']}</td>")
                print(f"<td>{r['score']}</td>")
                print(f"<td>{r['grade']}</td>")
                print(f"<td>{stats['high_severity']}</td>")
                print(f"<td>{stats['medium_severity']}</td>")
                print("</tr>")
            print("</table></body></html>")
    else:
        # Text format
        technical = config.get("technical", True)
        for r in valid_results:
            print(format_text(r["findings"], r["score"], r["filename"], args.verbose, technical))
            if len(valid_results) > 1:
                print()  # Blank line between files

        # Summary for batch
        if len(valid_results) > 1:
            print("=" * 60)
            print("BATCH SUMMARY")
            print("=" * 60)
            print(f"Files checked: {len(valid_results)}")
            passing = sum(1 for r in valid_results if r["score"] >= min_score)
            print(f"Passing (>={min_score}): {passing}")
            print(f"Failing (<{min_score}): {len(valid_results) - passing}")
            avg_score = sum(r["score"] for r in valid_results) / len(valid_results)
            print(f"Average score: {avg_score:.1f}")
            print("=" * 60)

    # Exit code
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
