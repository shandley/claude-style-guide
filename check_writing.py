#!/usr/bin/env python3
"""
Writing Checker - Scan your documents for LLM-isms.

Usage:
    python check_writing.py document.md
    python check_writing.py document.md --verbose
    python check_writing.py document.md --json
    cat document.md | python check_writing.py --stdin
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Path to markers file
MARKERS_PATH = Path(__file__).parent / "results" / "markers.json"

# Severity thresholds (log-odds)
HIGH_SEVERITY = 2.5
MEDIUM_SEVERITY = 1.5

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


def load_markers() -> dict:
    """Load markers from analysis results."""
    if not MARKERS_PATH.exists():
        print(f"Error: Markers file not found at {MARKERS_PATH}", file=sys.stderr)
        print("Run 'python run_pipeline.py analyze' first.", file=sys.stderr)
        sys.exit(1)

    with open(MARKERS_PATH) as f:
        return json.load(f)


def get_severity(log_odds: float) -> str:
    """Get severity level from log-odds."""
    if log_odds >= HIGH_SEVERITY:
        return "high"
    elif log_odds >= MEDIUM_SEVERITY:
        return "medium"
    return "low"


def check_text(text: str, markers: list, verbose: bool = False) -> dict:
    """
    Check text for LLM patterns.

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

    text_lower = text.lower()

    # Check each marker
    for marker in markers:
        item = marker["item"]
        marker_type = marker["type"]
        log_odds = marker["log_odds"]

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

    return findings


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


def print_report(findings: dict, filename: str, verbose: bool = False):
    """Print human-readable report."""
    stats = findings["stats"]
    score = calculate_score(findings)

    print("=" * 60)
    print(f"Writing Analysis: {filename}")
    print("=" * 60)
    print(f"Words: {stats['total_words']:,}")
    print(f"Patterns found: {stats['patterns_found']}")
    print(f"  High severity: {stats['high_severity']}")
    print(f"  Medium severity: {stats['medium_severity']}")
    print()

    # Score
    if score >= 90:
        grade = "Excellent - Very human-like"
    elif score >= 75:
        grade = "Good - Minor issues"
    elif score >= 60:
        grade = "Fair - Some LLM patterns"
    elif score >= 40:
        grade = "Needs work - Notable LLM patterns"
    else:
        grade = "High AI signal - Many LLM patterns"

    print(f"Score: {score}/100 ({grade})")
    print()

    # High severity findings
    if findings["high"]:
        print("-" * 60)
        print("HIGH SEVERITY (strongly suggests AI)")
        print("-" * 60)
        for f in sorted(findings["high"], key=lambda x: -x.get("ratio", 0))[:15]:
            alt = f" -> {f['alternative']}" if f.get("alternative") else ""
            print(f"  [{f['count']}x] \"{f['pattern']}\"{alt}")
            if f.get("context") and verbose:
                print(f"       {f['context']}")
        print()

    # Medium severity findings
    if findings["medium"] and verbose:
        print("-" * 60)
        print("MEDIUM SEVERITY (moderately AI-like)")
        print("-" * 60)
        for f in sorted(findings["medium"], key=lambda x: -x.get("ratio", 0))[:10]:
            alt = f" -> {f['alternative']}" if f.get("alternative") else ""
            print(f"  [{f['count']}x] \"{f['pattern']}\"{alt}")
        print()

    # Summary by category
    if verbose and findings["by_category"]:
        print("-" * 60)
        print("BY CATEGORY")
        print("-" * 60)
        for cat_type, cat_findings in sorted(findings["by_category"].items()):
            cat_name = CATEGORIES.get(cat_type, cat_type)
            total = sum(f["count"] for f in cat_findings)
            print(f"  {cat_name}: {total} occurrences")
        print()

    # Suggestions
    if findings["high"]:
        print("-" * 60)
        print("SUGGESTIONS")
        print("-" * 60)
        suggestions = set()
        for f in findings["high"][:5]:
            if f.get("alternative"):
                suggestions.add(f"Replace \"{f['pattern']}\" with: {f['alternative']}")
        for s in list(suggestions)[:5]:
            print(f"  - {s}")
        print()

    print("=" * 60)


def print_json(findings: dict, filename: str):
    """Print JSON output."""
    output = {
        "filename": filename,
        "score": calculate_score(findings),
        "stats": findings["stats"],
        "high_severity": findings["high"],
        "medium_severity": findings["medium"],
    }
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Check your writing for LLM patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_writing.py document.md
  python check_writing.py document.md --verbose
  python check_writing.py document.md --json
  cat document.md | python check_writing.py --stdin
        """
    )
    parser.add_argument("file", nargs="?", help="File to check")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Get input text
    if args.stdin:
        text = sys.stdin.read()
        filename = "<stdin>"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text()
        filename = args.file
    else:
        parser.print_help()
        sys.exit(1)

    # Load markers and check
    data = load_markers()
    markers = data.get("markers", [])

    findings = check_text(text, markers, verbose=args.verbose)

    # Output
    if args.json:
        print_json(findings, filename)
    else:
        print_report(findings, filename, verbose=args.verbose)

    # Exit code based on score
    score = calculate_score(findings)
    if score < 60:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
