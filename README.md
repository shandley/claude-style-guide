# Claude Style Guide

A pipeline to analyze Claude model writing patterns and generate a personal styleguide for avoiding common LLM-isms in your own writing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This tool statistically compares AI-generated text against human writing to identify distinctive patterns. The output is an actionable styleguide with:

- Words and phrases to avoid
- Human alternatives for each pattern
- Self-editing checklist
- Before/after rewrite examples

Now includes multi-model comparison to track how patterns have evolved across Claude versions.

## Installation

**Install from GitHub:**

```bash
pip install git+https://github.com/shandley/claude-style-guide.git
```

**Or clone and install locally:**

```bash
git clone https://github.com/shandley/claude-style-guide.git
cd claude-style-guide
pip install -e .
```

## Quick Start

**Check your writing for AI patterns:**

```bash
# After pip install
prose-check document.md

# Or run directly
python check_writing.py document.md
```

**Run the full analysis pipeline** (generates custom markers from Claude samples):

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Run pipeline (~200 samples, ~$1-2 API cost)
python run_pipeline.py all --verbose
```

Output: `results/styleguide.md`

## Commands

### Basic Pipeline

```bash
python run_pipeline.py all --verbose      # Full pipeline
python run_pipeline.py status             # Check progress
python run_pipeline.py clean              # Remove generated files
```

### Individual Steps

```bash
python run_pipeline.py generate-prompts --n 300
python run_pipeline.py generate-samples --n 200
python run_pipeline.py fetch-human-corpus --n 10000
python run_pipeline.py analyze
python run_pipeline.py report
```

### Multi-Model Comparison

Compare how writing patterns have evolved across Claude versions:

```bash
# List available models
python run_pipeline.py list-models

# Generate samples from a specific model
python run_pipeline.py generate-samples --model sonnet-3.7 --n 100

# Compare patterns across all sampled models
python run_pipeline.py compare-models --verbose
```

Available models:
- `opus-4.5` - Claude Opus 4.5 (2025-11)
- `sonnet-4` - Claude Sonnet 4 (2025-05)
- `sonnet-3.7` - Claude 3.7 Sonnet (2025-02)
- `haiku-3.5` - Claude 3.5 Haiku (2024-10)
- `haiku-3` - Claude 3 Haiku (2024-03)

### Check Your Writing

Analyze any document for AI writing patterns:

```bash
# Check a text or markdown file
python check_writing.py document.md

# Check multiple files
python check_writing.py README.md docs/*.md

# Different output formats
python check_writing.py document.md --format text   # Default
python check_writing.py document.md --format json   # Machine-readable
python check_writing.py document.md --format html > report.html

# Interactive mode - review and fix findings one by one
python check_writing.py document.md --interactive

# Verbose output with low-severity findings
python check_writing.py document.md --verbose

# Strict mode for essays/articles (flags technical terms)
python check_writing.py essay.md --no-technical

# Read from stdin
echo "This is a comprehensive overview." | python check_writing.py --stdin
```

Output includes a score (0-100, higher = more human-like) and flags for:
- Overused punctuation (em dashes, colons, semicolons)
- AI-favored words and phrases
- Structural issues (short paragraphs, bullet overuse)

#### CLI Options

| Flag | Description |
|------|-------------|
| `--format {text,json,html}` | Output format (default: text) |
| `--interactive`, `-i` | Review findings interactively and apply fixes |
| `--verbose`, `-v` | Include low-severity findings |
| `--no-technical` | Stricter mode - don't exclude technical terms |
| `--stdin` | Read text from stdin instead of file |
| `--config PATH` | Custom config file path |

#### Configuration File

Create `.prose-check.yaml` in your project root:

```yaml
# Minimum score to pass (0-100, default: 60)
min_score: 70

# Exclude technical terms (default: true)
# Set to false for stricter prose checking (essays, articles)
technical: true

# Files/patterns to exclude
exclude:
  - "CHANGELOG.md"
  - "vendor/**"
  - "node_modules/**"

# Patterns to whitelist (by exact name)
ignore_patterns:
  - "comprehensive"  # We use this intentionally
```

#### CI Integration

**Pre-commit hook** - Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: prose-check
        name: Check prose for AI patterns
        entry: python check_writing.py
        language: python
        types: [markdown]
        pass_filenames: true
        additional_dependencies:
          - pyyaml>=6.0
          - jinja2>=3.0
```

**GitHub Action** - The repo includes `.github/workflows/prose-check.yml` that automatically comments on PRs with prose analysis results.

## How It Works

1. **Generate prompts** - Creates diverse prompts weighted toward technical/professional writing
2. **Collect samples** - Calls Claude API to generate responses
3. **Fetch human corpus** - Downloads comparison text from Wikipedia, OpenWebText, C4
4. **Analyze** - Computes log-odds ratios to find statistically significant patterns
5. **Report** - Generates markdown styleguide with actionable guidance

## Analysis Capabilities

The analyzer examines multiple dimensions of writing style:

| Analysis | What It Detects |
|----------|-----------------|
| **Lexical** | Overused words, bigrams, trigrams |
| **Punctuation** | Em dash, colon, semicolon frequency |
| **Sentence starters** | Formulaic openings ("This document...", "Comprehensive...") |
| **Transitions** | Formal vs casual transition word usage |
| **Hedging** | Qualifying words (typically, often, sometimes) |
| **Structure** | Paragraph length, list density, sentence distribution |
| **Passive voice** | Passive construction frequency |

## Output

### results/styleguide.md

- Top distinctive patterns ranked by log-odds ratio
- Categorized phrases: hedging, transitions, fillers, LLM favorites
- Structural analysis: sentence length, list usage, punctuation
- Human alternatives for each flagged pattern

### results/model_comparison.md

- Side-by-side pattern comparison across model versions
- Punctuation evolution (em dash, colon, semicolon usage)
- Phrase frequency trends over time

## Key Findings

### Multi-Model Comparison (January 2026)

**Em dash overuse is Opus 4.5 specific:**

| Model | Em Dash vs Human |
|-------|------------------|
| Haiku 3 | 0.0x |
| Sonnet 3.7 | 0.8x |
| Sonnet 4 | 0.9x |
| **Opus 4.5** | **16.8x** |

Other models are at or below human levels for em dashes.

**Each model has different word quirks:**

| Word | Worst Offender |
|------|----------------|
| "robust" | Haiku 3 (43x) |
| "nuanced" | Sonnet 4 (56x) |
| "comprehensive" | Haiku 3 (40x) |
| "paradigm" | Sonnet 4 (37x) |

### Opus 4.5 vs Human (200 samples vs 6000 texts)

**Punctuation:**
- Em dash: 16.9x more common
- Colon: 4.1x more common
- Semicolon: 3.1x more common

**Structure:**
- Avg paragraph: 16 words (human: 210 words)
- Passive voice: 4.7% (human: 14.9%)
- List items/doc: 9.5 (human: ~0)

**Phrases to avoid:**
- "comprehensive" (24x)
- "fundamentally" (17x)
- "nuanced" (17x)
- "paradigm" (15x)

### Deep Analysis Findings

**Sentence Starters** - AI overuses formulaic openings:
- "This document..." (623x)
- "Comprehensive..." (680x)
- "Introduction..." (58x)

**Transitions (Counterintuitive!)** - AI uses FEWER transitions than humans:
- Formal: AI 0.3 vs Human 0.9 per 100 sentences
- Only "conversely" (50x) and "nevertheless" (8x) are AI-overused

**Hedging Words** - AI overuses qualifying language:
- "typically" (9.6x), "often" (4.9x), "sometimes" (4.2x), "usually" (3.4x)

## Requirements

- Python 3.10+
- Anthropic API key
- ~$1-2 for 200 samples from one model

## Claude Code Skill

This repo includes a Claude Code skill for proactive writing guidance. When installed, Claude will automatically apply human writing patterns when generating prose content.

**Location:** `.claude/skills/human-writing/`

The skill includes:
- Model-specific pattern warnings
- Words and phrases to avoid with alternatives
- Structural guidelines (paragraph length, list usage)
- Statistical reference data

## Project Structure

```
opus-styleguide/
├── src/
│   ├── generate_prompts.py    # Prompt generation
│   ├── generate_samples.py    # Multi-model API sampling
│   ├── fetch_human_corpus.py  # Human text download
│   ├── analyze.py             # Statistical analysis
│   ├── compare.py             # Cross-model comparison
│   └── report.py              # Styleguide generation
├── tests/
│   └── test_check_writing.py  # Unit tests (30 tests)
├── .claude/skills/
│   └── human-writing/         # Claude Code skill
│       ├── SKILL.md           # Writing guidelines
│       └── statistics.md      # Reference data
├── .github/workflows/
│   └── prose-check.yml        # GitHub Action for PR checks
├── data/                      # Generated data (gitignored)
├── results/                   # Output reports
├── run_pipeline.py            # CLI entry point
├── check_writing.py           # Document checker CLI
├── pyproject.toml             # Package configuration
├── .prose-check.yaml          # Default config
├── .pre-commit-hooks.yaml     # Pre-commit hook definition
└── requirements.txt
```

## License

MIT
