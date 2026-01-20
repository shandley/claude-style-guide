# Claude Style Guide

A pipeline to analyze Claude model writing patterns and generate a personal styleguide for avoiding common LLM-isms in your own writing.

## Overview

This tool statistically compares AI-generated text against human writing to identify distinctive patterns. The output is an actionable styleguide with:

- Words and phrases to avoid
- Human alternatives for each pattern
- Self-editing checklist
- Before/after rewrite examples

Now includes multi-model comparison to track how patterns have evolved across Claude versions.

## Quick Start

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
python run_pipeline.py generate-samples --model sonnet-3.5 --n 100

# Generate samples from multiple models at once
python run_pipeline.py generate-all-models --n 100 --models "opus-3,sonnet-3.5,opus-4.5"

# Compare patterns across all sampled models
python run_pipeline.py compare-models --verbose
```

Available models:
- `opus-4.5` - Claude Opus 4.5 (2025-11)
- `sonnet-4` - Claude Sonnet 4 (2025-05)
- `sonnet-3.5` - Claude 3.5 Sonnet (2024-10)
- `opus-3` - Claude 3 Opus (2024-02)
- `sonnet-3` - Claude 3 Sonnet (2024-02)
- `haiku-3` - Claude 3 Haiku (2024-03)

## How It Works

1. **Generate prompts** - Creates diverse prompts weighted toward technical/professional writing
2. **Collect samples** - Calls Claude API to generate responses
3. **Fetch human corpus** - Downloads comparison text from Wikipedia, OpenWebText, C4
4. **Analyze** - Computes log-odds ratios to find statistically significant patterns
5. **Report** - Generates markdown styleguide with actionable guidance

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

From analysis of 200 Opus 4.5 samples vs 6000 human texts:

**Punctuation:**
- Em dash: 16.9x more common in AI
- Colon: 4.1x more common in AI
- Semicolon: 3.1x more common in AI

**Phrases to avoid:**
- "comprehensive" (24x)
- "in essence" (68x)
- "fundamentally" (16x)
- "nuanced" (14x)

## Requirements

- Python 3.10+
- Anthropic API key
- ~$1-2 for 200 samples from one model

## Project Structure

```
claude-style-guide/
├── src/
│   ├── generate_prompts.py    # Prompt generation
│   ├── generate_samples.py    # Multi-model API sampling
│   ├── fetch_human_corpus.py  # Human text download
│   ├── analyze.py             # Statistical analysis
│   ├── compare.py             # Cross-model comparison
│   └── report.py              # Styleguide generation
├── data/                      # Generated data (gitignored)
├── results/                   # Output reports
├── run_pipeline.py            # CLI entry point
└── requirements.txt
```

## License

MIT
