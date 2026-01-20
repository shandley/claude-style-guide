# Claude Style Guide

A pipeline to analyze Claude Opus 4.5's writing patterns and generate a personal styleguide for avoiding common LLM-isms in your own writing.

## Overview

This tool statistically compares AI-generated text against human writing to identify distinctive patterns. The output is an actionable styleguide with:

- Words and phrases to avoid
- Human alternatives for each pattern
- Self-editing checklist
- Before/after rewrite examples

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

```bash
python run_pipeline.py all --verbose      # Full pipeline
python run_pipeline.py status             # Check progress
python run_pipeline.py clean              # Remove generated files

# Individual steps
python run_pipeline.py generate-prompts --n 300
python run_pipeline.py generate-samples --n 200
python run_pipeline.py fetch-human-corpus --n 10000
python run_pipeline.py analyze
python run_pipeline.py report
```

## How It Works

1. **Generate prompts** - Creates diverse prompts weighted toward technical/professional writing
2. **Collect Opus samples** - Calls Claude Opus 4.5 API to generate responses
3. **Fetch human corpus** - Downloads comparison text from Wikipedia, OpenWebText, C4
4. **Analyze** - Computes log-odds ratios to find statistically significant patterns
5. **Report** - Generates markdown styleguide with actionable guidance

## Output

The styleguide includes:

- Top distinctive patterns ranked by log-odds ratio
- Categorized phrases: hedging, transitions, fillers, LLM favorites
- Structural analysis: sentence length, list usage, punctuation
- Human alternatives for each flagged pattern

## Requirements

- Python 3.10+
- Anthropic API key
- ~$1-2 for 200 samples

## Project Structure

```
opus-styleguide/
├── src/
│   ├── generate_prompts.py
│   ├── generate_samples.py
│   ├── fetch_human_corpus.py
│   ├── analyze.py
│   └── report.py
├── data/                    # Generated data (gitignored)
├── results/                 # Output styleguide
├── run_pipeline.py
└── requirements.txt
```

## License

MIT
