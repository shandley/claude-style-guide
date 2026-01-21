# Claude Model Evolution: Writing Pattern Comparison

*Generated 2026-01-21*

## Models Analyzed

| Model | Samples | Words |
|-------|---------|-------|
| haiku-3 | 51 | 15,603 |
| sonnet-4 | 51 | 16,028 |
| opus-4.5 | 201 | 47,104 |
| sonnet-3.7 | 50 | 12,601 |

## Punctuation Patterns

How punctuation usage has changed across model versions:

| Punctuation | Human | haiku-3 | sonnet-4 | opus-4.5 | sonnet-3.7 |
|-------------|-------|-------|-------|-------|-------|
| em dash | 0.28 | 0.00 (0.0x) | 0.25 (0.9x) | 4.78 (16.8x) | 0.23 (0.8x) |
| colon | 1.01 | 2.43 (2.4x) | 4.68 (4.6x) | 4.11 (4.1x) | 3.90 (3.9x) |
| semicolon | 0.23 | 0.01 (0.0x) | 0.41 (1.8x) | 0.69 (3.0x) | 0.30 (1.3x) |

*Values are occurrences per 1,000 characters. Ratio vs human in parentheses.*

## LLM Phrase Patterns

How distinctive phrases have evolved:

| Phrase | Human | haiku-3 | sonnet-4 | opus-4.5 | sonnet-3.7 |
|--------|-------|-------|-------|-------|-------|
| comprehensive | 0.06 | 2.24 (39.5x) | 2.15 (37.9x) | 1.39 (24.4x) | 1.71 (30.1x) |
| essentially | 0.04 | 0 | 0.22 (6.2x) | 0.11 (3.1x) | 0.19 (5.3x) |
| fundamentally | 0.01 | 0 | 0 | 0.13 (17.0x) | 0.19 (24.0x) |
| in essence | 0.00 | 0.1 (high) | 0 | 0.0 (high) | 0 |
| nuanced | 0.00 | 0 | 0.15 (56.1x) | 0.04 (17.0x) | 0.3 (high) |
| paradigm | 0.01 | 0.09 (7.5x) | 0.44 (37.4x) | 0.18 (15.1x) | 0.28 (24.0x) |
| robust | 0.03 | 1.25 (43.2x) | 0.22 (7.7x) | 0.09 (3.1x) | 0.19 (6.5x) |

*Values are occurrences per 10,000 characters.*

## Evolution Trends

### haiku-3 vs sonnet-3.7


## Methodology

Each model was prompted with the same set of technical writing prompts.
Patterns were counted and normalized by corpus size for comparison.
Human baseline from Wikipedia, OpenWebText, and C4 datasets.
