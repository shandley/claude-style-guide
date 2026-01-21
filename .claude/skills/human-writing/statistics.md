# Statistical Reference

Data from analysis of 200 Claude Opus 4.5 samples vs 6,000 human texts.

## Punctuation Overuse

| Punctuation | AI Rate (per 1k chars) | Human Rate | Ratio |
|-------------|------------------------|------------|-------|
| Em dash (—) | 4.79 | 0.28 | 16.9x |
| Colon | 4.12 | 1.01 | 4.1x |
| Semicolon | 0.69 | 0.23 | 3.1x |

## Sentence Length Distribution

| Metric | AI | Human |
|--------|-----|-------|
| Mean length | 23.9 words | 19.8 words |
| Coefficient of variation | 137.1% | 70.5% |
| Short sentences (1-10 words) | 39.9% | 23.9% |
| Medium sentences (11-25 words) | 34.1% | 50.3% |
| Long sentences (26+ words) | 26.1% | 25.8% |

Key insight: AI has MORE variation, but it's bimodal (alternating very short/very long). Human writing clusters around medium length.

## Passive Voice

| Metric | AI | Human |
|--------|-----|-------|
| Passive voice usage | 4.7% | 14.9% |

Key insight: AI uses LESS passive voice than humans. Don't over-correct.

## Paragraph Structure

| Metric | AI | Human |
|--------|-----|-------|
| Avg paragraph length | 16.2 words | 209.6 words |
| Paragraphs per document | 18.4 | ~1 |

Key insight: AI fragments into many short paragraphs. Combine related ideas.

## List Usage

| Metric | AI | Human |
|--------|-----|-------|
| List items per document | 9.5 | ~0 |

Key insight: AI overuses bullets. Convert to prose when possible.

## Most Distinctive Words/Phrases

| Pattern | AI vs Human Ratio |
|---------|-------------------|
| comprehensive | 24.5x |
| in essence | very high |
| fundamentally | 17.0x |
| nuanced | 17.0x |
| paradigm | 15.1x |
| robust | 3.1x |
| essentially | 3.1x |

## Patterns NOT Found in Opus 4.5

These classic LLM-isms appear to have been reduced in newer models:
- delve
- tapestry
- vibrant
- myriad

## Source

Analysis conducted using log-odds ratios comparing:
- 200 Claude Opus 4.5 samples (47,012 words)
- 6,000 human texts from Wikipedia, OpenWebText, C4 (1,208,418 words)
