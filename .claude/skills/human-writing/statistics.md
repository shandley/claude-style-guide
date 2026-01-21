# Statistical Reference

Data from analysis of 200 Claude Opus 4.5 samples vs 6,000 human texts.

## Punctuation Overuse

| Punctuation | AI Rate (per 1k chars) | Human Rate | Ratio |
|-------------|------------------------|------------|-------|
| Em dash (—) | 4.79 | 0.28 | 16.9x |
| Colon | 4.12 | 1.01 | 4.1x |
| Semicolon | 0.69 | 0.23 | 3.1x |

## Most Distinctive Words/Phrases

| Pattern | AI vs Human Ratio |
|---------|-------------------|
| comprehensive | 24.5x |
| in essence | very high (near zero in human) |
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
