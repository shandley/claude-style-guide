"""
Statistical analysis comparing Opus 4.5 output to human writing.

Identifies distinctive patterns (LLM-isms) that appear more frequently
in Opus output than in human text.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import nltk
from tqdm import tqdm

# Download required NLTK data
def ensure_nltk_data():
    """Download NLTK data if not present."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt_tab', quiet=True)


class Marker(NamedTuple):
    """A distinctive pattern marker."""
    type: str  # word, bigram, trigram, phrase, structural
    item: str
    opus_rate: float
    human_rate: float
    log_odds: float
    ci_lower: float
    ci_upper: float
    opus_count: int
    human_count: int
    example_context: str


def load_corpus(path: Path, text_field: str = "response") -> list[str]:
    """Load texts from JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get(text_field, data.get("text", ""))
            if text:
                texts.append(text)
    return texts


def tokenize_texts(texts: list[str], verbose: bool = False) -> tuple[list[list[str]], list[list[str]]]:
    """Tokenize texts using NLTK, returning words and sentences."""
    ensure_nltk_data()

    all_words = []
    all_sentences = []

    iterator = tqdm(texts, desc="Tokenizing", disable=not verbose)
    for text in iterator:
        # Sentence tokenization
        sentences = nltk.sent_tokenize(text)
        all_sentences.append(sentences)

        # Word tokenization - extract alphabetic words, lowercase
        words = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            words.extend([t.lower() for t in tokens if t.isalpha()])
        all_words.append(words)

    return all_words, all_sentences


def get_ngrams(words: list[str], n: int) -> list[str]:
    """Extract n-grams from word list."""
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def calculate_log_odds_ratio(
    opus_count: int,
    human_count: int,
    opus_total: int,
    human_total: int,
    smoothing: float = 0.5
) -> tuple[float, float, float]:
    """
    Calculate log-odds ratio with confidence interval.

    Uses Agresti-Coull method for CI calculation.
    Returns (log_odds, ci_lower, ci_upper)
    """
    # Add smoothing to avoid division by zero
    opus_rate = (opus_count + smoothing) / (opus_total + 2 * smoothing)
    human_rate = (human_count + smoothing) / (human_total + 2 * smoothing)

    # Log odds ratio
    log_odds = math.log(opus_rate / human_rate)

    # Standard error using delta method
    se = math.sqrt(
        1 / (opus_count + smoothing) +
        1 / (opus_total - opus_count + smoothing) +
        1 / (human_count + smoothing) +
        1 / (human_total - human_count + smoothing)
    )

    # 95% CI
    z = 1.96
    ci_lower = log_odds - z * se
    ci_upper = log_odds + z * se

    return log_odds, ci_lower, ci_upper


def analyze_lexical_patterns(
    opus_words: list[list[str]],
    human_words: list[list[str]],
    opus_texts: list[str],
    verbose: bool = False
) -> list[Marker]:
    """Analyze word and n-gram frequencies."""
    markers = []

    # Flatten word lists
    opus_flat = [w for words in opus_words for w in words]
    human_flat = [w for words in human_words for w in words]

    opus_total = len(opus_flat)
    human_total = len(human_flat)

    if verbose:
        print(f"  Opus tokens: {opus_total:,}")
        print(f"  Human tokens: {human_total:,}")

    # Count unigrams
    opus_unigrams = Counter(opus_flat)
    human_unigrams = Counter(human_flat)

    # Count bigrams
    opus_bigrams = Counter()
    human_bigrams = Counter()
    for words in opus_words:
        opus_bigrams.update(get_ngrams(words, 2))
    for words in human_words:
        human_bigrams.update(get_ngrams(words, 2))

    # Count trigrams
    opus_trigrams = Counter()
    human_trigrams = Counter()
    for words in opus_words:
        opus_trigrams.update(get_ngrams(words, 3))
    for words in human_words:
        human_trigrams.update(get_ngrams(words, 3))

    # Find example context for an item
    def find_context(item: str, texts: list[str]) -> str:
        pattern = re.compile(r".{0,40}" + re.escape(item) + r".{0,40}", re.IGNORECASE)
        for text in texts[:100]:  # Search first 100 texts
            match = pattern.search(text)
            if match:
                return "..." + match.group(0).strip() + "..."
        return ""

    # Analyze unigrams
    all_words = set(opus_unigrams.keys()) | set(human_unigrams.keys())
    for word in tqdm(all_words, desc="Analyzing unigrams", disable=not verbose):
        opus_count = opus_unigrams.get(word, 0)
        human_count = human_unigrams.get(word, 0)

        # Skip rare words
        if opus_count < 5:
            continue

        opus_rate = opus_count / opus_total
        human_rate = (human_count + 0.5) / (human_total + 1)

        # Only flag if opus rate > 2x human rate
        if opus_rate < 2 * human_rate:
            continue

        log_odds, ci_lower, ci_upper = calculate_log_odds_ratio(
            opus_count, human_count, opus_total, human_total
        )

        # Only include if CI doesn't cross 0 (statistically significant)
        if ci_lower <= 0:
            continue

        markers.append(Marker(
            type="word",
            item=word,
            opus_rate=opus_rate,
            human_rate=human_rate,
            log_odds=log_odds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            opus_count=opus_count,
            human_count=human_count,
            example_context=find_context(word, opus_texts),
        ))

    # Analyze bigrams
    all_bigrams = set(opus_bigrams.keys()) | set(human_bigrams.keys())
    opus_bigram_total = sum(opus_bigrams.values())
    human_bigram_total = sum(human_bigrams.values())

    for bigram in tqdm(all_bigrams, desc="Analyzing bigrams", disable=not verbose):
        opus_count = opus_bigrams.get(bigram, 0)
        human_count = human_bigrams.get(bigram, 0)

        if opus_count < 3:
            continue

        opus_rate = opus_count / opus_bigram_total
        human_rate = (human_count + 0.5) / (human_bigram_total + 1)

        if opus_rate < 2 * human_rate:
            continue

        log_odds, ci_lower, ci_upper = calculate_log_odds_ratio(
            opus_count, human_count, opus_bigram_total, human_bigram_total
        )

        if ci_lower <= 0:
            continue

        markers.append(Marker(
            type="bigram",
            item=bigram,
            opus_rate=opus_rate,
            human_rate=human_rate,
            log_odds=log_odds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            opus_count=opus_count,
            human_count=human_count,
            example_context=find_context(bigram, opus_texts),
        ))

    # Analyze trigrams
    all_trigrams = set(opus_trigrams.keys()) | set(human_trigrams.keys())
    opus_trigram_total = sum(opus_trigrams.values())
    human_trigram_total = sum(human_trigrams.values())

    for trigram in tqdm(all_trigrams, desc="Analyzing trigrams", disable=not verbose):
        opus_count = opus_trigrams.get(trigram, 0)
        human_count = human_trigrams.get(trigram, 0)

        if opus_count < 3:
            continue

        opus_rate = opus_count / opus_trigram_total
        human_rate = (human_count + 0.5) / (human_trigram_total + 1)

        if opus_rate < 2 * human_rate:
            continue

        log_odds, ci_lower, ci_upper = calculate_log_odds_ratio(
            opus_count, human_count, opus_trigram_total, human_trigram_total
        )

        if ci_lower <= 0:
            continue

        markers.append(Marker(
            type="trigram",
            item=trigram,
            opus_rate=opus_rate,
            human_rate=human_rate,
            log_odds=log_odds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            opus_count=opus_count,
            human_count=human_count,
            example_context=find_context(trigram, opus_texts),
        ))

    return markers


def analyze_structural_patterns(
    opus_sentences: list[list[str]],
    human_sentences: list[list[str]],
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> tuple[list[Marker], dict]:
    """Analyze structural patterns like sentence starters, lengths, punctuation."""
    markers = []
    summary_stats = {}

    # Flatten sentences
    opus_flat = [s for sents in opus_sentences for s in sents]
    human_flat = [s for sents in human_sentences for s in sents]

    # Sentence length analysis
    opus_lengths = [len(s.split()) for s in opus_flat]
    human_lengths = [len(s.split()) for s in human_flat]

    summary_stats["opus_avg_sentence_length"] = sum(opus_lengths) / len(opus_lengths) if opus_lengths else 0
    summary_stats["human_avg_sentence_length"] = sum(human_lengths) / len(human_lengths) if human_lengths else 0
    summary_stats["opus_median_sentence_length"] = sorted(opus_lengths)[len(opus_lengths)//2] if opus_lengths else 0
    summary_stats["human_median_sentence_length"] = sorted(human_lengths)[len(human_lengths)//2] if human_lengths else 0

    # Sentence starters
    def get_sentence_starter(sentence: str) -> str:
        words = sentence.split()
        if words:
            # Get first word, lowercased
            return words[0].lower().strip(".,!?;:")
        return ""

    opus_starters = Counter(get_sentence_starter(s) for s in opus_flat if s)
    human_starters = Counter(get_sentence_starter(s) for s in human_flat if s)

    opus_starter_total = sum(opus_starters.values())
    human_starter_total = sum(human_starters.values())

    all_starters = set(opus_starters.keys()) | set(human_starters.keys())
    for starter in all_starters:
        if not starter or len(starter) < 2:
            continue

        opus_count = opus_starters.get(starter, 0)
        human_count = human_starters.get(starter, 0)

        if opus_count < 3:
            continue

        opus_rate = opus_count / opus_starter_total
        human_rate = (human_count + 0.5) / (human_starter_total + 1)

        if opus_rate < 1.5 * human_rate:  # Lower threshold for starters
            continue

        log_odds, ci_lower, ci_upper = calculate_log_odds_ratio(
            opus_count, human_count, opus_starter_total, human_starter_total
        )

        if ci_lower <= 0:
            continue

        # Find example sentence
        example = ""
        for s in opus_flat[:200]:
            if s.lower().startswith(starter):
                example = s[:100] + "..." if len(s) > 100 else s
                break

        markers.append(Marker(
            type="sentence_starter",
            item=starter,
            opus_rate=opus_rate,
            human_rate=human_rate,
            log_odds=log_odds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            opus_count=opus_count,
            human_count=human_count,
            example_context=example,
        ))

    # Punctuation analysis
    def count_punctuation(texts: list[str]) -> dict:
        counts = {
            "em_dash": 0,
            "semicolon": 0,
            "colon": 0,
            "exclamation": 0,
            "question": 0,
            "parentheses": 0,
            "quotes": 0,
        }
        total_chars = 0
        for text in texts:
            total_chars += len(text)
            counts["em_dash"] += text.count("—") + text.count("--")
            counts["semicolon"] += text.count(";")
            counts["colon"] += text.count(":")
            counts["exclamation"] += text.count("!")
            counts["question"] += text.count("?")
            counts["parentheses"] += text.count("(")
            counts["quotes"] += text.count('"') + text.count('"') + text.count('"')
        return counts, total_chars

    opus_punct, opus_chars = count_punctuation(opus_texts)
    human_punct, human_chars = count_punctuation(human_texts)

    for punct_type in opus_punct:
        opus_rate = opus_punct[punct_type] / opus_chars * 1000  # per 1000 chars
        human_rate = human_punct[punct_type] / human_chars * 1000

        summary_stats[f"opus_{punct_type}_per_1k"] = opus_rate
        summary_stats[f"human_{punct_type}_per_1k"] = human_rate

    # List/bullet usage
    def count_lists(texts: list[str]) -> int:
        count = 0
        for text in texts:
            # Count bullet patterns
            count += len(re.findall(r"^[\s]*[-•*]\s", text, re.MULTILINE))
            count += len(re.findall(r"^[\s]*\d+\.\s", text, re.MULTILINE))
        return count

    opus_lists = count_lists(opus_texts)
    human_lists = count_lists(human_texts)

    summary_stats["opus_list_items_per_text"] = opus_lists / len(opus_texts) if opus_texts else 0
    summary_stats["human_list_items_per_text"] = human_lists / len(human_texts) if human_texts else 0

    # Paragraph analysis
    def get_para_lengths(texts: list[str]) -> list[int]:
        lengths = []
        for text in texts:
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            lengths.extend(len(p.split()) for p in paras)
        return lengths

    opus_para_lengths = get_para_lengths(opus_texts)
    human_para_lengths = get_para_lengths(human_texts)

    summary_stats["opus_avg_para_length"] = sum(opus_para_lengths) / len(opus_para_lengths) if opus_para_lengths else 0
    summary_stats["human_avg_para_length"] = sum(human_para_lengths) / len(human_para_lengths) if human_para_lengths else 0

    return markers, summary_stats


def analyze_phrase_patterns(
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> list[Marker]:
    """Analyze common phrase patterns (hedging, transitions, fillers)."""
    markers = []

    # Known LLM-ism phrases to check
    phrase_patterns = {
        # Hedging phrases
        "it's important to note": "hedging",
        "it's worth noting": "hedging",
        "it's worth mentioning": "hedging",
        "it should be noted": "hedging",
        "generally speaking": "hedging",
        "in general": "hedging",
        "for the most part": "hedging",
        "in many cases": "hedging",
        "it depends on": "hedging",
        "that said": "hedging",
        "having said that": "hedging",
        "that being said": "hedging",
        "with that in mind": "hedging",

        # Transitions
        "additionally": "transition",
        "furthermore": "transition",
        "moreover": "transition",
        "in addition": "transition",
        "on the other hand": "transition",
        "conversely": "transition",
        "nevertheless": "transition",
        "nonetheless": "transition",
        "in contrast": "transition",
        "as a result": "transition",
        "consequently": "transition",
        "therefore": "transition",
        "thus": "transition",
        "hence": "transition",
        "accordingly": "transition",

        # Fillers
        "in order to": "filler",
        "due to the fact that": "filler",
        "the fact that": "filler",
        "it is important that": "filler",
        "it is essential that": "filler",
        "it is crucial that": "filler",
        "it is necessary to": "filler",
        "in terms of": "filler",
        "when it comes to": "filler",
        "with respect to": "filler",
        "with regard to": "filler",
        "at the end of the day": "filler",
        "at this point in time": "filler",
        "for all intents and purposes": "filler",

        # Structure phrases
        "let me explain": "structure",
        "let's break this down": "structure",
        "let's dive into": "structure",
        "let's explore": "structure",
        "here's the thing": "structure",
        "here's what": "structure",
        "the key thing": "structure",
        "the main thing": "structure",
        "first and foremost": "structure",
        "last but not least": "structure",

        # Summary/conclusion
        "in summary": "conclusion",
        "to summarize": "conclusion",
        "in conclusion": "conclusion",
        "to conclude": "conclusion",
        "overall": "conclusion",
        "all in all": "conclusion",
        "to sum up": "conclusion",
        "in essence": "conclusion",
        "essentially": "conclusion",
        "ultimately": "conclusion",
        "at its core": "conclusion",
        "fundamentally": "conclusion",

        # Emphasis
        "absolutely": "emphasis",
        "definitely": "emphasis",
        "certainly": "emphasis",
        "clearly": "emphasis",
        "obviously": "emphasis",
        "of course": "emphasis",
        "naturally": "emphasis",
        "undoubtedly": "emphasis",
        "without a doubt": "emphasis",
        "indeed": "emphasis",

        # Known LLM favorites
        "delve": "llm_favorite",
        "crucial": "llm_favorite",
        "vital": "llm_favorite",
        "pivotal": "llm_favorite",
        "robust": "llm_favorite",
        "comprehensive": "llm_favorite",
        "nuanced": "llm_favorite",
        "multifaceted": "llm_favorite",
        "intricate": "llm_favorite",
        "meticulous": "llm_favorite",
        "seamlessly": "llm_favorite",
        "leverage": "llm_favorite",
        "utilize": "llm_favorite",
        "facilitate": "llm_favorite",
        "foster": "llm_favorite",
        "realm": "llm_favorite",
        "landscape": "llm_favorite",
        "paradigm": "llm_favorite",
        "myriad": "llm_favorite",
        "plethora": "llm_favorite",
        "tapestry": "llm_favorite",
        "embark": "llm_favorite",
        "endeavor": "llm_favorite",
        "aforementioned": "llm_favorite",
    }

    opus_total = sum(len(t) for t in opus_texts)
    human_total = sum(len(t) for t in human_texts)

    for phrase, category in tqdm(phrase_patterns.items(), desc="Analyzing phrases", disable=not verbose):
        # Count occurrences (case-insensitive)
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)

        opus_count = sum(len(pattern.findall(t)) for t in opus_texts)
        human_count = sum(len(pattern.findall(t)) for t in human_texts)

        if opus_count < 2:
            continue

        # Rate per 10k characters
        opus_rate = opus_count / opus_total * 10000
        human_rate = (human_count + 0.5) / (human_total + 1) * 10000

        if opus_rate < 1.5 * human_rate:
            continue

        log_odds, ci_lower, ci_upper = calculate_log_odds_ratio(
            opus_count, human_count, opus_total // 100, human_total // 100  # Normalize
        )

        if ci_lower <= 0:
            continue

        # Find example
        example = ""
        for t in opus_texts[:100]:
            match = pattern.search(t)
            if match:
                start = max(0, match.start() - 30)
                end = min(len(t), match.end() + 30)
                example = "..." + t[start:end] + "..."
                break

        markers.append(Marker(
            type=f"phrase_{category}",
            item=phrase,
            opus_rate=opus_rate,
            human_rate=human_rate,
            log_odds=log_odds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            opus_count=opus_count,
            human_count=human_count,
            example_context=example,
        ))

    return markers


def run_analysis(
    opus_path: Path,
    human_path: Path,
    output_path: Path,
    verbose: bool = False
) -> dict:
    """Run full analysis and save results."""

    # Load corpora
    if verbose:
        print("Loading corpora...")
    opus_texts = load_corpus(opus_path, text_field="response")
    human_texts = load_corpus(human_path, text_field="text")

    if verbose:
        print(f"  Opus samples: {len(opus_texts)}")
        print(f"  Human samples: {len(human_texts)}")

    # Tokenize
    if verbose:
        print("\nTokenizing...")
    opus_words, opus_sentences = tokenize_texts(opus_texts, verbose)
    human_words, human_sentences = tokenize_texts(human_texts, verbose)

    # Lexical analysis
    if verbose:
        print("\nAnalyzing lexical patterns...")
    lexical_markers = analyze_lexical_patterns(opus_words, human_words, opus_texts, verbose)

    # Structural analysis
    if verbose:
        print("\nAnalyzing structural patterns...")
    structural_markers, summary_stats = analyze_structural_patterns(
        opus_sentences, human_sentences, opus_texts, human_texts, verbose
    )

    # Phrase analysis
    if verbose:
        print("\nAnalyzing phrase patterns...")
    phrase_markers = analyze_phrase_patterns(opus_texts, human_texts, verbose)

    # Combine and sort all markers by log-odds
    all_markers = lexical_markers + structural_markers + phrase_markers
    all_markers.sort(key=lambda m: -m.log_odds)

    # Convert to serializable format
    results = {
        "corpus_stats": {
            "opus_samples": len(opus_texts),
            "human_samples": len(human_texts),
            "opus_total_words": sum(len(w) for w in opus_words),
            "human_total_words": sum(len(w) for w in human_words),
        },
        "summary_stats": summary_stats,
        "markers": [
            {
                "type": m.type,
                "item": m.item,
                "opus_rate": m.opus_rate,
                "human_rate": m.human_rate,
                "log_odds": m.log_odds,
                "ci_lower": m.ci_lower,
                "ci_upper": m.ci_upper,
                "opus_count": m.opus_count,
                "human_count": m.human_count,
                "example_context": m.example_context,
            }
            for m in all_markers
        ],
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nAnalysis complete!")
        print(f"  Total markers found: {len(all_markers)}")
        print(f"  Results saved to: {output_path}")

        # Show top 10
        print("\nTop 10 most distinctive markers:")
        for m in all_markers[:10]:
            ratio = m.opus_rate / m.human_rate if m.human_rate > 0 else float('inf')
            print(f"  {m.item} ({m.type}): {ratio:.1f}x more common in Opus")

    return results


def main(
    opus_path: Path,
    human_path: Path,
    output_path: Path,
    verbose: bool = False
) -> dict:
    """Main entry point."""
    print("Running style analysis...")
    print(f"  Opus samples: {opus_path}")
    print(f"  Human samples: {human_path}")
    print(f"  Output: {output_path}")
    print()

    return run_analysis(opus_path, human_path, output_path, verbose)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    opus_path = base_path / "data" / "opus_samples.jsonl"
    human_path = base_path / "data" / "human_samples.jsonl"
    output_path = base_path / "results" / "markers.json"
    main(opus_path, human_path, output_path, verbose=True)
