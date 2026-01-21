"""
Statistical analysis comparing Opus 4.5 output to human writing.

Identifies distinctive patterns (LLM-isms) that appear more frequently
in Opus output than in human text.
"""

import json
import math
import re
import statistics
from collections import Counter
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


def analyze_sentence_length_distribution(
    opus_sentences: list[list[str]],
    human_sentences: list[list[str]],
    verbose: bool = False
) -> dict:
    """
    Analyze sentence length distribution patterns.

    AI tends to produce more uniform sentence lengths, while human writing
    has more variation. This function computes metrics to detect this.
    """
    # Flatten sentences and get lengths
    opus_flat = [s for sents in opus_sentences for s in sents]
    human_flat = [s for sents in human_sentences for s in sents]

    opus_lengths = [len(s.split()) for s in opus_flat if s.strip()]
    human_lengths = [len(s.split()) for s in human_flat if s.strip()]

    def compute_distribution_stats(lengths: list[int]) -> dict:
        if not lengths:
            return {}

        mean = statistics.mean(lengths)
        stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0

        # Coefficient of variation (lower = more uniform)
        cv = (stdev / mean * 100) if mean > 0 else 0

        # Distribution buckets
        short = sum(1 for length in lengths if length <= 10)
        medium = sum(1 for length in lengths if 10 < length <= 25)
        long = sum(1 for length in lengths if length > 25)
        total = len(lengths)

        # Percentiles
        sorted_lengths = sorted(lengths)
        p10 = sorted_lengths[int(len(sorted_lengths) * 0.1)]
        p25 = sorted_lengths[int(len(sorted_lengths) * 0.25)]
        p50 = sorted_lengths[int(len(sorted_lengths) * 0.5)]
        p75 = sorted_lengths[int(len(sorted_lengths) * 0.75)]
        p90 = sorted_lengths[int(len(sorted_lengths) * 0.9)]

        return {
            "mean": round(mean, 1),
            "stdev": round(stdev, 1),
            "coefficient_of_variation": round(cv, 1),
            "min": min(lengths),
            "max": max(lengths),
            "p10": p10,
            "p25": p25,
            "p50_median": p50,
            "p75": p75,
            "p90": p90,
            "pct_short_1_10": round(short / total * 100, 1),
            "pct_medium_11_25": round(medium / total * 100, 1),
            "pct_long_26_plus": round(long / total * 100, 1),
        }

    opus_stats = compute_distribution_stats(opus_lengths)
    human_stats = compute_distribution_stats(human_lengths)

    if verbose:
        print("\n  Sentence length distribution:")
        print(f"    Opus CV: {opus_stats.get('coefficient_of_variation', 0)}% (lower = more uniform)")
        print(f"    Human CV: {human_stats.get('coefficient_of_variation', 0)}%")

    return {
        "opus_sentence_distribution": opus_stats,
        "human_sentence_distribution": human_stats,
    }


def detect_passive_voice(
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> dict:
    """
    Detect passive voice constructions.

    AI writing often uses more passive voice than human writing.
    Looks for patterns like "was/were/is/are/been + past participle"
    """
    # Common passive voice patterns
    # This is a heuristic - not perfect but catches most cases
    passive_patterns = [
        r'\b(is|are|was|were|be|been|being)\s+(\w+ed)\b',  # is completed, was started
        r'\b(is|are|was|were|be|been|being)\s+(\w+en)\b',  # is taken, was written
        r'\b(is|are|was|were|be|been|being)\s+(made|done|seen|known|shown|found|given|told|left|kept|felt|thought|brought|bought|caught|taught|sought)\b',
        r'\b(has|have|had)\s+been\s+(\w+ed)\b',  # has been completed
        r'\b(has|have|had)\s+been\s+(\w+en)\b',  # has been taken
        r'\b(will|would|should|could|might|must)\s+be\s+(\w+ed)\b',  # will be completed
        r'\b(will|would|should|could|might|must)\s+be\s+(\w+en)\b',  # will be taken
    ]

    def count_passive(texts: list[str]) -> tuple[int, int]:
        passive_count = 0
        total_sentences = 0
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            total_sentences += len(sentences)
            for sent in sentences:
                for pattern in passive_patterns:
                    if re.search(pattern, sent, re.IGNORECASE):
                        passive_count += 1
                        break  # Count each sentence once
        return passive_count, total_sentences

    opus_passive, opus_total = count_passive(opus_texts)
    human_passive, human_total = count_passive(human_texts)

    opus_pct = (opus_passive / opus_total * 100) if opus_total > 0 else 0
    human_pct = (human_passive / human_total * 100) if human_total > 0 else 0

    if verbose:
        print("\n  Passive voice usage:")
        print(f"    Opus: {opus_pct:.1f}% of sentences")
        print(f"    Human: {human_pct:.1f}% of sentences")

    return {
        "opus_passive_voice_pct": round(opus_pct, 1),
        "human_passive_voice_pct": round(human_pct, 1),
        "opus_passive_count": opus_passive,
        "opus_total_sentences": opus_total,
        "human_passive_count": human_passive,
        "human_total_sentences": human_total,
    }


def analyze_paragraph_patterns(
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> dict:
    """
    Analyze paragraph structure patterns.

    Looks at paragraph length variation, opening patterns, and structure.
    """
    def get_paragraph_stats(texts: list[str]) -> dict:
        all_para_lengths = []
        paras_per_doc = []
        opening_words = Counter()

        for text in texts:
            # Split into paragraphs
            paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            paras_per_doc.append(len(paras))

            for para in paras:
                words = para.split()
                if words:
                    all_para_lengths.append(len(words))
                    # Track opening word of paragraph
                    opening_words[words[0].lower().strip('.,!?;:"')] += 1

        if not all_para_lengths:
            return {}

        mean_len = statistics.mean(all_para_lengths)
        stdev_len = statistics.stdev(all_para_lengths) if len(all_para_lengths) > 1 else 0
        cv = (stdev_len / mean_len * 100) if mean_len > 0 else 0

        # Top opening words
        top_openers = opening_words.most_common(10)

        return {
            "avg_paragraphs_per_doc": round(statistics.mean(paras_per_doc), 1) if paras_per_doc else 0,
            "avg_para_length_words": round(mean_len, 1),
            "para_length_stdev": round(stdev_len, 1),
            "para_length_cv": round(cv, 1),
            "min_para_length": min(all_para_lengths),
            "max_para_length": max(all_para_lengths),
            "top_para_openers": top_openers,
        }

    opus_stats = get_paragraph_stats(opus_texts)
    human_stats = get_paragraph_stats(human_texts)

    if verbose:
        print("\n  Paragraph patterns:")
        print(f"    Opus avg para length: {opus_stats.get('avg_para_length_words', 0)} words (CV: {opus_stats.get('para_length_cv', 0)}%)")
        print(f"    Human avg para length: {human_stats.get('avg_para_length_words', 0)} words (CV: {human_stats.get('para_length_cv', 0)}%)")

    return {
        "opus_paragraph_stats": opus_stats,
        "human_paragraph_stats": human_stats,
    }


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


def analyze_sentence_starters(
    opus_sentences: list[list[str]],
    human_sentences: list[list[str]],
    verbose: bool = False
) -> dict:
    """Analyze what words/patterns start sentences."""

    def get_starters(sentences_list: list[list[str]], n_words: int = 2) -> Counter:
        """Extract first n words from each sentence."""
        starters = Counter()
        for doc_sentences in sentences_list:
            for sent in doc_sentences:
                words = sent.split()[:n_words]
                if words:
                    starter = " ".join(words).lower()
                    # Clean punctuation from start
                    starter = re.sub(r'^[^a-z]+', '', starter)
                    if starter and len(starter) > 2:
                        starters[starter] += 1
        return starters

    # Get single-word and two-word starters
    opus_starters_1 = get_starters(opus_sentences, 1)
    human_starters_1 = get_starters(human_sentences, 1)
    opus_starters_2 = get_starters(opus_sentences, 2)
    human_starters_2 = get_starters(human_sentences, 2)

    opus_total_1 = sum(opus_starters_1.values())
    human_total_1 = sum(human_starters_1.values())
    opus_total_2 = sum(opus_starters_2.values())
    human_total_2 = sum(human_starters_2.values())

    # Find distinctive single-word starters
    distinctive_starters_1 = []
    for starter, opus_count in opus_starters_1.most_common(100):
        human_count = human_starters_1.get(starter, 0)
        if opus_count < 5:
            continue
        opus_rate = opus_count / opus_total_1
        human_rate = (human_count + 0.5) / (human_total_1 + 1)
        ratio = opus_rate / human_rate
        if ratio > 1.5:
            distinctive_starters_1.append({
                "starter": starter,
                "opus_pct": round(opus_rate * 100, 2),
                "human_pct": round(human_rate * 100, 2),
                "ratio": round(ratio, 1)
            })

    # Find distinctive two-word starters
    distinctive_starters_2 = []
    for starter, opus_count in opus_starters_2.most_common(200):
        human_count = human_starters_2.get(starter, 0)
        if opus_count < 3:
            continue
        opus_rate = opus_count / opus_total_2
        human_rate = (human_count + 0.5) / (human_total_2 + 1)
        ratio = opus_rate / human_rate
        if ratio > 2.0:
            distinctive_starters_2.append({
                "starter": starter,
                "opus_pct": round(opus_rate * 100, 2),
                "human_pct": round(human_rate * 100, 2),
                "ratio": round(ratio, 1)
            })

    # Sort by ratio
    distinctive_starters_1.sort(key=lambda x: -x["ratio"])
    distinctive_starters_2.sort(key=lambda x: -x["ratio"])

    if verbose:
        print(f"  Found {len(distinctive_starters_1)} distinctive single-word starters")
        print(f"  Found {len(distinctive_starters_2)} distinctive two-word starters")
        if distinctive_starters_1:
            print(f"  Top single-word: {distinctive_starters_1[0]}")
        if distinctive_starters_2:
            print(f"  Top two-word: {distinctive_starters_2[0]}")

    return {
        "sentence_starters_1word": distinctive_starters_1[:20],
        "sentence_starters_2word": distinctive_starters_2[:20],
    }


def analyze_transition_words(
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> dict:
    """Analyze frequency of formal transition words."""

    # Formal transitions (AI tends to overuse)
    formal_transitions = [
        "however", "furthermore", "moreover", "additionally", "consequently",
        "nevertheless", "therefore", "thus", "hence", "accordingly",
        "subsequently", "conversely", "alternatively", "specifically",
        "notably", "importantly", "significantly", "ultimately"
    ]

    # Casual transitions (human-like)
    casual_transitions = [
        "but", "and", "so", "also", "still", "yet", "then", "now",
        "plus", "anyway", "besides", "though"
    ]

    def count_transitions(texts: list[str], transitions: list[str]) -> dict:
        """Count transition words at sentence boundaries."""
        counts = Counter()
        total_sentences = 0
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            total_sentences += len(sentences)
            for sent in sentences:
                first_word = sent.split()[0].lower().rstrip(",") if sent.split() else ""
                if first_word in transitions:
                    counts[first_word] += 1
        return counts, total_sentences

    opus_formal, opus_sents = count_transitions(opus_texts, formal_transitions)
    human_formal, human_sents = count_transitions(human_texts, formal_transitions)
    opus_casual, _ = count_transitions(opus_texts, casual_transitions)
    human_casual, _ = count_transitions(human_texts, casual_transitions)

    # Calculate rates per 100 sentences
    opus_formal_total = sum(opus_formal.values())
    human_formal_total = sum(human_formal.values())
    opus_casual_total = sum(opus_casual.values())
    human_casual_total = sum(human_casual.values())

    opus_formal_rate = (opus_formal_total / opus_sents * 100) if opus_sents > 0 else 0
    human_formal_rate = (human_formal_total / human_sents * 100) if human_sents > 0 else 0
    opus_casual_rate = (opus_casual_total / opus_sents * 100) if opus_sents > 0 else 0
    human_casual_rate = (human_casual_total / human_sents * 100) if human_sents > 0 else 0

    # Per-word breakdown
    formal_breakdown = []
    for word in formal_transitions:
        opus_count = opus_formal.get(word, 0)
        human_count = human_formal.get(word, 0)
        opus_rate = (opus_count / opus_sents * 100) if opus_sents > 0 else 0
        human_rate = (human_count / human_sents * 100) if human_sents > 0 else 0
        if opus_count > 0 or human_count > 0:
            ratio = opus_rate / human_rate if human_rate > 0 else float('inf')
            formal_breakdown.append({
                "word": word,
                "opus_per_100_sents": round(opus_rate, 2),
                "human_per_100_sents": round(human_rate, 2),
                "ratio": round(ratio, 1) if ratio != float('inf') else "high"
            })

    formal_breakdown.sort(key=lambda x: -(x["ratio"] if isinstance(x["ratio"], (int, float)) else 999))

    if verbose:
        print(f"  Opus formal transitions: {opus_formal_rate:.1f} per 100 sentences")
        print(f"  Human formal transitions: {human_formal_rate:.1f} per 100 sentences")
        print(f"  Opus casual transitions: {opus_casual_rate:.1f} per 100 sentences")
        print(f"  Human casual transitions: {human_casual_rate:.1f} per 100 sentences")

    return {
        "transition_formal_opus_rate": round(opus_formal_rate, 2),
        "transition_formal_human_rate": round(human_formal_rate, 2),
        "transition_casual_opus_rate": round(opus_casual_rate, 2),
        "transition_casual_human_rate": round(human_casual_rate, 2),
        "transition_formal_ratio": round(opus_formal_rate / human_formal_rate, 1) if human_formal_rate > 0 else "high",
        "transition_breakdown": formal_breakdown[:15],
    }


def analyze_hedging_language(
    opus_texts: list[str],
    human_texts: list[str],
    verbose: bool = False
) -> dict:
    """Analyze frequency of hedging/qualifying language."""

    # Hedging words and phrases
    hedging_words = [
        "might", "could", "may", "perhaps", "possibly", "potentially",
        "generally", "typically", "usually", "often", "sometimes",
        "likely", "unlikely", "probably", "arguably", "seemingly",
        "somewhat", "relatively", "fairly", "rather", "quite"
    ]

    # Hedging phrases
    hedging_phrases = [
        "it seems", "it appears", "it is possible", "it is likely",
        "to some extent", "in some cases", "in many cases",
        "tend to", "tends to", "can be", "may be", "might be",
        "it could be", "it might be", "it may be"
    ]

    def count_hedging(texts: list[str]) -> tuple[int, int, int]:
        """Count hedging words, phrases, and total words."""
        word_count = 0
        phrase_count = 0
        total_words = 0

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            total_words += len(words)

            # Count hedging words
            for word in words:
                clean_word = re.sub(r'[^a-z]', '', word)
                if clean_word in hedging_words:
                    word_count += 1

            # Count hedging phrases
            for phrase in hedging_phrases:
                phrase_count += text_lower.count(phrase)

        return word_count, phrase_count, total_words

    opus_words, opus_phrases, opus_total = count_hedging(opus_texts)
    human_words, human_phrases, human_total = count_hedging(human_texts)

    # Calculate rates per 1000 words
    opus_word_rate = (opus_words / opus_total * 1000) if opus_total > 0 else 0
    human_word_rate = (human_words / human_total * 1000) if human_total > 0 else 0
    opus_phrase_rate = (opus_phrases / opus_total * 1000) if opus_total > 0 else 0
    human_phrase_rate = (human_phrases / human_total * 1000) if human_total > 0 else 0

    opus_total_rate = opus_word_rate + opus_phrase_rate
    human_total_rate = human_word_rate + human_phrase_rate

    # Per-word breakdown
    word_breakdown = []
    for hedge_word in hedging_words:
        opus_count = sum(1 for text in opus_texts
                       for word in text.lower().split()
                       if re.sub(r'[^a-z]', '', word) == hedge_word)
        human_count = sum(1 for text in human_texts
                        for word in text.lower().split()
                        if re.sub(r'[^a-z]', '', word) == hedge_word)

        opus_rate = (opus_count / opus_total * 1000) if opus_total > 0 else 0
        human_rate = (human_count / human_total * 1000) if human_total > 0 else 0

        if opus_count >= 3:
            ratio = opus_rate / human_rate if human_rate > 0 else float('inf')
            word_breakdown.append({
                "word": hedge_word,
                "opus_per_1k": round(opus_rate, 2),
                "human_per_1k": round(human_rate, 2),
                "ratio": round(ratio, 1) if ratio != float('inf') else "high"
            })

    word_breakdown.sort(key=lambda x: -(x["ratio"] if isinstance(x["ratio"], (int, float)) else 999))

    if verbose:
        print(f"  Opus hedging rate: {opus_total_rate:.1f} per 1k words")
        print(f"  Human hedging rate: {human_total_rate:.1f} per 1k words")
        ratio = opus_total_rate / human_total_rate if human_total_rate > 0 else float('inf')
        print(f"  Ratio: {ratio:.1f}x")

    return {
        "hedging_opus_rate": round(opus_total_rate, 2),
        "hedging_human_rate": round(human_total_rate, 2),
        "hedging_ratio": round(opus_total_rate / human_total_rate, 1) if human_total_rate > 0 else "high",
        "hedging_word_rate_opus": round(opus_word_rate, 2),
        "hedging_word_rate_human": round(human_word_rate, 2),
        "hedging_breakdown": word_breakdown[:15],
    }


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

    # Enhanced structural analysis
    if verbose:
        print("\nAnalyzing sentence length distribution...")
    sentence_dist_stats = analyze_sentence_length_distribution(
        opus_sentences, human_sentences, verbose
    )
    summary_stats.update(sentence_dist_stats)

    if verbose:
        print("\nDetecting passive voice...")
    passive_stats = detect_passive_voice(opus_texts, human_texts, verbose)
    summary_stats.update(passive_stats)

    if verbose:
        print("\nAnalyzing paragraph patterns...")
    paragraph_stats = analyze_paragraph_patterns(opus_texts, human_texts, verbose)
    summary_stats.update(paragraph_stats)

    # Sentence starter analysis
    if verbose:
        print("\nAnalyzing sentence starters...")
    starter_stats = analyze_sentence_starters(opus_sentences, human_sentences, verbose)
    summary_stats.update(starter_stats)

    # Transition word analysis
    if verbose:
        print("\nAnalyzing transition words...")
    transition_stats = analyze_transition_words(opus_texts, human_texts, verbose)
    summary_stats.update(transition_stats)

    # Hedging language analysis
    if verbose:
        print("\nAnalyzing hedging language...")
    hedging_stats = analyze_hedging_language(opus_texts, human_texts, verbose)
    summary_stats.update(hedging_stats)

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
        print("\nAnalysis complete!")
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
