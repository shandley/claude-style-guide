"""
Compare writing patterns across different Claude model versions.

Analyzes how LLM-isms have evolved over time by comparing
samples from multiple models against the same human baseline.
"""

import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime

import nltk

from generate_samples import AVAILABLE_MODELS


def ensure_nltk_data():
    """Download NLTK data if not present."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


def load_samples(path: Path) -> list[str]:
    """Load text samples from JSONL file."""
    texts = []
    if not path.exists():
        return texts
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("response", data.get("text", ""))
            if text:
                texts.append(text)
    return texts


def count_patterns(texts: list[str]) -> dict:
    """Count various patterns in texts."""
    ensure_nltk_data()

    # Initialize counters
    word_counts = Counter()
    bigram_counts = Counter()
    phrase_counts = Counter()
    total_words = 0
    total_chars = sum(len(t) for t in texts)

    # Phrases to check
    phrases = [
        "it's important to note", "it's worth noting", "that being said",
        "in essence", "fundamentally", "essentially", "comprehensive",
        "nuanced", "paradigm", "robust", "leverage", "utilize",
        "delve", "tapestry", "vibrant", "myriad", "plethora",
        "furthermore", "moreover", "additionally", "certainly"
    ]

    # Count phrases
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        count = sum(len(pattern.findall(t)) for t in texts)
        if count > 0:
            phrase_counts[phrase] = count

    # Tokenize and count words/bigrams
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        words = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            words.extend([t.lower() for t in tokens if t.isalpha()])

        word_counts.update(words)
        total_words += len(words)

        # Bigrams
        for i in range(len(words) - 1):
            bigram_counts[f"{words[i]} {words[i+1]}"] += 1

    # Count punctuation
    em_dash_count = sum(t.count("—") + t.count("--") for t in texts)
    colon_count = sum(t.count(":") for t in texts)
    semicolon_count = sum(t.count(";") for t in texts)

    return {
        "word_counts": word_counts,
        "bigram_counts": bigram_counts,
        "phrase_counts": phrase_counts,
        "total_words": total_words,
        "total_chars": total_chars,
        "num_samples": len(texts),
        "em_dash_per_1k": em_dash_count / total_chars * 1000 if total_chars > 0 else 0,
        "colon_per_1k": colon_count / total_chars * 1000 if total_chars > 0 else 0,
        "semicolon_per_1k": semicolon_count / total_chars * 1000 if total_chars > 0 else 0,
    }


def compare_models(
    data_dir: Path,
    human_path: Path,
    output_path: Path,
    verbose: bool = False
) -> dict:
    """
    Compare patterns across all available model samples.

    Args:
        data_dir: Directory containing model sample files
        human_path: Path to human samples
        output_path: Path to save comparison results

    Returns:
        Comparison results dict
    """
    results = {
        "generated": datetime.now().isoformat(),
        "models": {},
        "human_baseline": {},
        "comparisons": {},
    }

    # Load human baseline
    if verbose:
        print("Loading human baseline...")
    human_texts = load_samples(human_path)
    if human_texts:
        human_stats = count_patterns(human_texts)
        results["human_baseline"] = {
            "num_samples": human_stats["num_samples"],
            "total_words": human_stats["total_words"],
            "em_dash_per_1k": human_stats["em_dash_per_1k"],
            "colon_per_1k": human_stats["colon_per_1k"],
            "semicolon_per_1k": human_stats["semicolon_per_1k"],
        }
    else:
        print("Warning: No human samples found")
        human_stats = None

    # Load each model's samples
    model_stats = {}
    for model_name, model_id in AVAILABLE_MODELS.items():
        sample_path = data_dir / f"{model_name.replace('.', '_')}_samples.jsonl"

        # Also check for legacy naming
        if not sample_path.exists() and model_name == "opus-4.5":
            sample_path = data_dir / "opus_samples.jsonl"

        if sample_path.exists():
            if verbose:
                print(f"Loading {model_name} samples from {sample_path}...")
            texts = load_samples(sample_path)
            if texts:
                stats = count_patterns(texts)
                model_stats[model_name] = stats
                results["models"][model_name] = {
                    "num_samples": stats["num_samples"],
                    "total_words": stats["total_words"],
                    "em_dash_per_1k": stats["em_dash_per_1k"],
                    "colon_per_1k": stats["colon_per_1k"],
                    "semicolon_per_1k": stats["semicolon_per_1k"],
                }

    if not model_stats:
        print("No model samples found!")
        return results

    # Compare patterns across models
    if verbose:
        print("\nComparing patterns...")

    # Key phrases to compare
    key_phrases = [
        "in essence", "fundamentally", "essentially", "comprehensive",
        "nuanced", "paradigm", "robust", "delve", "tapestry", "myriad"
    ]

    phrase_comparison = {}
    for phrase in key_phrases:
        phrase_comparison[phrase] = {}
        human_rate = 0
        if human_stats and human_stats["total_chars"] > 0:
            human_count = human_stats["phrase_counts"].get(phrase, 0)
            human_rate = human_count / human_stats["total_chars"] * 10000
        phrase_comparison[phrase]["human"] = human_rate

        for model_name, stats in model_stats.items():
            if stats["total_chars"] > 0:
                count = stats["phrase_counts"].get(phrase, 0)
                rate = count / stats["total_chars"] * 10000
                phrase_comparison[phrase][model_name] = rate

    results["comparisons"]["phrases"] = phrase_comparison

    # Punctuation comparison
    punct_comparison = {
        "em_dash": {"human": results["human_baseline"].get("em_dash_per_1k", 0)},
        "colon": {"human": results["human_baseline"].get("colon_per_1k", 0)},
        "semicolon": {"human": results["human_baseline"].get("semicolon_per_1k", 0)},
    }
    for model_name in model_stats:
        punct_comparison["em_dash"][model_name] = results["models"][model_name]["em_dash_per_1k"]
        punct_comparison["colon"][model_name] = results["models"][model_name]["colon_per_1k"]
        punct_comparison["semicolon"][model_name] = results["models"][model_name]["semicolon_per_1k"]

    results["comparisons"]["punctuation"] = punct_comparison

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nComparison saved to {output_path}")

    return results


def generate_comparison_report(
    comparison_path: Path,
    output_path: Path,
    verbose: bool = False
) -> None:
    """Generate a markdown report comparing models."""

    with open(comparison_path) as f:
        data = json.load(f)

    lines = []
    lines.append("# Claude Model Evolution: Writing Pattern Comparison")
    lines.append("")
    lines.append(f"*Generated {data['generated'][:10]}*")
    lines.append("")

    # Models included
    lines.append("## Models Analyzed")
    lines.append("")
    models = list(data["models"].keys())
    # Sort by version (roughly)
    model_order = ["opus-3", "sonnet-3", "haiku-3", "sonnet-3.5", "sonnet-4", "opus-4.5"]
    models = sorted(models, key=lambda m: model_order.index(m) if m in model_order else 99)

    lines.append("| Model | Samples | Words |")
    lines.append("|-------|---------|-------|")
    for model in models:
        info = data["models"][model]
        lines.append(f"| {model} | {info['num_samples']} | {info['total_words']:,} |")
    lines.append("")

    # Punctuation evolution
    lines.append("## Punctuation Patterns")
    lines.append("")
    lines.append("How punctuation usage has changed across model versions:")
    lines.append("")

    punct = data["comparisons"]["punctuation"]
    lines.append("| Punctuation | Human | " + " | ".join(models) + " |")
    lines.append("|-------------|-------|" + "|".join(["-------"] * len(models)) + "|")

    for punct_type in ["em_dash", "colon", "semicolon"]:
        human_val = punct[punct_type].get("human", 0)
        row = f"| {punct_type.replace('_', ' ')} | {human_val:.2f} |"
        for model in models:
            val = punct[punct_type].get(model, 0)
            ratio = val / human_val if human_val > 0 else 0
            row += f" {val:.2f} ({ratio:.1f}x) |"
        lines.append(row)
    lines.append("")
    lines.append("*Values are occurrences per 1,000 characters. Ratio vs human in parentheses.*")
    lines.append("")

    # Phrase evolution
    lines.append("## LLM Phrase Patterns")
    lines.append("")
    lines.append("How distinctive phrases have evolved:")
    lines.append("")

    phrases = data["comparisons"]["phrases"]
    # Filter to phrases that appear in at least one model
    active_phrases = [p for p in phrases if any(phrases[p].get(m, 0) > 0 for m in models)]

    if active_phrases:
        lines.append("| Phrase | Human | " + " | ".join(models) + " |")
        lines.append("|--------|-------|" + "|".join(["-------"] * len(models)) + "|")

        for phrase in sorted(active_phrases):
            human_val = phrases[phrase].get("human", 0)
            row = f"| {phrase} | {human_val:.2f} |"
            for model in models:
                val = phrases[phrase].get(model, 0)
                if val > 0:
                    ratio = val / human_val if human_val > 0 else float('inf')
                    if ratio > 100:
                        row += f" {val:.1f} (high) |"
                    else:
                        row += f" {val:.2f} ({ratio:.1f}x) |"
                else:
                    row += " 0 |"
            lines.append(row)
        lines.append("")
        lines.append("*Values are occurrences per 10,000 characters.*")
    else:
        lines.append("*No tracked phrases found in samples.*")
    lines.append("")

    # Trends summary
    lines.append("## Evolution Trends")
    lines.append("")

    if len(models) >= 2:
        # Compare oldest to newest
        oldest = models[0]
        newest = models[-1]

        lines.append(f"### {oldest} vs {newest}")
        lines.append("")

        # Em dash trend
        old_em = punct["em_dash"].get(oldest, 0)
        new_em = punct["em_dash"].get(newest, 0)
        if old_em > 0:
            change = (new_em - old_em) / old_em * 100
            direction = "decreased" if change < 0 else "increased"
            lines.append(f"- Em dash usage {direction} by {abs(change):.0f}%")

        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("Each model was prompted with the same set of technical writing prompts.")
    lines.append("Patterns were counted and normalized by corpus size for comparison.")
    lines.append("Human baseline from Wikipedia, OpenWebText, and C4 datasets.")
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"Report saved to {output_path}")


def main(
    data_dir: Path,
    human_path: Path,
    output_dir: Path,
    verbose: bool = False
) -> None:
    """Main entry point."""
    print("Comparing Claude models...")

    comparison_path = output_dir / "model_comparison.json"
    report_path = output_dir / "model_comparison.md"

    # Run comparison
    compare_models(data_dir, human_path, comparison_path, verbose)

    # Generate report
    if comparison_path.exists():
        generate_comparison_report(comparison_path, report_path, verbose)

    print("\nComparison complete!")
    print(f"  Data: {comparison_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data"
    human_path = data_dir / "human_samples.jsonl"
    output_dir = base_path / "results"
    main(data_dir, human_path, output_dir, verbose=True)
