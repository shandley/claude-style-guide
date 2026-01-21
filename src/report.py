"""
Generate a personal writing styleguide from analysis results.

Focused on actionable "what to avoid" guidance for professional writing
to help avoid common LLM patterns.
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Human alternatives for common LLM-isms
ALTERNATIVES = {
    # Hedging - often can just be deleted
    "it's important to note": ["Note:", "delete entirely", "just state the fact"],
    "it's worth noting": ["Note:", "delete entirely"],
    "it's worth mentioning": ["delete entirely", "just state it"],
    "it should be noted": ["Note:", "delete entirely"],
    "generally speaking": ["Usually,", "Often,", "delete entirely"],
    "in general": ["Usually,", "Often,", "delete entirely"],
    "that said": ["But", "However,", "delete entirely"],
    "having said that": ["But", "However,"],
    "that being said": ["But", "However,", "Still,"],
    "with that in mind": ["So", "Given this,", "delete entirely"],

    # Transitions - often overly formal
    "additionally": ["Also,", "And", "delete and merge sentences"],
    "furthermore": ["Also,", "And", "Plus,"],
    "moreover": ["Also,", "And"],
    "in addition": ["Also,", "Plus,"],
    "conversely": ["But", "On the flip side,"],
    "nevertheless": ["Still,", "But", "Even so,"],
    "nonetheless": ["Still,", "But", "Even so,"],
    "consequently": ["So", "As a result,"],
    "therefore": ["So", "This means"],
    "thus": ["So", "This means"],
    "hence": ["So", "This means"],
    "accordingly": ["So", "Based on this,"],

    # Fillers - almost always delete
    "in order to": ["to"],
    "due to the fact that": ["because", "since"],
    "the fact that": ["that", "delete entirely"],
    "it is important that": ["delete entirely", "just state it"],
    "it is essential that": ["delete entirely", "just state it"],
    "it is crucial that": ["delete entirely", "just state it"],
    "it is necessary to": ["you need to", "delete entirely"],
    "in terms of": ["for", "regarding", "delete entirely"],
    "when it comes to": ["for", "with", "regarding"],
    "with respect to": ["for", "about", "regarding"],
    "with regard to": ["for", "about"],
    "at the end of the day": ["Ultimately,", "delete entirely"],
    "at this point in time": ["now", "currently"],

    # Structure phrases
    "let me explain": ["delete entirely"],
    "let's break this down": ["delete entirely", "just break it down"],
    "let's dive into": ["delete entirely"],
    "let's explore": ["delete entirely"],
    "here's the thing": ["delete entirely", "The thing is,"],
    "first and foremost": ["First,"],
    "last but not least": ["Finally,", "Also,"],

    # Conclusion phrases
    "in summary": ["To sum up,", "In short,"],
    "to summarize": ["In short,", "delete entirely"],
    "in conclusion": ["delete entirely", "So,"],
    "to conclude": ["delete entirely", "Finally,"],
    "all in all": ["Overall,", "delete entirely"],
    "in essence": ["Basically,", "delete entirely"],
    "essentially": ["Basically,", "delete entirely"],
    "ultimately": ["In the end,", "delete entirely"],
    "at its core": ["Basically,", "delete entirely"],
    "fundamentally": ["Basically,", "delete entirely"],

    # Emphasis - often unnecessary
    "absolutely": ["delete entirely", "yes"],
    "definitely": ["delete entirely"],
    "certainly": ["delete entirely"],
    "clearly": ["delete entirely"],
    "obviously": ["delete entirely"],
    "of course": ["delete entirely"],
    "naturally": ["delete entirely"],
    "undoubtedly": ["delete entirely"],
    "without a doubt": ["delete entirely"],
    "indeed": ["delete entirely"],

    # LLM favorites - use simpler words
    "delve": ["explore", "look at", "examine", "dig into"],
    "crucial": ["important", "key", "critical"],
    "vital": ["important", "essential", "key"],
    "pivotal": ["important", "key", "central"],
    "robust": ["strong", "solid", "reliable"],
    "comprehensive": ["complete", "full", "thorough"],
    "nuanced": ["subtle", "complex", "detailed"],
    "multifaceted": ["complex", "varied"],
    "intricate": ["complex", "detailed"],
    "meticulous": ["careful", "thorough", "detailed"],
    "seamlessly": ["smoothly", "easily"],
    "leverage": ["use", "apply"],
    "utilize": ["use"],
    "facilitate": ["help", "enable", "make easier"],
    "foster": ["encourage", "support", "build"],
    "realm": ["area", "field", "domain"],
    "landscape": ["field", "space", "area"],
    "paradigm": ["model", "approach", "way"],
    "myriad": ["many", "lots of", "numerous"],
    "plethora": ["many", "lots of"],
    "tapestry": ["mix", "combination", "blend"],
    "embark": ["start", "begin"],
    "endeavor": ["try", "attempt", "effort"],
    "aforementioned": ["this", "that", "the"],
}

# Severity levels based on log-odds
def get_severity(log_odds: float) -> str:
    """Categorize marker severity."""
    if log_odds > 2.5:
        return "high"
    elif log_odds > 1.5:
        return "medium"
    else:
        return "low"


def generate_styleguide(
    markers_path: Path,
    output_path: Path,
    verbose: bool = False
) -> None:
    """Generate the markdown styleguide."""

    # Load analysis results
    with open(markers_path) as f:
        results = json.load(f)

    markers = results["markers"]
    corpus_stats = results["corpus_stats"]
    summary_stats = results.get("summary_stats", {})

    # Group markers by type and severity
    by_type = defaultdict(list)
    for m in markers:
        by_type[m["type"]].append(m)

    # Start building the report
    lines = []

    # Header
    lines.append("# Personal Writing Styleguide: Avoiding LLM Patterns")
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d')} from analysis of {corpus_stats['opus_samples']} AI samples vs {corpus_stats['human_samples']} human texts*")
    lines.append("")

    # Executive Summary
    lines.append("## Quick Reference: Top Patterns to Avoid")
    lines.append("")
    lines.append("These are the most distinctively \"AI-sounding\" patterns. Avoiding these will make your writing sound more human.")
    lines.append("")

    # Top 15 most distinctive
    top_markers = sorted(markers, key=lambda m: -m["log_odds"])[:15]
    lines.append("| Pattern | Type | AI vs Human | Action |")
    lines.append("|---------|------|-------------|--------|")
    for m in top_markers:
        ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
        item = m["item"]
        alternatives = ALTERNATIVES.get(item.lower(), ["consider rephrasing"])
        action = alternatives[0] if alternatives else "rephrase"
        lines.append(f"| {item} | {m['type']} | {ratio:.1f}x | {action} |")
    lines.append("")

    # Self-Editing Checklist
    lines.append("## Self-Editing Checklist")
    lines.append("")
    lines.append("Run through this checklist when reviewing your writing:")
    lines.append("")
    lines.append("### High Priority (Very AI-like)")
    lines.append("")
    high_priority = [m for m in markers if get_severity(m["log_odds"]) == "high"][:20]
    for m in high_priority:
        lines.append(f"- [ ] Search for \"{m['item']}\" and replace or delete")
    lines.append("")

    lines.append("### Medium Priority (Moderately AI-like)")
    lines.append("")
    medium_priority = [m for m in markers if get_severity(m["log_odds"]) == "medium"][:15]
    for m in medium_priority:
        lines.append(f"- [ ] Check uses of \"{m['item']}\"")
    lines.append("")

    # Words to Avoid
    lines.append("## Words to Avoid")
    lines.append("")
    lines.append("These individual words appear significantly more often in AI writing than human writing.")
    lines.append("")

    word_markers = [m for m in by_type.get("word", []) if m["opus_count"] >= 5]
    word_markers.sort(key=lambda m: -m["log_odds"])

    if word_markers:
        lines.append("| Word | AI Rate | Human Rate | Ratio | Alternatives |")
        lines.append("|------|---------|------------|-------|--------------|")
        for m in word_markers[:30]:
            ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
            alts = ALTERNATIVES.get(m["item"].lower(), [])
            alt_str = ", ".join(alts[:3]) if alts else "—"
            lines.append(f"| {m['item']} | {m['opus_rate']*100:.2f}% | {m['human_rate']*100:.3f}% | {ratio:.1f}x | {alt_str} |")
        lines.append("")

    # Phrases to Avoid
    lines.append("## Phrases to Avoid")
    lines.append("")
    lines.append("These multi-word phrases are telltale signs of AI-generated text.")
    lines.append("")

    # Group phrases by category
    phrase_categories = defaultdict(list)
    for m in markers:
        if m["type"].startswith("phrase_"):
            category = m["type"].replace("phrase_", "")
            phrase_categories[category].append(m)

    category_names = {
        "hedging": "Hedging Phrases (often unnecessary)",
        "transition": "Overly Formal Transitions",
        "filler": "Filler Phrases (delete these)",
        "structure": "Meta-Structure Phrases",
        "conclusion": "Conclusion Phrases",
        "emphasis": "Emphasis Words (usually delete)",
        "llm_favorite": "AI Favorite Words",
    }

    for category, cat_markers in sorted(phrase_categories.items()):
        if not cat_markers:
            continue

        cat_markers.sort(key=lambda m: -m["log_odds"])
        lines.append(f"### {category_names.get(category, category.title())}")
        lines.append("")

        for m in cat_markers[:10]:
            ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
            alts = ALTERNATIVES.get(m["item"].lower(), ["rephrase"])

            lines.append(f"**{m['item']}** ({ratio:.1f}x more common in AI)")
            if m["example_context"]:
                lines.append(f"> {m['example_context']}")
            lines.append(f"- *Instead:* {', '.join(alts)}")
            lines.append("")

    # Sentence Starters
    lines.append("## Sentence Starters to Vary")
    lines.append("")
    lines.append("AI tends to overuse certain sentence openers. If you find yourself starting many sentences the same way, vary it up.")
    lines.append("")

    starter_markers = [m for m in by_type.get("sentence_starter", []) if m["opus_count"] >= 5]
    starter_markers.sort(key=lambda m: -m["log_odds"])

    if starter_markers:
        lines.append("| Starter | AI Usage | Human Usage | Ratio |")
        lines.append("|---------|----------|-------------|-------|")
        for m in starter_markers[:15]:
            ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
            lines.append(f"| {m['item'].title()} | {m['opus_rate']*100:.1f}% | {m['human_rate']*100:.2f}% | {ratio:.1f}x |")
        lines.append("")

    # N-grams
    lines.append("## Common AI Bigrams and Trigrams")
    lines.append("")
    lines.append("These word combinations are distinctively AI-like.")
    lines.append("")

    bigram_markers = [m for m in by_type.get("bigram", []) if m["opus_count"] >= 3]
    bigram_markers.sort(key=lambda m: -m["log_odds"])

    if bigram_markers:
        lines.append("### Bigrams (2-word combinations)")
        lines.append("")
        for m in bigram_markers[:15]:
            ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
            lines.append(f"- **\"{m['item']}\"** — {ratio:.1f}x more common in AI")
        lines.append("")

    trigram_markers = [m for m in by_type.get("trigram", []) if m["opus_count"] >= 3]
    trigram_markers.sort(key=lambda m: -m["log_odds"])

    if trigram_markers:
        lines.append("### Trigrams (3-word combinations)")
        lines.append("")
        for m in trigram_markers[:15]:
            ratio = m["opus_rate"] / m["human_rate"] if m["human_rate"] > 0 else float('inf')
            lines.append(f"- **\"{m['item']}\"** — {ratio:.1f}x more common in AI")
        lines.append("")

    # Structural Patterns
    lines.append("## Structural Patterns")
    lines.append("")
    lines.append("Beyond word choice, AI writing has distinctive structural patterns.")
    lines.append("")

    if summary_stats:
        # Sentence Length Distribution
        lines.append("### Sentence Length Distribution")
        lines.append("")
        opus_dist = summary_stats.get("opus_sentence_distribution", {})
        human_dist = summary_stats.get("human_sentence_distribution", {})

        if opus_dist and human_dist:
            lines.append("| Metric | AI | Human | Insight |")
            lines.append("|--------|-----|-------|---------|")
            lines.append(f"| Mean length | {opus_dist.get('mean', 0)} words | {human_dist.get('mean', 0)} words | AI sentences slightly longer |")
            lines.append(f"| Coefficient of variation | {opus_dist.get('coefficient_of_variation', 0)}% | {human_dist.get('coefficient_of_variation', 0)}% | AI has more extreme variation |")
            lines.append(f"| Short sentences (1-10 words) | {opus_dist.get('pct_short_1_10', 0)}% | {human_dist.get('pct_short_1_10', 0)}% | AI uses more short sentences |")
            lines.append(f"| Medium sentences (11-25 words) | {opus_dist.get('pct_medium_11_25', 0)}% | {human_dist.get('pct_medium_11_25', 0)}% | Human writing more consistent |")
            lines.append(f"| Long sentences (26+ words) | {opus_dist.get('pct_long_26_plus', 0)}% | {human_dist.get('pct_long_26_plus', 0)}% | Similar long sentence usage |")
            lines.append("")
            lines.append("*Tip: AI tends to alternate between very short and very long sentences. Human writing has more medium-length sentences.*")
            lines.append("")

        # Passive Voice
        lines.append("### Passive Voice")
        lines.append("")
        opus_passive = summary_stats.get("opus_passive_voice_pct", 0)
        human_passive = summary_stats.get("human_passive_voice_pct", 0)
        lines.append(f"- AI uses passive voice in **{opus_passive}%** of sentences")
        lines.append(f"- Human writing uses passive voice in **{human_passive}%** of sentences")
        if opus_passive < human_passive:
            lines.append("- *Surprisingly, AI uses LESS passive voice than humans. Don't over-correct by avoiding all passive constructions.*")
        lines.append("")

        # Paragraph Patterns
        lines.append("### Paragraph Structure")
        lines.append("")
        opus_para = summary_stats.get("opus_paragraph_stats", {})

        if opus_para:
            lines.append(f"- AI average: **{opus_para.get('avg_para_length_words', 0)} words** per paragraph")
            lines.append(f"- AI uses **{opus_para.get('avg_paragraphs_per_doc', 0)} paragraphs** per document on average")
            lines.append("")
            lines.append("*Tip: AI tends to fragment text into many short paragraphs. Consider combining related ideas into longer, more developed paragraphs.*")
            lines.append("")

        lines.append("### List Usage")
        lines.append("")
        opus_lists = summary_stats.get("opus_list_items_per_text", 0)
        human_lists = summary_stats.get("human_list_items_per_text", 0)
        lines.append(f"- AI uses **{opus_lists:.1f}** list items per response on average")
        lines.append(f"- Human writing uses **{human_lists:.1f}** list items per text on average")
        if opus_lists > human_lists * 1.5:
            lines.append("- *Tip: You might be over-using bullet points. Consider prose instead.*")
        lines.append("")

        lines.append("### Punctuation")
        lines.append("")
        lines.append("| Punctuation | AI (per 1k chars) | Human (per 1k chars) | Ratio |")
        lines.append("|-------------|-------------------|----------------------|-------|")
        for punct in ["em_dash", "colon", "semicolon"]:
            opus_rate = summary_stats.get(f"opus_{punct}_per_1k", 0)
            human_rate = summary_stats.get(f"human_{punct}_per_1k", 0)
            if opus_rate > 0 or human_rate > 0:
                ratio = opus_rate / human_rate if human_rate > 0 else float('inf')
                lines.append(f"| {punct.replace('_', ' ')} | {opus_rate:.2f} | {human_rate:.2f} | {ratio:.1f}x |")
        lines.append("")
        lines.append("*Tip: Em dashes are a strong AI signal. Replace with commas, periods, or parentheses.*")
        lines.append("")

    # Before/After Examples
    lines.append("## Before/After Examples")
    lines.append("")
    lines.append("Here's how to rewrite AI-sounding text to sound more natural:")
    lines.append("")

    examples = [
        (
            "It's important to note that this approach has several crucial advantages. Additionally, it facilitates seamless integration.",
            "This approach has several key advantages. It also makes integration easier."
        ),
        (
            "Let's delve into the intricacies of this multifaceted problem. Furthermore, we should leverage existing solutions.",
            "Let's look at this complex problem. We should also use existing solutions."
        ),
        (
            "In order to achieve optimal results, it is essential that you utilize the comprehensive documentation.",
            "To get the best results, use the full documentation."
        ),
        (
            "That being said, the aforementioned methodology provides a robust framework for addressing these challenges.",
            "Still, this method gives you a solid framework for these challenges."
        ),
        (
            "Ultimately, at its core, this represents a pivotal shift in the landscape of software development.",
            "This is a major shift in software development."
        ),
    ]

    for before, after in examples:
        lines.append("**Before (AI-like):**")
        lines.append(f"> {before}")
        lines.append("")
        lines.append("**After (more natural):**")
        lines.append(f"> {after}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("This styleguide was generated by:")
    lines.append("")
    lines.append(f"1. Generating {corpus_stats['opus_samples']} text samples from Claude Opus 4.5")
    lines.append(f"2. Collecting {corpus_stats['human_samples']} human-written texts from Wikipedia, OpenWebText, and other sources")
    lines.append("3. Comparing word frequencies, phrase patterns, and structural features")
    lines.append("4. Identifying patterns where AI usage is statistically significantly higher (p < 0.01) and at least 2x more frequent")
    lines.append("")
    lines.append("The log-odds ratio measures how much more likely a pattern is in AI text vs human text. Higher values = more distinctively AI.")
    lines.append("")

    # Write the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"Styleguide written to {output_path}")
        print(f"  Total markers included: {len(markers)}")


def main(
    markers_path: Path,
    output_path: Path,
    verbose: bool = False
) -> None:
    """Main entry point."""
    print("Generating styleguide...")
    print(f"  Input: {markers_path}")
    print(f"  Output: {output_path}")
    print()

    generate_styleguide(markers_path, output_path, verbose)

    print(f"\nStyleguide complete: {output_path}")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    markers_path = base_path / "results" / "markers.json"
    output_path = base_path / "results" / "styleguide.md"
    main(markers_path, output_path, verbose=True)
