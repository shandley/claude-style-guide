"""
Fetch human-written text corpus from HuggingFace for comparison.

Uses a mix of sources to get varied professional/technical writing:
- Wikipedia for explanatory content
- StackExchange for technical Q&A
- OpenWebText for general web content
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Random seed for reproducibility
random.seed(42)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Basic cleaning
    text = text.strip()
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text


def filter_by_length(text: str, min_chars: int = 100, max_chars: int = 3000) -> bool:
    """Filter texts by character length."""
    length = len(text)
    return min_chars <= length <= max_chars


def fetch_wikipedia_samples(num_samples: int, verbose: bool = False) -> list[dict]:
    """Fetch samples from Wikipedia dataset."""
    if verbose:
        print(f"Fetching {num_samples} Wikipedia samples...")

    samples = []
    try:
        # Use Wikipedia dataset - streaming to avoid downloading everything
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        collected = 0
        for item in tqdm(dataset, desc="Wikipedia", disable=not verbose):
            text = clean_text(item.get("text", ""))

            # Skip very short or very long articles
            if not filter_by_length(text, min_chars=200, max_chars=2500):
                continue

            # Take first ~2000 chars to match Opus response lengths
            if len(text) > 2000:
                # Try to break at sentence boundary
                cutoff = text[:2000].rfind(". ")
                if cutoff > 500:
                    text = text[:cutoff + 1]
                else:
                    text = text[:2000]

            samples.append({
                "source": "wikipedia",
                "text": text,
                "title": item.get("title", ""),
            })
            collected += 1

            if collected >= num_samples:
                break

    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")

    return samples


def fetch_openwebtext_samples(num_samples: int, verbose: bool = False) -> list[dict]:
    """Fetch samples from OpenWebText dataset."""
    if verbose:
        print(f"Fetching {num_samples} OpenWebText samples...")

    samples = []
    try:
        dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        collected = 0
        for item in tqdm(dataset, desc="OpenWebText", disable=not verbose):
            text = clean_text(item.get("text", ""))

            if not filter_by_length(text, min_chars=150, max_chars=2500):
                continue

            # Trim to reasonable length
            if len(text) > 2000:
                cutoff = text[:2000].rfind(". ")
                if cutoff > 500:
                    text = text[:cutoff + 1]
                else:
                    text = text[:2000]

            samples.append({
                "source": "openwebtext",
                "text": text,
            })
            collected += 1

            if collected >= num_samples:
                break

    except Exception as e:
        print(f"Error fetching OpenWebText: {e}")

    return samples


def fetch_c4_samples(num_samples: int, verbose: bool = False) -> list[dict]:
    """Fetch samples from C4 dataset (cleaner web text)."""
    if verbose:
        print(f"Fetching {num_samples} C4 samples...")

    samples = []
    try:
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        collected = 0
        for item in tqdm(dataset, desc="C4", disable=not verbose):
            text = clean_text(item.get("text", ""))

            if not filter_by_length(text, min_chars=150, max_chars=2500):
                continue

            if len(text) > 2000:
                cutoff = text[:2000].rfind(". ")
                if cutoff > 500:
                    text = text[:cutoff + 1]
                else:
                    text = text[:2000]

            samples.append({
                "source": "c4",
                "text": text,
                "url": item.get("url", ""),
            })
            collected += 1

            if collected >= num_samples:
                break

    except Exception as e:
        print(f"Error fetching C4: {e}")

    return samples


def fetch_pile_samples(num_samples: int, verbose: bool = False) -> list[dict]:
    """Fetch samples from The Pile subset (diverse sources)."""
    if verbose:
        print(f"Fetching {num_samples} Pile samples...")

    samples = []
    try:
        # Use a subset of the pile
        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        collected = 0
        # Skip some to get variety
        skip_count = 0
        for item in tqdm(dataset, desc="Pile", disable=not verbose):
            skip_count += 1
            if skip_count % 3 != 0:  # Sample every 3rd item
                continue

            text = clean_text(item.get("text", ""))
            meta = item.get("meta", {})
            pile_set = meta.get("pile_set_name", "unknown") if isinstance(meta, dict) else "unknown"

            # Skip code-heavy sources for this analysis
            if pile_set in ["Github", "DM Mathematics", "Ubuntu IRC"]:
                continue

            if not filter_by_length(text, min_chars=150, max_chars=2500):
                continue

            if len(text) > 2000:
                cutoff = text[:2000].rfind(". ")
                if cutoff > 500:
                    text = text[:cutoff + 1]
                else:
                    text = text[:2000]

            samples.append({
                "source": f"pile_{pile_set}",
                "text": text,
            })
            collected += 1

            if collected >= num_samples:
                break

    except Exception as e:
        print(f"Error fetching Pile: {e}")

    return samples


def save_samples(samples: list[dict], output_path: Path) -> None:
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, sample in enumerate(samples):
            sample["id"] = f"human_{i:05d}"
            f.write(json.dumps(sample) + "\n")


def fetch_human_corpus(
    output_path: Path,
    num_samples: int = 10000,
    verbose: bool = False,
    skip_existing: bool = True
) -> dict:
    """
    Fetch human text corpus from multiple sources.

    Distribution:
    - Wikipedia: 40% (explanatory, encyclopedic)
    - OpenWebText: 30% (web articles, blogs)
    - C4: 20% (clean web text)
    - Pile: 10% (diverse sources)
    """
    if skip_existing and output_path.exists():
        # Count existing
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples * 0.9:
            print(f"Found {existing} existing samples, skipping fetch")
            return {"total": existing, "skipped": True}

    # Calculate per-source counts
    wiki_count = int(num_samples * 0.4)
    owt_count = int(num_samples * 0.3)
    c4_count = int(num_samples * 0.2)
    pile_count = num_samples - wiki_count - owt_count - c4_count

    all_samples = []

    # Fetch from each source
    wiki_samples = fetch_wikipedia_samples(wiki_count, verbose)
    all_samples.extend(wiki_samples)

    owt_samples = fetch_openwebtext_samples(owt_count, verbose)
    all_samples.extend(owt_samples)

    c4_samples = fetch_c4_samples(c4_count, verbose)
    all_samples.extend(c4_samples)

    pile_samples = fetch_pile_samples(pile_count, verbose)
    all_samples.extend(pile_samples)

    # Shuffle
    random.shuffle(all_samples)

    # Save
    save_samples(all_samples, output_path)

    # Stats
    from collections import Counter
    sources = Counter(s["source"].split("_")[0] for s in all_samples)

    stats = {
        "total": len(all_samples),
        "by_source": dict(sources),
    }

    if verbose:
        print(f"\nFetched {len(all_samples)} human text samples:")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count}")

    return stats


def main(
    output_path: Path,
    num_samples: int = 10000,
    verbose: bool = False,
    skip_existing: bool = True
) -> dict:
    """Main entry point."""
    print("Fetching human corpus...")
    print(f"  Output: {output_path}")
    print(f"  Target samples: {num_samples}")
    print()

    stats = fetch_human_corpus(
        output_path=output_path,
        num_samples=num_samples,
        verbose=verbose,
        skip_existing=skip_existing
    )

    print(f"\nCorpus fetch complete: {stats['total']} samples")
    return stats


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "human_samples.jsonl"
    main(output_path, num_samples=10000, verbose=True)
