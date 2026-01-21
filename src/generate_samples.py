"""
Generate text samples from Claude models via the Anthropic API.

Implements resumability, rate limiting, and progress tracking.
Supports multiple Claude model versions for comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Available models for comparison
AVAILABLE_MODELS = {
    "opus-4.5": "claude-opus-4-5-20251101",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-3.7": "claude-3-7-sonnet-20250219",
    "haiku-3.5": "claude-3-5-haiku-20241022",
    "haiku-3": "claude-3-haiku-20240307",
}

DEFAULT_MODEL = "opus-4.5"

# Rate limiting settings
REQUESTS_PER_MINUTE = 50  # Conservative default
MIN_DELAY_BETWEEN_REQUESTS = 1.2  # seconds


def get_model_id(model_name: str) -> str:
    """Get full model ID from short name."""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    # Allow passing full model ID directly
    return model_name


def load_prompts(prompts_path: Path) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_existing_samples(samples_path: Path) -> set[str]:
    """Load IDs of already-generated samples for resumability."""
    existing_ids = set()
    if samples_path.exists():
        with open(samples_path) as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    existing_ids.add(sample["id"])
                except json.JSONDecodeError:
                    continue
    return existing_ids


def generate_sample(
    client: anthropic.Anthropic,
    prompt_data: dict,
    model_id: str,
    max_tokens: int = 1024
) -> dict | None:
    """Generate a single sample from the specified model."""
    try:
        # Map expected length to token limits
        length_tokens = {
            "short": 256,
            "medium": 512,
            "long": 1024,
        }
        tokens = length_tokens.get(prompt_data.get("expected_length", "medium"), 512)

        response = client.messages.create(
            model=model_id,
            max_tokens=tokens,
            messages=[
                {"role": "user", "content": prompt_data["prompt"]}
            ]
        )

        # Extract text content
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        return {
            "id": prompt_data["id"],
            "category": prompt_data["category"],
            "prompt": prompt_data["prompt"],
            "response": text_content,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
        }

    except anthropic.RateLimitError:
        print("\nRate limited, waiting 60 seconds...")
        time.sleep(60)
        return None
    except anthropic.APIError as e:
        print(f"\nAPI error for {prompt_data['id']}: {e}")
        return None
    except Exception as e:
        print(f"\nUnexpected error for {prompt_data['id']}: {e}")
        return None


def save_sample(sample: dict, output_path: Path) -> None:
    """Append a single sample to the output file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(sample) + "\n")


def generate_samples(
    prompts_path: Path,
    output_path: Path,
    model_id: str,
    num_samples: int | None = None,
    resume: bool = True,
    verbose: bool = False
) -> dict:
    """
    Generate samples from the specified model for the given prompts.

    Args:
        prompts_path: Path to prompts JSONL file
        output_path: Path to output samples JSONL file
        model_id: Full model ID to use
        num_samples: Maximum number of samples to generate (None = all)
        resume: Skip already-generated samples
        verbose: Print detailed progress

    Returns:
        Stats dict with counts
    """
    # Initialize client
    client = anthropic.Anthropic()

    # Load prompts
    prompts = load_prompts(prompts_path)
    if verbose:
        print(f"Loaded {len(prompts)} prompts")

    # Check for existing samples
    existing_ids = set()
    if resume:
        existing_ids = load_existing_samples(output_path)
        if verbose and existing_ids:
            print(f"Found {len(existing_ids)} existing samples, will skip")

    # Filter to remaining prompts
    remaining_prompts = [p for p in prompts if p["id"] not in existing_ids]

    # Limit number if specified
    if num_samples is not None:
        remaining_prompts = remaining_prompts[:num_samples]

    if not remaining_prompts:
        print("No samples to generate")
        return {"generated": 0, "skipped": len(existing_ids), "failed": 0}

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples with progress bar
    stats = {"generated": 0, "skipped": len(existing_ids), "failed": 0}
    last_request_time = 0

    with tqdm(total=len(remaining_prompts), desc=f"Generating ({model_id})") as pbar:
        for prompt_data in remaining_prompts:
            # Rate limiting
            elapsed = time.time() - last_request_time
            if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
                time.sleep(MIN_DELAY_BETWEEN_REQUESTS - elapsed)

            # Generate
            last_request_time = time.time()
            sample = generate_sample(client, prompt_data, model_id)

            if sample:
                save_sample(sample, output_path)
                stats["generated"] += 1
                if verbose:
                    pbar.set_postfix(tokens=sample["output_tokens"])
            else:
                stats["failed"] += 1
                # Retry once after failure
                time.sleep(2)
                sample = generate_sample(client, prompt_data, model_id)
                if sample:
                    save_sample(sample, output_path)
                    stats["generated"] += 1
                    stats["failed"] -= 1

            pbar.update(1)

    return stats


def main(
    prompts_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    num_samples: int | None = None,
    resume: bool = True,
    verbose: bool = False
) -> dict:
    """Main entry point."""
    model_id = get_model_id(model)

    print("Generating samples...")
    print(f"  Model: {model} ({model_id})")
    print(f"  Prompts: {prompts_path}")
    print(f"  Output: {output_path}")
    print(f"  Max samples: {num_samples or 'all'}")
    print(f"  Resume: {resume}")
    print()

    stats = generate_samples(
        prompts_path=prompts_path,
        output_path=output_path,
        model_id=model_id,
        num_samples=num_samples,
        resume=resume,
        verbose=verbose
    )

    print("\nGeneration complete:")
    print(f"  Generated: {stats['generated']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")

    return stats


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    prompts_path = base_path / "data" / "prompts.jsonl"
    output_path = base_path / "data" / "opus_samples.jsonl"
    main(prompts_path, output_path, verbose=True)
