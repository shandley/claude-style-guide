#!/usr/bin/env python3
"""
Opus Styleguide Pipeline

Single CLI entry point for generating a personal writing styleguide
that identifies LLM patterns to avoid.

Usage:
    python run_pipeline.py all --verbose
    python run_pipeline.py generate-prompts
    python run_pipeline.py generate-samples --n 200
    python run_pipeline.py generate-samples --model sonnet-3.5 --n 200
    python run_pipeline.py generate-all-models --n 100
    python run_pipeline.py compare-models
    python run_pipeline.py fetch-human-corpus --n 10000
    python run_pipeline.py analyze
    python run_pipeline.py report
    python run_pipeline.py status
"""

import sys
from pathlib import Path

import click

# Add src to path
SRC_PATH = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Default paths
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"

PROMPTS_PATH = DATA_PATH / "prompts.jsonl"
OPUS_SAMPLES_PATH = DATA_PATH / "opus_samples.jsonl"
HUMAN_SAMPLES_PATH = DATA_PATH / "human_samples.jsonl"
MARKERS_PATH = RESULTS_PATH / "markers.json"
STYLEGUIDE_PATH = RESULTS_PATH / "styleguide.md"
COMPARISON_PATH = RESULTS_PATH / "model_comparison.json"
COMPARISON_REPORT_PATH = RESULTS_PATH / "model_comparison.md"

# Available models
AVAILABLE_MODELS = [
    "opus-4.5", "sonnet-4", "sonnet-3.7", "haiku-3.5", "haiku-3"
]


def get_model_samples_path(model: str) -> Path:
    """Get the samples path for a given model."""
    if model == "opus-4.5":
        return OPUS_SAMPLES_PATH  # Legacy path for backward compatibility
    return DATA_PATH / f"{model.replace('.', '_')}_samples.jsonl"


@click.group()
def cli():
    """Claude Styleguide Generator - Identify LLM patterns to avoid in your writing."""
    pass


@cli.command()
@click.option("--n", default=300, help="Number of prompts to generate")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def generate_prompts(n: int, verbose: bool):
    """Generate diverse prompt bank for sampling."""
    from generate_prompts import main as generate_prompts_main
    click.echo(f"Generating {n} prompts...")
    generate_prompts_main(PROMPTS_PATH, num_prompts=n)


@cli.command()
@click.option("--n", default=200, help="Number of samples to generate")
@click.option("--model", "-m", default="opus-4.5",
              type=click.Choice(AVAILABLE_MODELS),
              help="Model to use for generation")
@click.option("--resume/--no-resume", default=True, help="Resume from existing samples")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def generate_samples(n: int, model: str, resume: bool, verbose: bool):
    """Generate text samples from a Claude model via API."""
    from generate_prompts import main as generate_prompts_main
    from generate_samples import main as generate_samples_main

    output_path = get_model_samples_path(model)
    click.echo(f"Generating up to {n} samples from {model}...")

    # Check if prompts exist
    if not PROMPTS_PATH.exists():
        click.echo("No prompts found. Generating prompts first...")
        generate_prompts_main(PROMPTS_PATH, num_prompts=n + 50)

    generate_samples_main(
        prompts_path=PROMPTS_PATH,
        output_path=output_path,
        model=model,
        num_samples=n,
        resume=resume,
        verbose=verbose
    )


@cli.command("generate-all-models")
@click.option("--n", default=100, help="Number of samples per model")
@click.option("--models", "-m", default="opus-3,sonnet-3.5,opus-4.5",
              help="Comma-separated list of models to sample")
@click.option("--resume/--no-resume", default=True, help="Resume from existing samples")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def generate_all_models(n: int, models: str, resume: bool, verbose: bool):
    """Generate samples from multiple Claude models for comparison."""
    from generate_prompts import main as generate_prompts_main
    from generate_samples import main as generate_samples_main

    model_list = [m.strip() for m in models.split(",")]

    # Validate models
    for model in model_list:
        if model not in AVAILABLE_MODELS:
            click.echo(f"Error: Unknown model '{model}'", err=True)
            click.echo(f"Available: {', '.join(AVAILABLE_MODELS)}", err=True)
            sys.exit(1)

    click.echo("=" * 60)
    click.echo("Multi-Model Sample Generation")
    click.echo("=" * 60)
    click.echo(f"Models: {', '.join(model_list)}")
    click.echo(f"Samples per model: {n}")
    click.echo()

    # Check if prompts exist
    if not PROMPTS_PATH.exists():
        click.echo("Generating prompts first...")
        generate_prompts_main(PROMPTS_PATH, num_prompts=n + 50)
        click.echo()

    # Generate samples for each model
    for i, model in enumerate(model_list, 1):
        click.echo(f"[{i}/{len(model_list)}] Generating {model} samples...")
        output_path = get_model_samples_path(model)

        generate_samples_main(
            prompts_path=PROMPTS_PATH,
            output_path=output_path,
            model=model,
            num_samples=n,
            resume=resume,
            verbose=verbose
        )
        click.echo()

    click.echo("=" * 60)
    click.echo("All models complete!")
    click.echo("Run 'python run_pipeline.py compare-models' to generate comparison.")
    click.echo("=" * 60)


@cli.command("compare-models")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def compare_models(verbose: bool):
    """Compare patterns across different Claude model versions."""
    from compare import main as compare_main

    click.echo("Comparing model patterns...")

    # Check if human corpus exists
    if not HUMAN_SAMPLES_PATH.exists():
        click.echo("Error: No human corpus found. Run 'fetch-human-corpus' first.", err=True)
        sys.exit(1)

    # Check if any model samples exist
    found_models = []
    for model in AVAILABLE_MODELS:
        path = get_model_samples_path(model)
        if path.exists():
            found_models.append(model)

    if not found_models:
        click.echo("Error: No model samples found. Run 'generate-samples' first.", err=True)
        sys.exit(1)

    click.echo(f"Found samples for: {', '.join(found_models)}")

    compare_main(
        data_dir=DATA_PATH,
        human_path=HUMAN_SAMPLES_PATH,
        output_dir=RESULTS_PATH,
        verbose=verbose
    )

    click.echo(f"\nComparison report: {COMPARISON_REPORT_PATH}")


@cli.command()
@click.option("--n", default=10000, help="Number of human samples to fetch")
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip if already fetched")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def fetch_human_corpus(n: int, skip_existing: bool, verbose: bool):
    """Fetch human text corpus from HuggingFace."""
    from fetch_human_corpus import main as fetch_human_corpus_main
    click.echo(f"Fetching {n} human text samples...")
    fetch_human_corpus_main(
        output_path=HUMAN_SAMPLES_PATH,
        num_samples=n,
        verbose=verbose,
        skip_existing=skip_existing
    )


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def analyze(verbose: bool):
    """Run statistical analysis comparing Opus to human text."""
    from analyze import main as analyze_main
    click.echo("Running analysis...")

    # Check prerequisites
    if not OPUS_SAMPLES_PATH.exists():
        click.echo("Error: No Opus samples found. Run 'generate-samples' first.", err=True)
        sys.exit(1)

    if not HUMAN_SAMPLES_PATH.exists():
        click.echo("Error: No human corpus found. Run 'fetch-human-corpus' first.", err=True)
        sys.exit(1)

    analyze_main(
        opus_path=OPUS_SAMPLES_PATH,
        human_path=HUMAN_SAMPLES_PATH,
        output_path=MARKERS_PATH,
        verbose=verbose
    )


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def report(verbose: bool):
    """Generate markdown styleguide from analysis."""
    from report import main as report_main
    click.echo("Generating styleguide...")

    if not MARKERS_PATH.exists():
        click.echo("Error: No analysis results found. Run 'analyze' first.", err=True)
        sys.exit(1)

    report_main(
        markers_path=MARKERS_PATH,
        output_path=STYLEGUIDE_PATH,
        verbose=verbose
    )

    click.echo(f"\nStyleguide ready: {STYLEGUIDE_PATH}")


@cli.command()
@click.option("--n-samples", default=200, help="Number of Opus samples to generate")
@click.option("--n-human", default=10000, help="Number of human samples to fetch")
@click.option("--resume/--no-resume", default=True, help="Resume interrupted operations")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def all(n_samples: int, n_human: int, resume: bool, verbose: bool):
    """Run the complete pipeline."""
    from generate_prompts import main as generate_prompts_main
    from generate_samples import main as generate_samples_main
    from fetch_human_corpus import main as fetch_human_corpus_main
    from analyze import main as analyze_main
    from report import main as report_main

    click.echo("=" * 60)
    click.echo("Claude Styleguide Generator - Full Pipeline")
    click.echo("=" * 60)
    click.echo()

    # Step 1: Generate prompts
    click.echo("[1/5] Generating prompts...")
    if PROMPTS_PATH.exists() and resume:
        click.echo("  Prompts already exist, skipping.")
    else:
        generate_prompts_main(PROMPTS_PATH, num_prompts=n_samples + 50)
    click.echo()

    # Step 2: Generate Opus samples
    click.echo("[2/5] Generating Opus samples...")
    generate_samples_main(
        prompts_path=PROMPTS_PATH,
        output_path=OPUS_SAMPLES_PATH,
        model="opus-4.5",
        num_samples=n_samples,
        resume=resume,
        verbose=verbose
    )
    click.echo()

    # Step 3: Fetch human corpus
    click.echo("[3/5] Fetching human corpus...")
    fetch_human_corpus_main(
        output_path=HUMAN_SAMPLES_PATH,
        num_samples=n_human,
        verbose=verbose,
        skip_existing=resume
    )
    click.echo()

    # Step 4: Analyze
    click.echo("[4/5] Running analysis...")
    analyze_main(
        opus_path=OPUS_SAMPLES_PATH,
        human_path=HUMAN_SAMPLES_PATH,
        output_path=MARKERS_PATH,
        verbose=verbose
    )
    click.echo()

    # Step 5: Generate report
    click.echo("[5/5] Generating styleguide...")
    report_main(
        markers_path=MARKERS_PATH,
        output_path=STYLEGUIDE_PATH,
        verbose=verbose
    )

    click.echo()
    click.echo("=" * 60)
    click.echo("Pipeline complete!")
    click.echo(f"  Styleguide: {STYLEGUIDE_PATH}")
    click.echo(f"  Raw data: {MARKERS_PATH}")
    click.echo("=" * 60)


@cli.command()
def status():
    """Check pipeline status and file counts."""
    click.echo("Pipeline Status")
    click.echo("-" * 50)

    def count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with open(path) as f:
            return sum(1 for _ in f)

    # Prompts
    if PROMPTS_PATH.exists():
        count = count_lines(PROMPTS_PATH)
        click.echo(f"Prompts:        {count} prompts ready")
    else:
        click.echo("Prompts:        Not generated")

    # Model samples
    click.echo("\nModel Samples:")
    for model in AVAILABLE_MODELS:
        path = get_model_samples_path(model)
        if path.exists():
            count = count_lines(path)
            click.echo(f"  {model:12}  {count} samples")

    # Human corpus
    click.echo()
    if HUMAN_SAMPLES_PATH.exists():
        count = count_lines(HUMAN_SAMPLES_PATH)
        click.echo(f"Human corpus:   {count} samples")
    else:
        click.echo("Human corpus:   Not fetched")

    # Analysis
    if MARKERS_PATH.exists():
        import json
        with open(MARKERS_PATH) as f:
            data = json.load(f)
        markers = len(data.get("markers", []))
        click.echo(f"Analysis:       {markers} markers identified")
    else:
        click.echo("Analysis:       Not run")

    # Styleguide
    if STYLEGUIDE_PATH.exists():
        click.echo("Styleguide:     Ready")
    else:
        click.echo("Styleguide:     Not generated")

    # Model comparison
    if COMPARISON_REPORT_PATH.exists():
        click.echo("Comparison:     Ready")
    else:
        click.echo("Comparison:     Not generated")

    click.echo("-" * 50)


@cli.command()
def clean():
    """Remove all generated files."""
    import shutil

    click.confirm("This will delete all generated data. Continue?", abort=True)

    for path in [DATA_PATH, RESULTS_PATH]:
        if path.exists():
            shutil.rmtree(path)
            click.echo(f"Removed {path}")

    click.echo("Clean complete.")


@cli.command("list-models")
def list_models():
    """List available Claude models for sampling."""
    from generate_samples import AVAILABLE_MODELS as models

    click.echo("Available models:")
    click.echo("-" * 40)
    for short_name, full_id in models.items():
        click.echo(f"  {short_name:12}  {full_id}")
    click.echo("-" * 40)
    click.echo("\nUsage: python run_pipeline.py generate-samples --model <name>")


if __name__ == "__main__":
    cli()
