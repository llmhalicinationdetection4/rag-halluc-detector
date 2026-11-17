"""Command line interface. Heavy imports happen inside commands so --help works everywhere."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Contextual hallucination detection for RAG answers.")


def _read_text(value: str | None, file: Path | None, name: str) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if value:
        return value
    raise typer.BadParameter(f"Provide --{name} or --{name}-file")


@app.command()
def verify(
    context: str = typer.Option(None, help="Retrieved context text"),
    answer: str = typer.Option(None, help="LLM answer to check"),
    context_file: Path = typer.Option(None, help="Read context from a file"),
    answer_file: Path = typer.Option(None, help="Read answer from a file"),
    verifier: str = typer.Option("baseline", help="baseline | hhem"),
    config: Path = typer.Option(None, help="Path to a config YAML"),
):
    """Verify one context + answer pair and print JSON."""
    from halludetect.config import get_device, load_config, set_seed
    from halludetect.verifiers.baseline import build_verifier

    ctx = _read_text(context, context_file, "context")
    ans = _read_text(answer, answer_file, "answer")
    cfg = load_config(config)
    set_seed(cfg.seed)
    v = build_verifier(verifier, cfg, device=get_device(cfg.device))
    typer.echo(json.dumps(v.verify(ctx, ans).to_dict(), indent=2))


@app.command("eval")
def eval_cmd(
    dataset: str = typer.Option("demo", help="demo | ragtruth"),
    max_examples: int = typer.Option(None, help="Optional cap per split (default: use everything)"),
    hhem: bool = typer.Option(True, "--hhem/--no-hhem", help="Also evaluate HHEM"),
    config: Path = typer.Option(None, help="Path to a config YAML"),
):
    """Run the Phase 1 evaluation (baseline NLI vs HHEM)."""
    from halludetect.config import load_config
    from halludetect.eval.run_eval import run_phase1

    run_phase1(load_config(config), dataset=dataset, max_examples=max_examples, use_hhem=hhem)


@app.command("phase3")
def phase3_cmd():
    """Run Phase 3 claim-level verifier training, evaluation, and CIs."""
    from halludetect.eval.run_phase3 import run_evaluation
    run_evaluation()


@app.command("inspect-ragtruth")
def inspect_cmd(
    dir: Path = typer.Option(None, help="Folder with response.jsonl and source_info.jsonl"),
    config: Path = typer.Option(None, help="Path to a config YAML"),
):
    """Print the keys of the first records so you can check them against the RAGTruth README."""
    from halludetect.config import load_config
    from halludetect.data.ragtruth import inspect_ragtruth

    cfg = load_config(config)
    typer.echo(inspect_ragtruth(dir or cfg.data.ragtruth_dir))


@app.command()
def inject(
    max_per_type: int = typer.Option(200, help="Max injected examples per task and error type"),
    out: Path = typer.Option(None, help="Output file (default: results/phase2_injected.jsonl)"),
    allow_regex_entities: bool = typer.Option(False, "--allow-regex-entities",
                                               help="Use a rough regex instead of spaCy for entity swaps"),
    config: Path = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 2: build the injected-error test set from faithful RAGTruth test responses (no GPU needed)."""
    from halludetect.config import load_config
    from halludetect.inject.injectors import run_injection

    run_injection(load_config(config), max_per_type=max_per_type, out_path=out,
                  allow_regex_entities=allow_regex_entities)


@app.command()
def analyze(
    injected: Path = typer.Option(None, help="Injected set (default: results/phase2_injected.jsonl)"),
    predictions: Path = typer.Option(None, help="Phase 1 predictions (default: results/phase1_predictions.csv)"),
    summary: Path = typer.Option(None, help="Phase 1 summary with tuned thresholds"),
    hhem: bool = typer.Option(True, "--hhem/--no-hhem", help="Also score with HHEM"),
    config: Path = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 2: score the injected set with both verifiers and break down real RAGTruth errors by type."""
    from halludetect.analysis.failure import run_failure_analysis
    from halludetect.config import load_config, set_seed

    cfg = load_config(config)
    set_seed(cfg.seed)
    run_failure_analysis(cfg, injected_path=injected, predictions_path=predictions, summary_path=summary,
                         use_hhem=hhem)


@app.command()
def serve(port: int = typer.Option(8000, help="Port for the FastAPI server")):
    """Launch the FastAPI server (app.py in the project root)."""
    import subprocess
    import sys

    from halludetect.config import REPO_ROOT

    app_path = REPO_ROOT / "app.py"
    if not app_path.exists():
        raise typer.BadParameter(f"{app_path} not found")
    subprocess.run([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)], check=True)


if __name__ == "__main__":
    app()