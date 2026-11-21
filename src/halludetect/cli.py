"""Command line interface. Heavy imports happen inside commands so --help works everywhere."""
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Contextual hallucination detection for RAG answers.")

def _read_text(value: Optional[str], file: Optional[Path], name: str) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if value:
        return value
    raise typer.BadParameter(f"Provide --{name} or --{name}-file")

@app.command()
def verify(
    context: Optional[str] = typer.Option(None, help="Retrieved context text"),
    answer: Optional[str] = typer.Option(None, help="LLM answer to check"),
    context_file: Optional[Path] = typer.Option(None, help="Read context from a file"),
    answer_file: Optional[Path] = typer.Option(None, help="Read answer from a file"),
    verifier: str = typer.Option("baseline", help="baseline | hhem"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
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
    max_examples: Optional[int] = typer.Option(None, help="Optional cap per split (default: use everything)"),
    hhem: bool = typer.Option(True, "--hhem/--no-hhem", help="Also evaluate HHEM"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Run the Phase 1 evaluation (baseline NLI vs HHEM)."""
    from halludetect.config import load_config
    from halludetect.eval.run_eval import run_phase1
    run_phase1(load_config(config), dataset=dataset, max_examples=max_examples, use_hhem=hhem)

@app.command("phase3")
def phase3_cmd(
    max_train: Optional[int] = typer.Option(3000, help="Train responses used to choose the aggregation and thresholds (seeded random subset; 0 = all)"),
    max_test: Optional[int] = typer.Option(None, help="Cap on test responses (default: the whole test split)"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 3: score RAGTruth with the claim-level verifier, tune on train, report on test per task."""
    from halludetect.config import load_config, set_seed
    from halludetect.eval.run_phase3 import run_phase3

    cfg = load_config(config)
    set_seed(cfg.seed)
    run_phase3(cfg, max_train=max_train or None, max_test=max_test)


@app.command("ablate")
def ablate_cmd(
    per_task: int = typer.Option(300, help="Test responses per task used for the ablation"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 3 ablations: top-k, chunk size, retrieval on/off, claim splitting on/off (needs `phase3` first)."""
    from halludetect.config import load_config, set_seed
    from halludetect.eval.run_ablation import run_ablation

    cfg = load_config(config)
    set_seed(cfg.seed)
    run_ablation(cfg, per_task=per_task)


@app.command("phase4")
def phase4_cmd(
    device: Optional[str] = typer.Option(None, help="Device for the latency test: cuda | cpu (default: from the config)"),
    latency_n: int = typer.Option(100, help="Real test responses timed per engine (0 = skip latency)"),
    target_accuracy: float = typer.Option(0.80, help="Accuracy the abstention margin should reach on train"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 4: calibration (ECE), abstention and latency (needs `phase3` first)."""
    from halludetect.config import load_config, set_seed
    from halludetect.eval.run_phase4 import run_phase4

    cfg = load_config(config)
    set_seed(cfg.seed)
    run_phase4(cfg, latency_n=latency_n, target_accuracy=target_accuracy, device=device)


@app.command("inspect-ragtruth")
def inspect_cmd(
    dir: Optional[Path] = typer.Option(None, help="Folder with response.jsonl and source_info.jsonl"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Print the keys of the first records so you can check them against the RAGTruth README."""
    from halludetect.config import load_config
    from halludetect.data.ragtruth import inspect_ragtruth

    cfg = load_config(config)
    typer.echo(inspect_ragtruth(dir or cfg.data.ragtruth_dir))

@app.command()
def inject(
    max_per_type: int = typer.Option(200, help="Max injected examples per task and error type"),
    out: Optional[Path] = typer.Option(None, help="Output file (default: results/phase2_injected.jsonl)"),
    allow_regex_entities: bool = typer.Option(False, "--allow-regex-entities", help="Use a rough regex instead of spaCy for entity swaps", is_flag=True),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 2: build the injected-error test set from faithful RAGTruth test responses."""
    from halludetect.config import load_config
    from halludetect.inject.injectors import run_injection
    run_injection(load_config(config), max_per_type=max_per_type, out_path=out, allow_regex_entities=allow_regex_entities)

@app.command()
def analyze(
    injected: Optional[Path] = typer.Option(None, help="Injected set (default: results/phase2_injected.jsonl)"),
    predictions: Optional[Path] = typer.Option(None, help="Phase 1 predictions (default: results/phase1_predictions.csv)"),
    summary: Optional[Path] = typer.Option(None, help="Phase 1 summary with tuned thresholds"),
    hhem: bool = typer.Option(True, "--hhem/--no-hhem", help="Also score with HHEM"),
    config: Optional[Path] = typer.Option(None, help="Path to a config YAML"),
):
    """Phase 2: score the injected set with both verifiers and break down real RAGTruth errors by type."""
    from halludetect.analysis.failure import run_failure_analysis
    from halludetect.config import load_config, set_seed

    cfg = load_config(config)
    set_seed(cfg.seed)
    run_failure_analysis(cfg, injected_path=injected, predictions_path=predictions, summary_path=summary, use_hhem=hhem)

@app.command()
def serve(port: int = typer.Option(8000, help="Port for the FastAPI server")):
    """Launch the dashboard: FastAPI server (app.py) serving index.html at http://127.0.0.1:8000."""
    import subprocess
    import sys
    from halludetect.config import REPO_ROOT

    app_path = REPO_ROOT / "app.py"
    if not app_path.exists():
        raise typer.BadParameter(f"{app_path} not found")
    subprocess.run([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
                   check=True, cwd=str(REPO_ROOT))

if __name__ == "__main__":
    app()