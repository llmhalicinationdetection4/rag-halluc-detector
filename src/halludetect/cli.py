from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import typer
from halludetect.config import load_config, resolve_device
from halludetect.data.demo import load_demo_examples
from halludetect.eval.run_eval import execute_evaluation
from halludetect.inject.perturb import create_perturbed_dataset
from halludetect.models.nli import NLIModel
from halludetect.text.retrieval import ChunkRetriever
from halludetect.verifiers.claim_level import ClaimLevelVerifier

app = typer.Typer(help="Automated Contextual Hallucination Detector CLI")


@app.command()
def verify(
    context: str = typer.Option(None, "--context", "-c", help="Retrieved context string"),
    answer: str = typer.Option(None, "--answer", "-a", help="Generated answer string"),
    config: str = typer.Option("config/default.yaml", "--config", help="Path to config YAML"),
) -> None:
    """Verifies a single (context, answer) pair. Uses demo example 1 if omitted."""
    cfg = load_config(config)
    device = resolve_device(cfg["models"]["device"])

    if context is None or answer is None:
        demo_examples = load_demo_examples(cfg["paths"]["demo_data"])
        context = demo_examples[0].context
        answer = demo_examples[0].answer
        typer.echo("Using default demo record for verification...")

    nli = NLIModel(
        model_name=cfg["models"]["nli_model_name"],
        fallback_name=cfg["models"]["nli_fallback_name"],
        device=device,
    )
    retriever = ChunkRetriever(model_name=cfg["models"]["embedding_model_name"], device=device)

    verifier = ClaimLevelVerifier(
        nli_model=nli,
        retriever=retriever,
        chunk_tokens=cfg["chunking"]["chunk_tokens"],
        stride=cfg["chunking"]["stride"],
        top_k=cfg["retrieval"]["top_k"],
        tau_support=cfg["thresholds"]["tau_support"],
        tau_contra=cfg["thresholds"]["tau_contra"],
        tau_answer=cfg["thresholds"]["tau_answer"],
    )

    result = verifier.verify(context, answer)
    output = {
        "score": result.score,
        "label": result.label,
        "hallucinated": result.hallucinated,
        "latency_ms": result.latency_ms,
        "claims": [
            {
                "claim": c.claim,
                "verdict": c.verdict,
                "p_entail": c.p_entail,
                "p_neutral": c.p_neutral,
                "p_contra": c.p_contra,
                "best_chunks": c.best_chunks,
            }
            for c in result.claims
        ],
    }
    typer.echo(json.dumps(output, indent=2))


@app.command()
def eval(
    dataset: str = typer.Option("demo", "--dataset", "-d", help="'demo' or 'ragtruth'"),
    config: str = typer.Option("config/default.yaml", "--config", help="Path to config YAML"),
) -> None:
    """Runs Phase 1-4 full benchmark evaluation and writes reports into results/."""
    execute_evaluation(dataset_source=dataset, config_path=config)


@app.command()
def inject(
    config: str = typer.Option("config/default.yaml", "--config", help="Path to config YAML"),
) -> None:
    """Generates perturbed hallucinated samples for failure mode evaluation."""
    cfg = load_config(config)
    demo_examples = load_demo_examples(cfg["paths"]["demo_data"])
    create_perturbed_dataset(demo_examples, cfg["paths"]["injected_data"], seed=cfg["seed"])
    typer.echo(f"Perturbed samples saved to {cfg['paths']['injected_data']}")


@app.command()
def analyze(
    config: str = typer.Option("config/default.yaml", "--config", help="Path to config YAML"),
) -> None:
    """Analyzes failure buckets from injected data."""
    execute_evaluation(dataset_source="demo", config_path=config)
    typer.echo("Analysis complete. Check results/*.csv and failure_report.json")


@app.command()
def serve() -> None:
    """Launches the Streamlit interactive UI application."""
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    app()