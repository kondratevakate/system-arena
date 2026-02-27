"""Command-line interface for System Arena."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="arena",
    help="System Arena: Benchmark framework for case-based agents",
    no_args_is_help=True,
)
console = Console()


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to workflow directory or manifest file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Validate a workflow manifest and dataset."""
    from arena.datasets.validator import validate_directory, validate_manifest
    from arena.datasets.manifest import load_manifest

    if path.is_dir():
        console.print(f"Validating directory: [bold]{path}[/bold]")
        result = validate_directory(path)
    elif path.suffix in (".yaml", ".yml"):
        console.print(f"Validating manifest: [bold]{path}[/bold]")
        manifest = load_manifest(path)
        result = validate_manifest(manifest)
    else:
        console.print(f"[red]Error:[/red] Unknown file type: {path.suffix}")
        raise typer.Exit(1)

    if result.valid:
        console.print("[green]✓ Validation passed[/green]")
    else:
        console.print("[red]✗ Validation failed[/red]")

    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors:
            console.print(f"  • {error.field}: {error.message}")

    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning.field}: {warning.message}")

    if not result.valid:
        raise typer.Exit(1)


@app.command()
def replay(
    path: Path = typer.Argument(..., help="Path to workflow directory"),
    policy: str = typer.Option("threshold", "--policy", "-p", help="Policy to use (random, threshold, rule)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for traces"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max decision points to process"),
) -> None:
    """Run a benchmark replay on historical data."""
    from arena.datasets.loader import load_from_directory
    from arena.policy.baselines import RandomPolicy, ThresholdPolicy, create_followup_policy
    from arena.replay.runner import BenchmarkConfig, BenchmarkRunner
    from arena.eval.report import generate_report

    console.print(f"Loading workflow from: [bold]{path}[/bold]")

    try:
        manifest, event_log = load_from_directory(path)
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(event_log)} events across {len(event_log.case_ids)} cases")

    # Select policy
    if policy == "random":
        selected_policy = RandomPolicy(name="random")
    elif policy == "threshold":
        from arena.policy.baselines import ThresholdPolicy, ThresholdConfig
        selected_policy = ThresholdPolicy(
            thresholds=[
                ThresholdConfig(
                    field="days_since_last_contact",
                    threshold=3.0,
                    action_above="send_reminder",
                    action_below="wait",
                )
            ],
            name="threshold",
        )
    elif policy == "rule":
        selected_policy = create_followup_policy(name="rule")
    else:
        console.print(f"[red]Unknown policy:[/red] {policy}")
        raise typer.Exit(1)

    console.print(f"  Using policy: [bold]{selected_policy.name}[/bold]")

    # Configure and run benchmark
    config = BenchmarkConfig(
        decision_rules=manifest.get_decision_rules(),
        constraints=manifest.get_constraints(),
        max_decision_points=limit,
        name=manifest.name,
    )

    runner = BenchmarkRunner(config)

    console.print("\nRunning replay...")
    result = runner.run(event_log, selected_policy)

    # Show results
    console.print(f"\n[green]✓ Replay complete[/green]")
    console.print(f"  Decision points: {result.total_decisions}")
    console.print(f"  Violations: {result.total_violations}")
    console.print(f"  Duration: {result.duration_ms:.2f}ms")

    # Show action distribution
    if result.traces.action_distribution:
        console.print("\n[bold]Action Distribution:[/bold]")
        table = Table(show_header=True)
        table.add_column("Action")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")

        total = sum(result.traces.action_distribution.values())
        for action, count in sorted(
            result.traces.action_distribution.items(),
            key=lambda x: -x[1]
        ):
            pct = count / total * 100 if total > 0 else 0
            table.add_row(action, str(count), f"{pct:.1f}%")

        console.print(table)

    # Save output if requested
    if output:
        result.traces.to_jsonl(str(output))
        console.print(f"\nTraces saved to: [bold]{output}[/bold]")


@app.command()
def report(
    traces_path: Path = typer.Argument(..., help="Path to traces JSONL file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format (text, markdown, json)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
) -> None:
    """Generate a report from benchmark traces."""
    # Note: Full implementation would load traces and generate report
    console.print(f"Generating report from: [bold]{traces_path}[/bold]")
    console.print("[yellow]Note: Full report generation from saved traces not yet implemented[/yellow]")


@app.command()
def info(
    path: Path = typer.Argument(..., help="Path to workflow directory"),
) -> None:
    """Show information about a workflow."""
    from arena.datasets.manifest import load_manifest

    manifest_path = path / "manifest.yaml"

    if not manifest_path.exists():
        console.print(f"[red]Manifest not found:[/red] {manifest_path}")
        raise typer.Exit(1)

    manifest = load_manifest(manifest_path)

    console.print(f"\n[bold]Workflow: {manifest.name}[/bold]")
    console.print(f"Version: {manifest.version}")

    if manifest.description:
        console.print(f"Description: {manifest.description}")

    console.print(f"\n[bold]Case Configuration:[/bold]")
    console.print(f"  ID Field: {manifest.case.id_field}")
    console.print(f"  Timestamp Field: {manifest.case.timestamp_field}")

    console.print(f"\n[bold]Decision Points ({len(manifest.decision_points)}):[/bold]")
    for dp in manifest.decision_points:
        console.print(f"  • {dp.name}")
        console.print(f"    Trigger: {dp.trigger}")
        if dp.condition:
            console.print(f"    Condition: {dp.condition}")

    console.print(f"\n[bold]Allowed Actions ({len(manifest.allowed_actions)}):[/bold]")
    for action in manifest.allowed_actions:
        params = f" ({', '.join(action.params)})" if action.params else ""
        console.print(f"  • {action.action}{params}")

    console.print(f"\n[bold]Constraints ({len(manifest.constraints)}):[/bold]")
    for constraint in manifest.constraints:
        console.print(f"  • {constraint.name} [{constraint.severity}]")

    console.print(f"\n[bold]Outcomes:[/bold]")
    if manifest.outcomes.positive:
        console.print(f"  Positive: {', '.join(manifest.outcomes.positive)}")
    if manifest.outcomes.negative:
        console.print(f"  Negative: {', '.join(manifest.outcomes.negative)}")
    if manifest.outcomes.neutral:
        console.print(f"  Neutral: {', '.join(manifest.outcomes.neutral)}")


@app.command()
def version() -> None:
    """Show version information."""
    from arena import __version__

    console.print(f"System Arena v{__version__}")


if __name__ == "__main__":
    app()
