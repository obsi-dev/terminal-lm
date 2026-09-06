import os
import click
import readline
import typer
from app.inference import InferenceEngine
from sandbox.executor import run_in_sandbox

app = typer.Typer()
engine = InferenceEngine()


def process_query(query: str) -> None:
    typer.echo("Loading model..." if engine.model is None else "")
    if engine.model is None:
        engine.load()

    command = engine.generate(query)
    # typer.echo(f"Suggested: {command}")
    typer.echo(
        typer.style("> ", fg=typer.colors.GREEN, bold=True)
        + typer.style(command, fg=typer.colors.CYAN)
    )

    result = run_in_sandbox(command, working_dir=os.getcwd())

    # typer.echo(f"Sandbox result (exit code {result.exit_code})")
    # typer.echo(result.stdout + result.stderr)

    if result.exit_code != 0 or result.timed_out:
        typer.secho(
            f"This command failed in the sandbox (exit code {result.exit_code})",
            fg=typer.colors.YELLOW,
        )
        if result.stderr.strip():
            print(result.stderr.strip())

    if typer.confirm("Run this command?"):
        import subprocess

        final_result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=os.getcwd()
        )
        typer.echo(final_result.stdout + final_result.stderr)
    else:
        typer.echo("Cancelled.")


@app.command()
def run(query: str) -> None:
    process_query(query)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    typer.echo("tlm interactive mode. Ctrl + C to exit.")
    engine.load()
    while True:
        try:
            query = typer.prompt(">")
        except (KeyboardInterrupt, EOFError):
            typer.echo("Exiting.")
            raise typer.Exit()

        if query.strip().lower() == "clear":
            click.clear()
            continue

        if query.strip().lower() in ("exit", "quit"):
            typer.echo("Exiting.")
            raise typer.Exit()

        process_query(query)


if __name__ == "__main__":
    app()
