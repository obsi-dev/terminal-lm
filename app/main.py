from textual import work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Input, RichLog, Header, Footer

from app.inference import InferenceEngine
from app.screens import ConfirmScreen
from sandbox.executor import run_in_sandbox


class TerminalLMApp(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(RichLog(id="log", wrap=True))
        yield Input(placeholder="Ask me to do something...", id="query_input")
        yield Footer()

    def on_mount(self) -> None:
        self.engine = InferenceEngine()
        self.query_one("#query_input", Input).disabled = True
        self.load_model()

    @work(thread=True)
    def load_model(self) -> None:
        self.engine.load()
        self.call_from_thread(self.on_model_loaded)

    def on_model_loaded(self) -> None:
        self.query_one("#log", RichLog).write("[green]Model loaded. Ready.[/]")
        self.query_one("#query_input", Input).disabled = False

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value
        event.input.clear()
        log = self.query_one("#log", RichLog)
        log.write(f"[bold cyan]You:[/] {query}")
        self.run_pipeline(query)

    @work(thread=True)
    def run_pipeline(self, query: str) -> None:
        command = self.engine.generate(query)
        self.call_from_thread(self.on_command_generated, command)

    def on_command_generated(self, command: str) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"[bold yellow]Model Suggests:[/] {command}")
        self.run_sandbox(command)

    @work(thread=True)
    def run_sandbox(self, command: str) -> None:
        result = run_in_sandbox(command)
        self.call_from_thread(self.on_sandbox_done, command, result)

    def on_sandbox_done(self, command: str, result) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"[dim]Sandbox exit code: {result.exit_code}[/]")
        log.write(f"[dim]{result.stdout}{result.stderr}[/]")
        self.confirm_and_run(command, result)

    @work
    async def confirm_and_run(self, command: str, result) -> None:
        confirmed = await self.push_screen_wait(
            ConfirmScreen(command, result.stdout + result.stderr)
        )
        log = self.query_one("#log", RichLog)
        if confirmed:
            log.write(f"[bold red]Running for real:[/] {command}")
            # TODO: handle actual command execution
        else:
            log.write("[dim]Cancelled.[/]")


if __name__ == "__main__":
    TerminalLMApp().run()
