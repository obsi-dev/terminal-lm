from textual import work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Input, RichLog, Header, Footer


class TerminalLMApp(App):
    CSS_PATH = "styles.css"
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(RichLog(id="log", wrap=True))
        yield Input(placeholder="Ask me to do something...", id="query_input")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value
        event.input.clear()
        log = self.query_one("#log", RichLog)
        log.write(f"[bold cyan]You:[/] {query}")
        log.write("Thinking...")

        self.run_inference(query)

    @work(thread=True)
    def run_inference(self, query: str) -> None:
        command = self.model.generate(query)
        self.call_from_thread(self.show_command, command)

    def show_command(self, command: str) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"Model: {command}")


if __name__ == "__main__":
    TerminalLMApp.run()
