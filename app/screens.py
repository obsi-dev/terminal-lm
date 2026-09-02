from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Button


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, command: str, sandbox_output: str):
        super().__init__()
        self.command = command
        self.sandbox_output = sandbox_output

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm_dialog"):
            yield Static(f"[bold]Command:[/] {self.command}")
            yield Static(f"[dim]Sandbox output:[/]\n{self.sandbox_output}")
            yield Button("Run for real", id="confirm", variant="success")
            yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")
