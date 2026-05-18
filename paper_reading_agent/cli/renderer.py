from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def render_markdown(text: str) -> None:
    console.print(Markdown(text))


def render_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def render_error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def render_status(message: str) -> None:
    console.print(f"[yellow]⏳[/yellow] {message}")


def render_section_list(paper_title: str, sections: list) -> None:
    table = Table(title=f"Paper: {paper_title}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Section", style="cyan")
    table.add_column("Pages", style="dim")

    for i, section in enumerate(sections, 1):
        pages = f"{section.start_page + 1}-{section.end_page + 1}"
        table.add_row(str(i), section.title, pages)

    console.print(table)


def render_help() -> None:
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [yellow]/load <path>[/yellow]    Load a PDF paper
  [yellow]/report[/yellow]         Generate the interpretation report
  [yellow]/ask <question>[/yellow] Ask a question about the paper
  [yellow]/info[/yellow]           Show paper metadata and section list
  [yellow]/history[/yellow]        Show conversation history
  [yellow]/reset[/yellow]          Reset conversation (keep paper loaded)
  [yellow]/help[/yellow]           Show this help message
  [yellow]/quit[/yellow]           Exit the application
"""
    console.print(Panel(help_text, title="Help"))


def render_welcome() -> None:
    welcome = """
[bold cyan]Paper Reading Agent[/bold cyan]
AI-powered academic paper reader and analyst.

Type [yellow]/help[/yellow] for available commands.
Type [yellow]/load <path>[/yellow] to start reading a paper.
"""
    console.print(welcome)
