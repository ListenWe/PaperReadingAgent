from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paper_reading_agent.agent.core import PaperReadingAgent
from paper_reading_agent.config import AppConfig
from paper_reading_agent.cli.renderer import (
    console,
    render_error,
    render_help,
    render_markdown,
    render_section_list,
    render_status,
    render_success,
    render_welcome,
)


class CLIApp:
    def __init__(self) -> None:
        self._config = AppConfig.from_env()
        self._agent = PaperReadingAgent(self._config)

    def run(self) -> None:
        render_welcome()

        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import InMemoryHistory

            commands = ["/load", "/report", "/ask", "/info", "/history", "/reset", "/help", "/quit"]
            completer = WordCompleter(commands, ignore_case=True)
            session = PromptSession(history=InMemoryHistory())

            while True:
                try:
                    user_input = session.prompt("\n> ", completer=completer).strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""

                    if command == "/quit":
                        console.print("Goodbye!")
                        break
                    elif command == "/help":
                        render_help()
                    elif command == "/load":
                        self._handle_load(args)
                    elif command == "/report":
                        self._handle_report()
                    elif command == "/ask":
                        self._handle_ask(args)
                    elif command == "/info":
                        self._handle_info()
                    elif command == "/history":
                        self._handle_history()
                    elif command == "/reset":
                        self._agent.reset_conversation()
                        render_success("Conversation history cleared.")
                    else:
                        render_error(f"Unknown command: {command}")
                else:
                    # Treat as a question
                    self._handle_ask(user_input)

        except ImportError:
            # Fallback: simple input loop without prompt_toolkit
            console.print("[yellow]prompt-toolkit not available, using simple input mode.[/yellow]")
            self._simple_loop()

    def _simple_loop(self) -> None:
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "/quit":
                    console.print("Goodbye!")
                    break
                elif command == "/help":
                    render_help()
                elif command == "/load":
                    self._handle_load(args)
                elif command == "/report":
                    self._handle_report()
                elif command == "/ask":
                    self._handle_ask(args)
                elif command == "/info":
                    self._handle_info()
                elif command == "/history":
                    self._handle_history()
                elif command == "/reset":
                    self._agent.reset_conversation()
                    render_success("Conversation history cleared.")
                else:
                    render_error(f"Unknown command: {command}")
            else:
                self._handle_ask(user_input)

    def _handle_load(self, path_str: str) -> None:
        if not path_str:
            render_error("Usage: /load <path-to-pdf>")
            return
        path = Path(path_str.strip())
        if not path.exists():
            render_error(f"File not found: {path}")
            return
        if path.suffix.lower() != ".pdf":
            render_error("Only PDF files are supported.")
            return

        render_status(f"Loading paper: {path.name} ...")
        try:
            paper = self._agent.load_paper(path)
            render_success(f"Loaded: {paper.title or path.name}")
            render_success(f"Pages: {paper.metadata.get('page_count', '?')}, Sections: {len(paper.sections)}")
            if paper.sections:
                render_section_list(paper.title, paper.sections)
            render_success("Paper indexed. Use /report to generate interpretation or /ask to ask questions.")
        except Exception as e:
            render_error(f"Failed to load paper: {e}")

    def _handle_report(self) -> None:
        if self._agent.paper is None:
            render_error("No paper loaded. Use /load <path> first.")
            return

        render_status("Generating interpretation report (this may take a minute)...")
        try:
            report = ""
            for chunk in self._agent.generate_report_stream():
                report += chunk
                console.print(chunk, end="")
            console.print("\n")
            render_success("Report generation complete.")
        except Exception as e:
            render_error(f"Failed to generate report: {e}")

    def _handle_ask(self, question: str) -> None:
        if not question:
            render_error("Usage: /ask <your question>")
            return
        if self._agent.paper is None:
            render_error("No paper loaded. Use /load <path> first.")
            return

        render_status("Thinking...")
        try:
            response = ""
            for chunk in self._agent.ask_question_stream(question):
                response += chunk
                console.print(chunk, end="")
            console.print("\n")
        except Exception as e:
            render_error(f"Failed to answer: {e}")

    def _handle_info(self) -> None:
        paper = self._agent.paper
        if paper is None:
            render_error("No paper loaded.")
            return

        console.print(f"\n[bold]Title:[/bold] {paper.title or 'Unknown'}")
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        console.print(f"[bold]Authors:[/bold] {authors}")
        console.print(f"[bold]Pages:[/bold] {paper.metadata.get('page_count', '?')}")
        console.print(f"[bold]Sections:[/bold] {len(paper.sections)}")
        console.print(f"[bold]Abstract:[/bold] {paper.abstract[:500]}...")
        if paper.sections:
            render_section_list(paper.title, paper.sections)

    def _handle_history(self) -> None:
        history = self._agent.history
        if not history:
            console.print("[dim]No conversation history yet.[/dim]")
            return
        for msg in history:
            role_color = "cyan" if msg.role == "user" else "green"
            console.print(f"[{role_color}]{msg.role}:[/{role_color}] {msg.content[:200]}")
            if len(msg.content) > 200:
                console.print("[dim]... (truncated)[/dim]")
            console.print()
