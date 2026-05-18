"""Paper Reading Agent - Entry point.

Usage:
    # CLI mode (interactive REPL)
    python -m paper_reading_agent.main --mode cli

    # Web UI mode (Streamlit)
    python -m paper_reading_agent.main --mode web
    # Or directly:
    streamlit run paper_reading_agent/web/app.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Reading Agent - AI-powered academic paper reader"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Run mode: cli (interactive terminal) or web (Streamlit UI)",
    )
    args = parser.parse_args()

    if args.mode == "web":
        import subprocess
        app_path = Path(__file__).parent / "web" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    else:
        from paper_reading_agent.cli.app import CLIApp
        app = CLIApp()
        app.run()


if __name__ == "__main__":
    main()
