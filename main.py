"""
main.py – Entry point. Initialises DB then launches Streamlit.

Usage:
  python main.py
  # or directly:
  streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path


async def startup():
    """Initialize database tables."""
    from db.database import init_db
    await init_db()
    print("[OK] Database initialized.")


def main():
    print("=" * 50)
    print(" AI Investment Assistant - India")
    print("=" * 50)

    # Initialize DB
    asyncio.run(startup())

    # Launch Streamlit
    streamlit_app = Path(__file__).parent / "ui" / "streamlit_app.py"
    print(f"[*] Starting Streamlit UI at http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app),
         "--server.port", "8501", "--server.headless", "true"],
        check=True,
    )


if __name__ == "__main__":
    main()
