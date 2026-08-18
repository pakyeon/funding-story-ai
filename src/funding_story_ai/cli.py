from __future__ import annotations

import sys

from . import generation_cli, mcp_client_cli, mcp_server, p1_cli, smoke, validation_cli

COMMANDS = {
    "generate": generation_cli.main,
    "server": mcp_server.main,
    "submit": mcp_client_cli.main,
    "images": p1_cli.main,
    "smoke": smoke.main,
    "validate": validation_cli.main,
}


def _usage() -> str:
    return """Funding Story AI

Usage:
  funding-story generate [options]
  funding-story server [--host 127.0.0.1] [--port 8765]
  funding-story submit --brief-path PATH --live [options]
  funding-story images [options]
  funding-story smoke [--dry-run | --live]
  funding-story validate
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    command = sys.argv.pop(1)
    try:
        handler = COMMANDS[command]
    except KeyError as exc:
        raise SystemExit(f"Unknown command: {command}\n\n{_usage()}") from exc
    handler()


if __name__ == "__main__":
    main()
