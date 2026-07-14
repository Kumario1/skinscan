"""The recsys engine never touches the network — LLM/network code is allowed
only under recsys/tools/. Enforced by scanning imports of every non-tools
module."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1]
FORBIDDEN = {"urllib", "http", "socket", "ssl", "requests", "httpx", "aiohttp", "openai", "anthropic"}


def test_no_network_imports_outside_tools():
    offenders = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders
