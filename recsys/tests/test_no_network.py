"""The recsys engine never touches the network — LLM/network code is allowed
only under recsys/tools/. This is a static import lint over every non-tools
module, not a runtime sandbox: it catches a module that names a network library
in an import, and nothing else — a dynamic import or a shell-out still gets
through."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1]
FORBIDDEN = {
    # stdlib transports
    "urllib", "http", "socket", "ssl", "ftplib", "smtplib", "telnetlib",
    # third-party clients
    "urllib3", "requests", "httpx", "aiohttp", "pycurl", "websockets", "grpc", "boto3",
    # LLM SDKs
    "openai", "anthropic",
}


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
