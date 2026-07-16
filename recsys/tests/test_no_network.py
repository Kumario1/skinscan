"""The recsys engine never touches the network and never imports src/ —
LLM/network code and repo coupling live only under recsys/tools/. This is a
static import lint over every top-level engine module, not a runtime sandbox:
it catches a module that names a forbidden package in an import, and nothing
else — a dynamic import or a shell-out still gets through. The glob scans
recsys/*.py only, so tools/ (which legitimately imports src) and tests/ stay
out of scope."""
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
    # the rest of the repo: the engine couples to it through three file
    # contracts only (ARCHITECTURE.md), so "zero imports from src/" is a
    # boundary this lint enforces rather than a claim the docs merely make
    "src",
}


def _offenders(paths) -> list[str]:
    out = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    out.append(f"{path.name}: {name}")
    return out


def test_no_network_or_src_imports_outside_tools():
    assert _offenders(PACKAGE.glob("*.py")) == []


def test_lint_fires_on_a_src_import(tmp_path):
    """A lint that cannot fail passes for the wrong reason. Prove it live on a
    scratch COPY of a real engine module: the same scan that passes the shipped
    file must flag it once a src import is appended."""
    scratch = tmp_path / "contracts.py"
    scratch.write_text(
        (PACKAGE / "contracts.py").read_text(encoding="utf-8")
        + "\nfrom src.config import load_config\n",
        encoding="utf-8",
    )
    assert _offenders([scratch]) == ["contracts.py: src.config"]
