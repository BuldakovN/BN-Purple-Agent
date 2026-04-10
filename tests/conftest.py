import os

import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default=os.environ.get("AGENT_URL", "http://localhost:9009"),
        help="Agent URL (default: AGENT_URL env or http://localhost:9009)",
    )
    parser.addoption(
        "--competition-timeout",
        type=int,
        default=int(os.environ.get("COMPETITION_TIMEOUT", "3600")),
        help="HTTP timeout (seconds) for full competition run (default: COMPETITION_TIMEOUT env or 3600)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.exit(f"Agent at {url} returned status {response.status_code}", returncode=1)
    except Exception as e:
        pytest.exit(f"Could not connect to agent at {url}: {e}", returncode=1)

    return url


@pytest.fixture(scope="session")
def competition_timeout(request) -> int:
    return request.config.getoption("--competition-timeout")
