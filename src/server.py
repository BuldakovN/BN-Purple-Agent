import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor

load_dotenv(Path(__file__).parent.parent / ".env")


def _agent_card_url(host: str, port: int, card_url_arg: str | None) -> str:
    """
    URL в agent card должен быть достижим для клиентов A2A после fetch карточки.
    При bind на 0.0.0.0 нельзя подставлять 0.0.0.0 в URL — используйте AGENT_CARD_URL
    (например http://bn-purple-agent:9009/ в Docker Compose или http://localhost:9009/ с хоста).
    """
    if card_url_arg:
        u = card_url_arg.strip()
        return u if u.endswith("/") else u + "/"
    if host in ("0.0.0.0", "::"):
        env_u = os.environ.get("AGENT_CARD_URL", "").strip()
        if env_u:
            return env_u if env_u.endswith("/") else env_u + "/"
        return f"http://127.0.0.1:{port}/"
    return f"http://{host}:{port}/"


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("ml_agent").setLevel(logging.INFO)
logging.getLogger("agent").setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server",
    )
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument(
        "--card-url",
        type=str,
        help="URL in agent card (overrides AGENT_CARD_URL env when binding to 0.0.0.0)",
    )
    args = parser.parse_args()

    skill = AgentSkill(
        id="mle-bench-solver",
        name="MLE-Bench Competition Solver",
        description=(
            "Solves Kaggle ML competitions from the MLE-Bench benchmark. "
            "Receives competition data as a tar.gz archive and returns a submission.csv artifact."
        ),
        tags=["mle-bench", "kaggle", "machine-learning"],
        examples=[],
    )

    agent_card = AgentCard(
        name="MLE-Bench Purple Agent",
        description="An A2A-compatible agent that solves Kaggle competitions from the MLE-Bench benchmark.",
        url=_agent_card_url(args.host, args.port, args.card_url),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
