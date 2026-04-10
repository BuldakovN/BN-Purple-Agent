import argparse
import logging
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
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
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
        url=args.card_url or f"http://{args.host}:{args.port}/",
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
