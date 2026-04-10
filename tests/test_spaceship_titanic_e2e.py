"""
End-to-end check: download Spaceship Titanic from Kaggle, pack like MLE-Bench, call a running agent.

Requires:
  - Running agent: `uv run src/server.py` or Docker on the URL from `--agent-url`
  - Agent container/process must have OPENROUTER_API_KEY (and optional OPENROUTER_MODEL)
  - Kaggle: KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json
  - Competition rules accepted for spaceship-titanic on kaggle.com

Run (example):
  uv sync --extra test
  uv run pytest tests/test_spaceship_titanic_e2e.py -m integration --agent-url http://localhost:9009 -v
"""

from __future__ import annotations

import base64
import io
import os
import tarfile
import zipfile
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TaskState, TextPart

COMPETITION_ID = "spaceship-titanic"

SPACESHIP_DESCRIPTION = """# Spaceship Titanic

Binary classification: predict whether the passenger was **Transported** to another dimension.

Typical files: `train.csv`, `test.csv`, `sample_submission.csv`.

Use the sample submission column layout exactly for `submission.csv`.
"""


def _kaggle_configured() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return Path.home().joinpath(".kaggle", "kaggle.json").is_file()


def _download_spaceship_titanic(dest: Path) -> Path:
    pytest.importorskip("kaggle")
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION_ID, path=str(dest), quiet=True)

    for zpath in dest.glob("*.zip"):
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)

    if (dest / "train.csv").is_file():
        return dest
    for sub in dest.iterdir():
        if sub.is_dir() and (sub / "train.csv").is_file():
            return sub

    raise AssertionError(
        f"Could not find train.csv after Kaggle download under {dest} "
        f"(contents: {list(dest.iterdir())})"
    )


def _build_competition_tar_gz(data_dir: Path) -> bytes:
    desc = data_dir / "description.md"
    if not desc.is_file():
        desc.write_text(SPACESHIP_DESCRIPTION, encoding="utf-8")

    required = ("train.csv", "test.csv", "sample_submission.csv", "description.md")
    for name in required:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {data_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in required:
            tar.add(data_dir / name, arcname=f"home/data/{name}")
    return buf.getvalue()


def _submission_csv_from_task(task) -> bytes | None:
    if not getattr(task, "artifacts", None):
        return None
    for art in task.artifacts:
        for part in art.parts:
            root = part.root
            if isinstance(root, FilePart):
                f = root.file
                if isinstance(f, FileWithBytes) and f.name == "submission.csv":
                    return base64.b64decode(f.bytes)
    return None


async def _send_competition_to_agent(
    *,
    agent_url: str,
    tar_gz: bytes,
    instructions: str,
    timeout_sec: int,
) -> tuple[bytes | None, TaskState | None, str]:
    """
    Returns (submission_csv_bytes | None, final_task_state | None, status_hint).
    """
    timeout = httpx.Timeout(timeout_sec, connect=60.0, pool=60.0)
    last_task = None
    last_state: TaskState | None = None
    hint = ""

    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[
                Part(TextPart(kind="text", text=instructions)),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(tar_gz).decode("ascii"),
                            name="competition.tar.gz",
                            mime_type="application/gzip",
                        )
                    )
                ),
            ],
            message_id=uuid4().hex,
            context_id=None,
        )

        async for event in client.send_message(msg):
            match event:
                case (task, _):
                    last_task = task
                    last_state = task.status.state
                    if task.status.message and task.status.message.parts:
                        hint = " ".join(
                            p.root.text
                            for p in task.status.message.parts
                            if hasattr(p.root, "text")
                        )
                    sub = _submission_csv_from_task(task)
                    if sub is not None:
                        return sub, last_state, hint
                case Message():
                    pass
                case _:
                    pass

    if last_task is not None:
        sub = _submission_csv_from_task(last_task)
        return sub, last_state, hint

    return None, last_state, hint or "no task events received"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spaceship_titanic_kaggle_download_and_agent_prediction(
    agent,
    competition_timeout,
    tmp_path: Path,
):
    if not _kaggle_configured():
        pytest.skip(
            "Kaggle not configured: set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place credentials in ~/.kaggle/kaggle.json"
        )

    raw = tmp_path / "kaggle_download"
    data_dir = _download_spaceship_titanic(raw)
    tar_gz = _build_competition_tar_gz(data_dir)

    instructions = (
        "Solve the Spaceship Titanic competition from the bundled data under ./home/data/. "
        "Read description.md, follow the sample_submission format, and write ./submission.csv."
    )

    submission, state, hint = await _send_competition_to_agent(
        agent_url=agent,
        tar_gz=tar_gz,
        instructions=instructions,
        timeout_sec=competition_timeout,
    )

    assert state != TaskState.failed, f"Agent task failed (hint: {hint!r})"
    assert submission is not None, (
        f"No submission.csv artifact (state={state!r}, hint={hint!r}). "
        "Check agent logs and OPENROUTER_API_KEY on the server."
    )

    text = submission.decode("utf-8", errors="replace")
    assert "PassengerId" in text, f"Unexpected CSV header: {text[:500]!r}"
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    assert len(lines) >= 2, "Expected header + at least one prediction row"
