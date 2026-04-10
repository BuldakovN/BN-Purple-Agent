import asyncio
import base64
import glob
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from ml_agent import MLAgent

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-397b-a17b")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Extracting competition data..."),
        )

        competition_tar, instructions = self._parse_message(message)
        if competition_tar is None:
            await updater.failed(new_agent_text_message("No competition tar.gz received"))
            return

        with tempfile.TemporaryDirectory() as workdir:
            self._extract_tar(competition_tar, workdir)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Running ML agent (model={OPENROUTER_MODEL})..."),
            )

            loop = asyncio.get_event_loop()
            submission_path = await loop.run_in_executor(
                None,
                self._run_ml_agent,
                workdir,
                instructions,
                updater,
                loop,
            )

            if submission_path is None:
                await updater.failed(
                    new_agent_text_message("Agent did not produce submission.csv")
                )
                return

            self._patch_submission_columns(workdir, submission_path)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Submitting results..."),
            )

            submission_bytes = Path(submission_path).read_bytes()
            await updater.add_artifact(
                parts=[
                    Part(
                        root=FilePart(
                            file=FileWithBytes(
                                bytes=base64.b64encode(submission_bytes).decode("ascii"),
                                name="submission.csv",
                                mime_type="text/csv",
                            )
                        )
                    )
                ],
                name="submission",
            )

    def _parse_message(self, message: Message) -> tuple[bytes | None, str]:
        competition_tar: bytes | None = None
        instructions = get_message_text(message) or ""

        for part in message.parts:
            if isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    competition_tar = base64.b64decode(file_data.bytes)
                    break

        return competition_tar, instructions

    def _extract_tar(self, competition_tar: bytes, workdir: str) -> None:
        with tarfile.open(fileobj=io.BytesIO(competition_tar), mode="r:gz") as tar:
            tar.extractall(workdir, filter="data")

    def _patch_submission_columns(self, workdir: str, submission_path: Path) -> None:
        try:
            import pandas as pd

            data_dir = Path(workdir) / "home" / "data"
            sample_paths = sorted(glob.glob(str(data_dir / "sample_submission*.csv")))
            if not sample_paths:
                return

            sample = pd.read_csv(sample_paths[0])
            submission = pd.read_csv(submission_path)

            missing_cols = [col for col in sample.columns if col not in submission.columns]
            if not missing_cols:
                return

            logger.warning(
                "submission.csv is missing columns %s; backfilling from sample/test",
                missing_cols,
            )

            test_paths = sorted(glob.glob(str(data_dir / "test.csv")))
            if test_paths:
                test = pd.read_csv(test_paths[0])
                for col in missing_cols:
                    if col in test.columns:
                        submission[col] = test[col].values
                    else:
                        submission[col] = sample[col].values
            else:
                for col in missing_cols:
                    submission[col] = sample[col].values

            ordered_cols = [col for col in sample.columns if col in submission.columns]
            submission = submission[ordered_cols]
            submission.to_csv(submission_path, index=False)
            logger.info("Patched submission.csv columns: %s", submission.columns.tolist())
        except Exception as exc:
            logger.warning("Could not patch submission columns: %s", exc)

    def _run_ml_agent(
        self,
        workdir: str,
        instructions: str,
        updater: TaskUpdater,
        loop: asyncio.AbstractEventLoop,
    ) -> Path | None:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY is not set")
            return None

        agent = MLAgent(
            workdir=workdir,
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            max_iterations=MAX_ITERATIONS,
            code_timeout=CODE_TIMEOUT,
            updater=updater,
        )
        return agent.run(instructions, loop=loop)
