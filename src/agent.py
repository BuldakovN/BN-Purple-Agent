import asyncio
import base64
import glob
import io
import logging
import os
import shutil
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
MAX_ITERATIONS = int((os.environ.get("MAX_ITERATIONS") or "").strip() or "30")
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))
PIPELINE_BRANCHES = max(1, int((os.environ.get("PIPELINE_BRANCHES") or "1").strip() or "1"))
PIPELINE_EXPLORATION_ITERATIONS = int(
    (os.environ.get("PIPELINE_EXPLORATION_ITERATIONS") or "0").strip() or "0"
)
PIPELINE_REFINEMENT_ITERATIONS = int(
    (os.environ.get("PIPELINE_REFINEMENT_ITERATIONS") or "0").strip() or "0"
)

# Diverse first approaches when PIPELINE_BRANCHES > 1 (structural pass@k style).
_EXPLORATION_HINTS: list[str] = [
    (
        "Start with the SIMPLEST robust baseline: minimal preprocessing, one strong default model "
        "(e.g. LogisticRegression or GradientBoosting), valid submission first."
    ),
    (
        "Prioritize EDA: inspect missing values, target balance, feature types; then model in a way "
        "that directly addresses what you found."
    ),
    (
        "Prioritize model capacity: stronger boosting (XGBoost/LightGBM) or careful feature interactions "
        "after a quick baseline check."
    ),
]


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
                new_agent_text_message(
                    f"Running ML agent (model={OPENROUTER_MODEL}, "
                    f"pipeline_branches={PIPELINE_BRANCHES})..."
                ),
            )

            loop = asyncio.get_event_loop()
            submission_path = await loop.run_in_executor(
                None,
                self._execute_solve,
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

            self._patch_submission_columns(str(submission_path.parent), submission_path)

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

    @staticmethod
    def _post_pipeline_status(
        loop: asyncio.AbstractEventLoop | None,
        updater: TaskUpdater,
        text: str,
    ) -> None:
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            updater.update_status(TaskState.working, new_agent_text_message(text)),
            loop,
        )

    def _execute_solve(
        self,
        workdir: str,
        instructions: str,
        updater: TaskUpdater,
        loop: asyncio.AbstractEventLoop,
    ) -> Path | None:
        """Single-agent solve, or multi-branch exploration + best selection + refinement."""
        if PIPELINE_BRANCHES <= 1:
            path, _ = self._run_ml_agent(
                workdir,
                instructions,
                updater,
                loop,
                max_iterations=MAX_ITERATIONS,
                exploration_hint=None,
            )
            return path

        explore_n = PIPELINE_EXPLORATION_ITERATIONS
        if explore_n <= 0:
            explore_n = max(8, MAX_ITERATIONS // 3)

        refine_n = PIPELINE_REFINEMENT_ITERATIONS
        if refine_n <= 0:
            refine_n = max(12, MAX_ITERATIONS // 2)

        root = Path(workdir).resolve()
        home_src = root / "home"
        if not home_src.is_dir():
            logger.error("Expected %s after extract", home_src)
            return None

        branches = PIPELINE_BRANCHES
        results: list[tuple[float, int, Path | None]] = []

        for b in range(branches):
            hint = _EXPLORATION_HINTS[b % len(_EXPLORATION_HINTS)]
            branch_root = root / f"_branch_{b}"
            if branch_root.exists():
                shutil.rmtree(branch_root, ignore_errors=True)
            branch_root.mkdir(parents=True)
            shutil.copytree(home_src, branch_root / "home", dirs_exist_ok=True)

            self._post_pipeline_status(
                loop,
                updater,
                f"Pipeline exploration {b + 1}/{branches} ({explore_n} LLM steps, branch hint applied)...",
            )
            sub_path, cv = self._run_ml_agent(
                str(branch_root),
                instructions,
                updater,
                loop,
                max_iterations=explore_n,
                exploration_hint=hint,
            )
            cv_val = cv if cv is not None else float("-inf")
            results.append((cv_val, b, sub_path))
            logger.info(
                "Branch %d done: cv=%s submission=%s",
                b,
                cv,
                sub_path,
            )

        scored: list[tuple[float, int, Path]] = []
        for cv_val, bi, p in results:
            if p is not None and Path(p).is_file():
                scored.append((cv_val, bi, Path(p)))
        if not scored:
            logger.error("No branch produced submission.csv")
            return None
        winner_cv, winner_b, winner_sub = max(scored, key=lambda t: (t[0], -t[1]))

        winner_root = root / f"_branch_{winner_b}"
        cv_display = winner_cv if winner_cv > float("-inf") else "unknown"
        self._post_pipeline_status(
            loop,
            updater,
            f"Pipeline: selected branch {winner_b + 1} (best CV_SCORE≈{cv_display}). "
            f"Refining up to {refine_n} steps...",
        )

        refine_instructions = (
            f"{instructions}\n\n"
            f"[Refinement phase] The same workspace already contains your best exploration attempt "
            f"(approx. CV_SCORE={cv_display}). Improve predictions or model (ensembling, tuning, "
            f"calibration) while keeping validate_submission passing. Print updated CV_SCORE= lines "
            f"when you re-evaluate."
        )
        refined, refine_cv = self._run_ml_agent(
            str(winner_root),
            refine_instructions,
            updater,
            loop,
            max_iterations=refine_n,
            exploration_hint=None,
        )
        final_path = refined if refined is not None and Path(refined).is_file() else winner_sub
        if refine_cv is not None:
            logger.info("Refinement best CV: %s", refine_cv)
        fp = Path(final_path)
        return fp if fp.is_file() else None

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
        *,
        max_iterations: int | None = None,
        exploration_hint: str | None = None,
    ) -> tuple[Path | None, float | None]:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY is not set")
            return None, None

        cap = max_iterations if max_iterations is not None else MAX_ITERATIONS
        agent = MLAgent(
            workdir=workdir,
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            max_iterations=cap,
            code_timeout=CODE_TIMEOUT,
            updater=updater,
            exploration_hint=exploration_hint,
        )
        path = agent.run(instructions, loop=loop)
        return path, agent.last_cv_score
