"""
LangGraph-based ReAct agent that solves Kaggle ML competitions.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from interpreter import Interpreter
from llm import ChatOpenRouter
from ml_helpers import parse_latest_cv_score, strip_warnings, truncate_output, validate_submission_report

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000
RECOVERY_ITERATIONS_MIN = 5


SYSTEM_PROMPT = """\
You are an expert ML engineer solving a Kaggle competition from the MLE-bench benchmark.

TOOLS:
- `run_python`: executes Python in a PERSISTENT session (variables, imports, models survive).
- `validate_submission`: checks ./submission.csv vs sample submission and test row counts (no code).
  Call it BEFORE you finish; if it reports FAIL, fix with run_python and validate again.

ENVIRONMENT:
- Working directory contains all competition files
- Competition description: ./home/data/description.md
- Competition data: ./home/data/ (train.csv, test.csv, etc.)
- List data files first: use os.listdir('./home/data/') to find exact filenames
- Your output target: ./submission.csv (save here when done)

WORKFLOW:
1. List ./home/data/ to find exact filenames, then read description.md
2. Read the sample submission file (e.g. sample_submission.csv or sample_submission_null.csv)
   and print its columns - your submission MUST have the exact same columns (same order)
3. Explore train/test shapes, columns, target distribution (keep prints brief)
4. Build a model with cross-validation or a holdout aligned with the metric in description.md
   (for class imbalance consider stratified splits). Try to achieve THE BEST possible score!
5. After CV / holdout evaluation, print exactly one line to stdout: CV_SCORE=<float>
   (use your CV metric so that higher is better; e.g. negate loss if lower is better)
6. Predict on the test set and save ./submission.csv with ALL columns from the sample submission,
   only replacing the prediction column values (preserve Date, Comment, id, etc.)
7. Call `validate_submission` and fix any FAIL until it passes
8. Reply with plain text (no tool call) saying you are finished only after validate_submission passes

GUIDELINES:
- All file paths must start with ./ (e.g. ./home/data/train.csv)
- Standard ML libraries are available: pandas, numpy, sklearn, xgboost, lightgbm
- Keep it simple and robust; a working baseline beats a crashed complex model
- Print only key metrics (shape, score) - avoid verbose output
- Add `import warnings; warnings.filterwarnings('ignore')` to suppress sklearn warnings
- If code raises an error, read the traceback carefully and fix it in the next call
- On errors, tool output keeps both the start and end of the trace (truncated at {max_output} chars);
  on success, mostly the tail is kept — keep prints concise
- CRITICAL: submission.csv must match the sample submission schema and row count vs test when applicable
- Try to achieve THE BEST possible score!
""".format(max_output=MAX_OUTPUT_CHARS)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int


class MLAgent:
    def __init__(
        self,
        workdir: str | Path,
        api_key: str,
        model: str,
        max_iterations: int = 30,
        code_timeout: int = 600,
        updater: TaskUpdater | None = None,
        exploration_hint: str | None = None,
    ):
        self.workdir = Path(workdir).resolve()
        self.max_iterations = max_iterations
        self._iteration_cap: list[int] = [max_iterations]
        self.updater = updater
        self.exploration_hint = exploration_hint
        self._session_best_cv: float = float("-inf")
        self._loop: asyncio.AbstractEventLoop | None = None
        self.interpreter = Interpreter(workdir=str(self.workdir), timeout=code_timeout)
        self.llm = ChatOpenRouter(model=model, api_key=api_key)
        self._graph = self._build_graph()

    @property
    def last_cv_score(self) -> float | None:
        if self._session_best_cv == float("-inf"):
            return None
        return self._session_best_cv

    def _post_status(self, text: str) -> None:
        if self.updater is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.updater.update_status(TaskState.working, new_agent_text_message(text)),
            self._loop,
        )

    def _build_graph(self):
        interpreter = self.interpreter
        workdir = self.workdir
        session_started = [False]
        ml = self

        @lc_tool
        def run_python(code: str) -> str:
            """Execute Python code in a persistent session and return stdout/stderr."""
            result = interpreter.run(code, reset_session=not session_started[0])
            session_started[0] = True
            clean = strip_warnings(result.output)
            parsed = parse_latest_cv_score(clean)
            if parsed is not None:
                ml._session_best_cv = max(ml._session_best_cv, parsed)
            is_err = result.exc_type is not None or result.timed_out
            return truncate_output(clean, MAX_OUTPUT_CHARS, is_error=is_err)

        @lc_tool
        def validate_submission(_unused: str = "") -> str:
            """Check submission.csv vs sample submission schema, row count vs test, and NA values."""
            return validate_submission_report(workdir)

        llm_with_tools = self.llm.bind_tools([run_python, validate_submission])

        def llm_node(state: AgentState) -> dict:
            cap = self._iteration_cap[0]
            iteration = state["iteration"] + 1
            _log_separator(f"STEP {iteration}/{cap}")
            self._post_status(f"Step {iteration}/{cap}: thinking...")

            response = llm_with_tools.invoke(state["messages"])
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    name = tool_call["name"]
                    if name == "run_python":
                        code_preview = tool_call["args"].get("code", "")[:120].replace("\n", " ")
                        self._post_status(f"Step {iteration}/{cap}: run_python({code_preview}...)")
                    elif name == "validate_submission":
                        self._post_status(f"Step {iteration}/{cap}: validate_submission()")
                    else:
                        self._post_status(f"Step {iteration}/{cap}: {name}(...)")
            else:
                self._post_status(f"Step {iteration}/{cap}: done (no tool call)")

            return {"messages": [response], "iteration": iteration}

        def tool_node(state: AgentState) -> dict:
            last: AIMessage = state["messages"][-1]
            results = []
            for tool_call in last.tool_calls:
                name = tool_call["name"]
                args = tool_call.get("args") or {}
                if name == "run_python":
                    code = args.get("code", "")
                    truncated = run_python.invoke({"code": code})
                    output_preview = truncated[:200].replace("\n", " ")
                    self._post_status(f"Output: {output_preview}")
                    results.append(ToolMessage(content=truncated, tool_call_id=tool_call["id"]))
                elif name == "validate_submission":
                    report = validate_submission.invoke(args)
                    preview = report[:240].replace("\n", " ")
                    self._post_status(f"validate_submission: {preview}")
                    results.append(ToolMessage(content=report, tool_call_id=tool_call["id"]))
                else:
                    results.append(
                        ToolMessage(
                            content=f"Unknown tool: {name}",
                            tool_call_id=tool_call["id"],
                        )
                    )
            return {"messages": results}

        def router(state: AgentState) -> str:
            cap = self._iteration_cap[0]
            if state["iteration"] >= cap:
                logger.warning("Max iterations (%d) reached, stopping.", cap)
                return END
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", router)
        graph.add_edge("tools", "llm")
        return graph.compile()

    def run(self, instructions: str, loop: asyncio.AbstractEventLoop | None = None) -> Path | None:
        self._loop = loop
        file_handler = self._setup_trace_log()
        try:
            _log_separator("ML AGENT START")
            logger.info("workdir : %s", self.workdir)
            logger.info("model : %s", self.llm.model)
            logger.info("max_iter: %d", self.max_iterations)

            self._session_best_cv = float("-inf")
            self._iteration_cap[0] = self.max_iterations
            user_body = instructions
            if self.exploration_hint:
                user_body = (
                    f"[Exploration branch — follow this bias for your first approach]\n"
                    f"{self.exploration_hint}\n\n"
                    f"[Task instructions]\n{instructions}"
                )
            initial_state: AgentState = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_body),
                ],
                "iteration": 0,
            }

            try:
                state = self._graph.invoke(initial_state)
                submission = self.workdir / "submission.csv"
                if not submission.is_file():
                    recovery_cap = max(RECOVERY_ITERATIONS_MIN, self.max_iterations // 4)
                    logger.warning(
                        "No submission.csv after main phase; recovery phase (%d steps).",
                        recovery_cap,
                    )
                    self._iteration_cap[0] = recovery_cap
                    recovery_msg = HumanMessage(
                        content=(
                            "CRITICAL: ./submission.csv is still missing or was not written. "
                            "Use run_python to create it under the workspace root (exact path "
                            "./submission.csv), matching sample_submission columns. "
                            "Then call validate_submission until all checks pass."
                        )
                    )
                    state = {
                        "messages": list(state["messages"]) + [recovery_msg],
                        "iteration": 0,
                    }
                    state = self._graph.invoke(state)
            except Exception as exc:
                logger.error("Agent graph error: %s", exc, exc_info=True)
            finally:
                self.interpreter.cleanup()

            submission = self.workdir / "submission.csv"
            if submission.exists():
                _log_separator("ML AGENT DONE - submission.csv produced")
            else:
                _log_separator("ML AGENT DONE - no submission.csv found")
            return submission if submission.exists() else None
        finally:
            if file_handler is not None:
                logger.removeHandler(file_handler)
                file_handler.close()

    def _setup_trace_log(self) -> logging.FileHandler | None:
        try:
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"agent_{timestamp}.log"

            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.info("Trace log: %s", log_file)
            return handler
        except Exception as exc:
            logger.warning("Could not set up trace log file: %s", exc)
            return None


def _log_separator(title: str) -> None:
    bar = "-" * 70
    logger.info("\n%s\n %s\n%s", bar, title, bar)
