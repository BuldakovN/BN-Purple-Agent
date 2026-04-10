"""
LangGraph-based ReAct agent that solves Kaggle ML competitions.
"""

import asyncio
import logging
import re
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

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000

SYSTEM_PROMPT = """\
You are an expert ML engineer solving a Kaggle competition from the MLE-bench benchmark.

You have a `run_python` tool that executes Python code in a PERSISTENT interpreter session.
This means variables, imports, and trained models are preserved across calls.

ENVIRONMENT:
- Working directory contains all competition files
- Competition description: ./home/data/description.md
- Competition data: ./home/data/ (train.csv, test.csv, etc.)
- List data files first: use os.listdir('./home/data/') to find exact filenames
- Your output target: ./submission.csv (save here when done)

WORKFLOW:
1. List ./home/data/ to find exact filenames, then read description.md
2. Read the sample submission file (e.g. sample_submission.csv or sample_submission_null.csv)
   and print its columns - your submission MUST have the exact same columns
3. Explore train/test shapes, columns, target distribution (keep prints brief)
4. Build a model and generate predictions for the test set. Try to achieve THE BEST possible score!
5. Save ./submission.csv with ALL columns from the sample submission,
   only replacing the prediction column values (preserve Date, Comment, id, etc.)
6. When done, reply with plain text (no tool call) saying you are finished

GUIDELINES:
- All file paths must start with ./ (e.g. ./home/data/train.csv)
- Standard ML libraries are available: pandas, numpy, sklearn, xgboost, lightgbm
- Keep it simple and robust; a working baseline beats a crashed complex model
- Print only key metrics (shape, score) - avoid verbose output
- Add `import warnings; warnings.filterwarnings('ignore')` to suppress sklearn warnings
- If code raises an error, read the traceback carefully and fix it in the next call
- Tool output is truncated at {max_output} chars; keep prints concise
- CRITICAL: submission.csv must have the SAME columns as the sample submission file
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
    ):
        self.workdir = Path(workdir).resolve()
        self.max_iterations = max_iterations
        self.updater = updater
        self._loop: asyncio.AbstractEventLoop | None = None
        self.interpreter = Interpreter(workdir=str(self.workdir), timeout=code_timeout)
        self.llm = ChatOpenRouter(model=model, api_key=api_key)
        self._graph = self._build_graph()

    def _post_status(self, text: str) -> None:
        if self.updater is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.updater.update_status(TaskState.working, new_agent_text_message(text)),
            self._loop,
        )

    def _build_graph(self):
        interpreter = self.interpreter
        max_iterations = self.max_iterations
        session_started = [False]

        @lc_tool
        def run_python(code: str) -> str:
            """Execute Python code in a persistent session and return stdout/stderr."""
            result = interpreter.run(code, reset_session=not session_started[0])
            session_started[0] = True
            return result.output

        llm_with_tools = self.llm.bind_tools([run_python])

        def llm_node(state: AgentState) -> dict:
            iteration = state["iteration"] + 1
            _log_separator(f"STEP {iteration}/{max_iterations}")
            self._post_status(f"Step {iteration}/{max_iterations}: thinking...")

            response = llm_with_tools.invoke(state["messages"])
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    code_preview = tool_call["args"].get("code", "")[:120].replace("\n", " ")
                    self._post_status(
                        f"Step {iteration}/{max_iterations}: run_python({code_preview}...)"
                    )
            else:
                self._post_status(f"Step {iteration}/{max_iterations}: done (no tool call)")

            return {"messages": [response], "iteration": iteration}

        def tool_node(state: AgentState) -> dict:
            last: AIMessage = state["messages"][-1]
            results = []
            for tool_call in last.tool_calls:
                if tool_call["name"] == "run_python":
                    raw_output = run_python.invoke(tool_call["args"]["code"])
                    clean_output = _strip_warnings(raw_output)
                    truncated = _truncate_output(clean_output, MAX_OUTPUT_CHARS)
                    output_preview = truncated[:200].replace("\n", " ")
                    self._post_status(f"Output: {output_preview}")
                    results.append(ToolMessage(content=truncated, tool_call_id=tool_call["id"]))
            return {"messages": results}

        def router(state: AgentState) -> str:
            if state["iteration"] >= max_iterations:
                logger.warning("Max iterations (%d) reached, stopping.", max_iterations)
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

            initial_state: AgentState = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=instructions),
                ],
                "iteration": 0,
            }

            try:
                for _ in self._graph.stream(initial_state, stream_mode="updates"):
                    pass
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


_WARNING_BLOCK_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)


def _strip_warnings(text: str) -> str:
    return _WARNING_BLOCK_RE.sub("", text).strip()


def _truncate_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 80
    truncated_chars = len(text) - keep
    return f"[...{truncated_chars} chars truncated...]\n{text[-keep:]}"


def _log_separator(title: str) -> None:
    bar = "-" * 70
    logger.info("\n%s\n %s\n%s", bar, title, bar)
