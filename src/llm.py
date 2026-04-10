import json
from typing import Any, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


class ChatOpenRouter(BaseChatModel):
    """OpenRouter wrapper that preserves reasoning/tool-calls."""

    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    include_reasoning: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: dict = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @staticmethod
    def _format_messages(messages: list[BaseMessage]) -> list[dict]:
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry: dict[str, Any] = {"role": "assistant"}
                entry["content"] = msg.content or ""
                reasoning = msg.additional_kwargs.get("reasoning")
                if reasoning:
                    entry["reasoning"] = reasoning
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    }
                )
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "include_reasoning": self.include_reasoning,
            **self.extra_params,
        }
        if stop:
            payload["stop"] = stop
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        resp = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        raw_msg = choice["message"]

        additional_kwargs: dict[str, Any] = {}
        if raw_msg.get("reasoning"):
            additional_kwargs["reasoning"] = raw_msg["reasoning"]

        tool_calls = []
        if raw_msg.get("tool_calls"):
            for tc in raw_msg["tool_calls"]:
                args = tc["function"]["arguments"]
                tool_calls.append(
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(args) if isinstance(args, str) else args,
                    }
                )

        ai_msg = AIMessage(
            content=raw_msg.get("content") or "",
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def bind_tools(self, tools: list, **kwargs: Any):
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)
