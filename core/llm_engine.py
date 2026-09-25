from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic

from core.config import Settings
from observability.tracing import TokenUsage
from prompts.agent_prompt import AGENT_SYSTEM_PROMPT
from prompts.jewelry_prompt import JEWELRY_SYSTEM_PROMPT


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot complete a request."""


class LLMClient(Protocol):
    """Minimal interface required by the generation pipeline."""

    def generate(self, prompt: str) -> "LLMTextResponse | str":
        """Generate a raw text response for a prompt."""


@dataclass(frozen=True)
class LLMTextResponse:
    text: str
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolLLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: TokenUsage | None = None


class ToolUsingLLMClient(Protocol):
    """Provider-neutral interface for bounded native tool use."""

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict]
    ) -> ToolLLMResponse:
        """Return either final text or one or more native tool requests."""


class AnthropicLLMClient:
    """Anthropic implementation of the pipeline's LLM interface."""

    def __init__(self, settings: Settings, client: Anthropic | None = None):
        self._client = client or Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, prompt: str) -> LLMTextResponse:
        """Send a prompt to Claude and return its text response."""
        try:
            message = self._client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=JEWELRY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if not message.content or not hasattr(message.content[0], "text"):
                raise LLMProviderError("The provider returned no text content")
            raw = message.content[0].text.strip()
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError("The LLM provider request failed") from exc

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        provider_usage = getattr(message, "usage", None)
        usage = None
        if provider_usage is not None:
            usage = TokenUsage(
                input_tokens=provider_usage.input_tokens,
                output_tokens=provider_usage.output_tokens,
            )
        return LLMTextResponse(text=raw.strip(), usage=usage)

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict]
    ) -> ToolLLMResponse:
        """Run one native Anthropic tool-use turn."""
        anthropic_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
                "strict": True,
            }
            for tool in tools
        ]
        try:
            message = self._client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=AGENT_SYSTEM_PROMPT,
                messages=messages,
                tools=anthropic_tools,
            )
            text_parts = []
            tool_calls = []
            for block in message.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )
            text = "\n".join(text_parts).strip() or None
            provider_usage = getattr(message, "usage", None)
            usage = None
            if provider_usage is not None:
                usage = TokenUsage(
                    input_tokens=provider_usage.input_tokens,
                    output_tokens=provider_usage.output_tokens,
                )
            return ToolLLMResponse(
                text=text,
                tool_calls=tool_calls,
                stop_reason=message.stop_reason,
                usage=usage,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError("The LLM provider request failed") from exc
