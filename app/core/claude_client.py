"""Server-side Claude API integration stub.

Never call this from the client — the API key must never leave the backend.
Reads ANTHROPIC_API_KEY from the environment (set it via .env locally, and
via AWS Secrets Manager / SSM in deployed environments — never hardcode it
or put it in infra/*.tfvars).
"""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

_client = AsyncAnthropic()  # picks up ANTHROPIC_API_KEY from the environment

DEFAULT_MODEL = "claude-opus-5"


async def ask(prompt: str, *, system: str | None = None) -> str:
    """One-shot, non-streaming call — fine for short, latency-tolerant tasks."""
    response = await _client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


async def ask_streaming(prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
    """Streaming call — use this for chat/agent endpoints so the client sees
    tokens as they're generated (pairs well with the Lambda Function URL's
    RESPONSE_STREAM invoke mode configured in infra/main.tf)."""
    async with _client.messages.stream(
        model=DEFAULT_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
