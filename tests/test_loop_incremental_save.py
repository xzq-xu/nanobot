"""Tests for incremental session saving during the agent loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path, *, tool_iterations: int = 2) -> AgentLoop:
    """Build an AgentLoop whose provider returns *tool_iterations* rounds of
    tool calls followed by a final text response."""
    call_count = [0]
    total = tool_iterations

    async def _chat_with_retry(*, messages, tools, model):
        call_count[0] += 1
        if call_count[0] <= total:
            return LLMResponse(
                content=f"thinking-{call_count[0]}",
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_{call_count[0]}",
                        name="test_tool",
                        arguments={"n": call_count[0]},
                    )
                ],
            )
        return LLMResponse(content="final answer", tool_calls=[])

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = _chat_with_retry

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=0,
    )

    loop.tools.register(MagicMock(
        name="test_tool",
        to_schema=MagicMock(return_value={"type": "function", "function": {"name": "test_tool"}}),
    ))
    loop.tools.execute = AsyncMock(return_value="tool-result")
    loop.tools.get_definitions = MagicMock(return_value=[])

    return loop


@pytest.mark.asyncio
async def test_incremental_save_called_per_iteration(tmp_path) -> None:
    """on_turn_saved should fire after each tool-call iteration."""
    loop = _make_loop(tmp_path, tool_iterations=3)
    saved_snapshots: list[int] = []

    def on_saved(msgs):
        saved_snapshots.append(len(msgs))

    content, tools_used, messages = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}],
        on_turn_saved=on_saved,
    )
    assert content == "final answer"
    assert len(saved_snapshots) == 3
    assert saved_snapshots[0] < saved_snapshots[1] < saved_snapshots[2]


@pytest.mark.asyncio
async def test_incremental_save_not_called_for_direct_response(tmp_path) -> None:
    """When the LLM returns a direct response (no tools), on_turn_saved
    should NOT be called — no intermediate state to persist."""
    loop = _make_loop(tmp_path, tool_iterations=0)
    saved: list[int] = []

    content, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}],
        on_turn_saved=lambda msgs: saved.append(len(msgs)),
    )
    assert content == "final answer"
    assert saved == []


@pytest.mark.asyncio
async def test_process_direct_saves_incrementally(tmp_path) -> None:
    """process_direct should persist session after each tool iteration,
    so intermediate results survive a mid-loop crash."""
    loop = _make_loop(tmp_path, tool_iterations=2)

    save_call_count = [0]
    original_save = loop.sessions.save

    def tracking_save(session):
        save_call_count[0] += 1
        original_save(session)

    loop.sessions.save = tracking_save

    await loop.process_direct("hello", session_key="cli:test")

    # 2 incremental saves (one per tool iteration) + 1 final save = 3
    assert save_call_count[0] == 3

    session = loop.sessions.get_or_create("cli:test")
    assert len(session.messages) > 0
    tool_results = [m for m in session.messages if m.get("role") == "tool"]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_crash_mid_loop_preserves_earlier_iterations(tmp_path) -> None:
    """If the LLM provider crashes on iteration 2, iteration 1's messages
    should already be persisted in the session.

    Note: tool execution errors are caught by return_exceptions=True in
    asyncio.gather, so we crash the *provider* (outside gather) instead.
    """
    call_count = [0]

    async def _chat_with_retry(*, messages, tools, model):
        call_count[0] += 1
        if call_count[0] == 1:
            return LLMResponse(
                content="thinking-1",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="test_tool",
                        arguments={"n": 1},
                    )
                ],
            )
        raise RuntimeError("LLM provider boom")

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = _chat_with_retry

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=0,
    )
    loop.tools.register(MagicMock(
        name="test_tool",
        to_schema=MagicMock(return_value={"type": "function", "function": {"name": "test_tool"}}),
    ))
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.tools.get_definitions = MagicMock(return_value=[])

    session = loop.sessions.get_or_create("cli:test")

    save_offset = [1]
    def _incremental_save(msgs):
        loop._save_turn(session, msgs, save_offset[0])
        save_offset[0] = len(msgs)
        loop.sessions.save(session)

    with pytest.raises(RuntimeError, match="LLM provider boom"):
        await loop._run_agent_loop(
            [{"role": "user", "content": "hi"}],
            on_turn_saved=_incremental_save,
        )

    # Iteration 1 (assistant + tool_result) should already be saved
    assert len(session.messages) >= 2
    roles = [m["role"] for m in session.messages]
    assert "assistant" in roles
    assert "tool" in roles
