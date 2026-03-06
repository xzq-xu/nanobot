"""Tests for gateway history expansion."""

from __future__ import annotations

import pytest

from nanobot.gateway_bridge.agent import NanobotAgent
from nanobot.session.manager import Session


class _FakeSessions:
    def __init__(self, session: Session):
        self._session = session

    def get_or_create(self, _key: str) -> Session:
        return self._session


class _FakeLoop:
    def __init__(self, session: Session):
        self.sessions = _FakeSessions(session)


@pytest.mark.asyncio
async def test_get_history_preserves_tool_cards_and_message_bubbles() -> None:
    session = Session(key="agent:main:main")
    session.messages = [
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "content": "我先检查一下页面状态。",
            "tool_calls": [
                {
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"pwd"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_exec_1",
            "name": "exec",
            "content": "/tmp/workspace\n",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_message_1",
                    "type": "function",
                    "function": {
                        "name": "message",
                        "arguments": '{"content":"📸 当前状态","media":["/tmp/a.png","/tmp/b.png"]}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_message_1",
            "name": "message",
            "content": "Message sent to gateway:direct with 2 attachments",
        },
        {"role": "assistant", "content": "已经处理好了。"},
    ]

    agent = NanobotAgent.__new__(NanobotAgent)
    agent._agent = _FakeLoop(session)

    history = await NanobotAgent.get_history(agent, "agent:main:main")

    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我先检查一下页面状态。"},
        {
            "role": "tool",
            "name": "exec",
            "toolCallId": "call_exec_1",
            "args": '{"command":"pwd"}',
            "phase": "end",
            "result": "/tmp/workspace\n",
        },
        {
            "role": "tool",
            "name": "message",
            "toolCallId": "call_message_1",
            "args": '{"content":"📸 当前状态","media":["/tmp/a.png","/tmp/b.png"]}',
            "phase": "end",
            "result": "Message sent to gateway:direct with 2 attachments",
        },
        {
            "role": "assistant",
            "content": "📸 当前状态\n![image](/tmp/a.png)\n![image](/tmp/b.png)",
        },
        {"role": "assistant", "content": "已经处理好了。"},
    ]


@pytest.mark.asyncio
async def test_get_history_strips_steering_prefix_but_keeps_tool_cards() -> None:
    agent = NanobotAgent.__new__(NanobotAgent)
    session = Session(key="agent:main:main")
    session.messages = [
        {
            "role": "user",
            "content": agent._STEERING_PREFIX + "顺便再截图一下",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_exec_2",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"echo ok"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_exec_2",
            "name": "exec",
            "content": "ok\n",
        },
    ]
    agent._agent = _FakeLoop(session)

    history = await NanobotAgent.get_history(agent, "agent:main:main")

    assert history[0] == {"role": "user", "content": "顺便再截图一下"}
    assert history[1]["role"] == "tool"
    assert history[1]["name"] == "exec"
    assert history[1]["result"] == "ok\n"
