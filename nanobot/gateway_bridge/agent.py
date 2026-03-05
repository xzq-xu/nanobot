"""Nanobot agent wrapper for gateway bridge."""

import asyncio
import uuid
from typing import AsyncGenerator, Callable

from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.session.manager import SessionManager


class NanobotAgent:
    """Wrapper around AgentLoop for external API access."""

    def __init__(self, config, workspace_path):
        self.config = config
        self.workspace_path = workspace_path
        self.bus = MessageBus()
        self.provider = self._make_provider()
        self.session_manager = SessionManager(workspace_path)
        self._agent = None

    def _make_provider(self):
        """Create the appropriate LLM provider from config."""
        from nanobot.providers.custom_provider import CustomProvider
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        config = self.config
        model = config.agents.defaults.model
        provider_name = config.get_provider_name()
        p = config.get_provider()
        api_base = config.get_api_base()

        if model.startswith("openai-codex/"):
            return OpenAICodexProvider(default_model=model)

        if provider_name == "custom":
            return CustomProvider(
                api_key=p.api_key if p else "no-key",
                api_base=api_base or "http://localhost:8000/v1",
                default_model=model,
            )

        return LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=provider_name,
        )

    def _create_agent(self) -> AgentLoop:
        """Create and configure AgentLoop instance."""
        config = self.config
        exec_config = config.tools.exec

        from nanobot.config.loader import get_data_dir
        cron_store_path = get_data_dir() / "cron" / "jobs.json"

        from nanobot.cron.service import CronService
        cron = CronService(store_path=cron_store_path)

        return AgentLoop(
            bus=self.bus,
            provider=self.provider,
            workspace=self.workspace_path,
            model=config.agents.defaults.model,
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            max_iterations=config.agents.defaults.max_tool_iterations,
            memory_window=config.agents.defaults.memory_window,
            reasoning_effort=config.agents.defaults.reasoning_effort,
            brave_api_key=config.tools.web.search.api_key if config.tools.web.search else None,
            web_proxy=config.tools.web.proxy,
            exec_config=exec_config,
            cron_service=cron,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            session_manager=self.session_manager,
            mcp_servers=config.tools.mcp_servers or None,
            channels_config=config.channels,
            enable_steering=True,
        )

    async def start(self):
        """Start the agent."""
        if self._agent is None:
            self._agent = self._create_agent()

    async def stop(self):
        """Stop the agent."""
        if self._agent:
            await self._agent.close_mcp()
            await self._agent.stop()

    async def chat(self, message: str, session_id: str = "cli:direct") -> tuple[str, list[dict]]:
        """
        Send a chat message and get response.
        
        Returns: (final_content, tool_calls)
        """
        if self._agent is None:
            await self.start()

        run_id = str(uuid.uuid4())

        async def on_progress(content: str, tool_hint: bool = False):
            pass

        try:
            result = await self._agent.process_direct(
                content=message,
                session_key=session_id,
                channel="cli",
                chat_id="direct",
                on_progress=on_progress,
            )
            return (result, [])
        except Exception as e:
            logger.error("Error in chat: {}", e)
            return ("", [])

    async def chat_stream(self, message: str, session_id: str = "cli:direct") -> AsyncGenerator:
        """
        Send a chat message and yield streaming events.
        
        Yields: event dictionaries
        """
        if self._agent is None:
            await self.start()

        run_id = str(uuid.uuid4())
        yield {
            "type": "lifecycle",
            "phase": "start",
            "run_id": run_id,
            "session_key": session_id,
        }

        async def on_progress(content: str, tool_hint: bool = False):
            pass

        try:
            result = await self._agent.process_direct(
                content=message,
                session_key=session_id,
                channel="cli",
                chat_id="direct",
                on_progress=on_progress,
            )
            yield {
                "type": "final",
                "content": result,
                "run_id": run_id,
                "session_key": session_id,
            }
            yield {
                "type": "lifecycle",
                "phase": "end",
                "run_id": run_id,
                "session_key": session_id,
            }
        except Exception as e:
            logger.error("Error in chat_stream: {}", e)
            yield {
                "type": "lifecycle",
                "phase": "error",
                "error": str(e),
                "run_id": run_id,
                "session_key": session_id,
            }

    _STEERING_PREFIX = (
        "[The user just sent a new message while you were working. "
        "Read it and decide: continue current work, switch to the "
        "new request, or address both.]\n\n"
    )

    async def get_history(self, session_id: str) -> list[dict]:
        """Get chat history for a session, stripping internal steering metadata
        and surfacing media from message tool calls."""
        import json as _json

        if self._agent is None:
            await self.start()
        session = self._agent.sessions.get_or_create(session_id)
        messages = session.get_history()
        out = []
        for msg in messages:
            content = msg.get("content", "")
            if msg["role"] == "user" and isinstance(content, str) and content.startswith(self._STEERING_PREFIX):
                content = content[len(self._STEERING_PREFIX):]

            # Extract content + media from message tool calls
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                msg_text, media_md = self._extract_message_tool_content(msg["tool_calls"])
                if msg_text:
                    if not content or content.strip() in ("", "..."):
                        content = msg_text
                    else:
                        content = content + "\n" + msg_text
                if media_md:
                    content = (content or "") + media_md

            out.append({"role": msg["role"], "content": content})
        return out

    @staticmethod
    def _extract_message_tool_content(tool_calls: list) -> tuple[str, str]:
        """Extract text content and media from message tool calls.
        Returns (text_content, media_markdown)."""
        import json as _json
        texts = []
        media_parts = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != "message":
                continue
            try:
                args = _json.loads(fn.get("arguments", "{}"))
            except (ValueError, TypeError):
                continue
            text = args.get("content", "")
            if text:
                texts.append(text)
            for path in args.get("media", []):
                media_parts.append(f"\n![image]({path})")
        return "\n".join(texts), "".join(media_parts)

    async def abort(self, session_id: str) -> bool:
        """Abort a running task.

        For WebSocket connections, abort is handled at the bridge level by
        cancelling the asyncio Task running process_direct().  This method
        exists for the HTTP abort endpoint as a best-effort fallback.
        """
        tasks = self._agent._active_tasks.pop(session_id, [])
        cancelled = sum(t.cancel() for t in tasks)
        return cancelled > 0

    async def list_sessions(self) -> list[dict]:
        """List all sessions."""
        return self.session_manager.list_sessions()

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by key."""
        return self.session_manager.delete(session_id)
