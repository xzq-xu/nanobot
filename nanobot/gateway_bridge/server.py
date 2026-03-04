"""FastAPI server for nanobot gateway bridge."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sse_starlette import EventSourceResponse

from nanobot.agent.steering import InterruptionChecker
from nanobot.bus.events import InboundMessage
from nanobot.config.loader import load_config
from nanobot import __version__

from .agent import NanobotAgent
from .gateway_types import ChatRequest, HealthResponse


_agent: NanobotAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _agent

    logger.info("Starting nanobot gateway bridge...")
    config = load_config()
    workspace = config.workspace_path

    _agent = NanobotAgent(config, workspace)
    await _agent.start()

    logger.info("Nanobot gateway bridge started")

    yield

    logger.info("Shutting down nanobot gateway bridge...")
    if _agent:
        await _agent.stop()
    logger.info("Nanobot gateway bridge stopped")


app = FastAPI(
    title="nanobot-gateway-bridge",
    description="HTTP/WebSocket API for nanobot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> NanobotAgent:
    """Get the global agent instance."""
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    return _agent


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        nanobot_version=__version__,
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a chat message (non-streaming). Returns the final response."""
    agent = get_agent()

    try:
        result, tool_calls = await agent.chat(
            message=request.message,
            session_id=request.session_id,
        )

        return {
            "run_id": str(uuid.uuid4()),
            "content": result,
            "tool_calls": tool_calls,
        }
    except Exception as e:
        logger.error("Chat error: {}", e)
        return {
            "run_id": str(uuid.uuid4()),
            "content": "",
            "error": str(e),
        }


@app.get("/chat/{session_id}/history")
async def get_history(session_id: str):
    """Get chat history for a session."""
    agent = get_agent()

    try:
        history = await agent.get_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.error("History error: {}", e)
        return {"session_id": session_id, "messages": [], "error": str(e)}


@app.post("/chat/{session_id}/abort")
async def abort_chat(session_id: str):
    """Abort a running chat session."""
    agent = get_agent()

    try:
        success = await agent.abort(session_id)
        return {"success": success}
    except Exception as e:
        logger.error("Abort error: {}", e)
        return {"success": False, "error": str(e)}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.

    Protocol:
    - Client sends: {"message": "...", "session_id": "..."}
    - Client sends: {"type": "abort", "session_id": "..."}
    - Server sends events: {"type": "delta|tool|lifecycle|final", ...}

    Steering / follow-up: when a new message arrives while processing is
    in progress, the message is injected via InterruptionChecker.  The model
    sees the new message at the next tool boundary or LLM call and decides
    itself whether to continue, switch, or address both.  No task
    cancellation, no waiting — the model decides.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    agent = get_agent()
    send_q: asyncio.Queue[dict] = asyncio.Queue()

    # One active task per session; checker is alive while task runs.
    active_tasks: dict[str, asyncio.Task] = {}
    checkers: dict[str, InterruptionChecker] = {}

    # -- helpers ----------------------------------------------------------

    def _enqueue(event: dict) -> None:
        try:
            send_q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def _cancel_task(session_id: str) -> None:
        """Explicit abort — cancel the in-flight task."""
        task = active_tasks.pop(session_id, None)
        checkers.pop(session_id, None)
        if task is None or task.done():
            return
        logger.info("Cancelling task for session {}", session_id)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # -- per-message processing -------------------------------------------

    async def _run_chat(message: str, session_id: str, run_id: str) -> None:
        checker = InterruptionChecker()
        checkers[session_id] = checker

        async def on_progress(content: str, tool_hint: bool = False):
            if tool_hint:
                name = content.split("(")[0] if "(" in content else content
                _enqueue({
                    "type": "tool", "phase": "start",
                    "name": name, "args": content,
                    "run_id": run_id, "session_key": session_id,
                })
            else:
                _enqueue({
                    "type": "delta", "delta": content,
                    "run_id": run_id, "session_key": session_id,
                })

        _enqueue({
            "type": "lifecycle", "phase": "start",
            "run_id": run_id, "session_key": session_id,
        })
        try:
            logger.info("Processing message: run_id={} session={}", run_id, session_id)
            result = await agent._agent.process_direct(
                content=message,
                session_key=session_id,
                channel="gateway",
                chat_id="direct",
                on_progress=on_progress,
                interruption_checker=checker,
            )
            logger.info("Run complete: run_id={} result_len={}", run_id, len(result or ""))
            _enqueue({
                "type": "final", "content": result or "",
                "run_id": run_id, "session_key": session_id,
            })
            _enqueue({
                "type": "lifecycle", "phase": "end",
                "run_id": run_id, "session_key": session_id,
            })
        except asyncio.CancelledError:
            logger.warning("Run aborted: run_id={}", run_id)
            _enqueue({
                "type": "lifecycle", "phase": "aborted",
                "run_id": run_id, "session_key": session_id,
            })
        except Exception as e:
            logger.exception("Chat error (run_id={}): {}", run_id, e)
            _enqueue({
                "type": "lifecycle", "phase": "error",
                "error": str(e),
                "run_id": run_id, "session_key": session_id,
            })
        finally:
            checkers.pop(session_id, None)
            active_tasks.pop(session_id, None)

    # -- two concurrent loops ---------------------------------------------

    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "message")
                session_id = data.get("session_id", "cli:direct")

                if msg_type == "abort":
                    logger.info("Abort requested for session {}", session_id)
                    await _cancel_task(session_id)
                    continue

                message = data.get("message", "")
                if not message:
                    _enqueue({"type": "error", "error": "Empty message"})
                    continue

                # Check for active task → steer via InterruptionChecker
                existing = active_tasks.get(session_id)
                if existing and not existing.done():
                    checker = checkers.get(session_id)
                    if checker:
                        inbound = InboundMessage(
                            channel="gateway",
                            sender_id="user",
                            chat_id="direct",
                            content=message,
                        )
                        await checker.signal(inbound)
                        logger.info("Steering: injected message into active session {}", session_id)
                        continue

                # No active task → start fresh
                run_id = str(uuid.uuid4())
                task = asyncio.create_task(_run_chat(message, session_id, run_id))
                active_tasks[session_id] = task
        except WebSocketDisconnect:
            logger.info("recv_loop: client disconnected")
        except Exception as e:
            logger.exception("recv_loop crashed: {}", e)

    async def send_loop():
        try:
            while True:
                event = await send_q.get()
                try:
                    await websocket.send_json(event)
                except Exception as e:
                    logger.warning("send_loop: failed to send ({}): {}", event.get("type"), e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("send_loop crashed: {}", e)

    recv = asyncio.create_task(receive_loop())
    send = asyncio.create_task(send_loop())

    try:
        done, _pending = await asyncio.wait(
            [recv, send], return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            if t.exception() is not None:
                logger.error("Loop exited with error: {}", t.exception())
            else:
                logger.info("Loop exited normally: {}", t.get_name())
    except Exception as e:
        logger.error("WebSocket wait error: {}", e)
    finally:
        for task in active_tasks.values():
            if not task.done():
                task.cancel()
        recv.cancel()
        send.cancel()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="nanobot gateway bridge")
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    uvicorn.run(
        "nanobot.gateway_bridge.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
