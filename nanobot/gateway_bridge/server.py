"""FastAPI server for nanobot gateway bridge."""

import asyncio
import mimetypes
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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


_gateway_port: int = 18790

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_LOCAL_IMG_RE = re.compile(
    r'(!\[[^\]]*\])\((/[^)]+?\.(?:png|jpe?g|gif|webp|bmp|svg))\)',
    re.IGNORECASE,
)


def _rewrite_local_images(text: str) -> str:
    """Rewrite markdown images with local paths to gateway /files/ URLs."""
    if not text:
        return text

    def _replace(m):
        alt, fpath = m.group(1), m.group(2)
        if Path(fpath).is_file():
            return f"{alt}(http://127.0.0.1:{_gateway_port}/files{quote(fpath)})"
        return m.group(0)

    return _LOCAL_IMG_RE.sub(_replace, text)


def _file_paths_to_urls(paths: list[str]) -> list[str]:
    """Convert local file paths to gateway /files/ URLs."""
    urls = []
    for p in paths:
        fp = Path(p)
        if fp.is_file() and fp.suffix.lower() in _IMAGE_EXTS:
            urls.append(f"http://127.0.0.1:{_gateway_port}/files{quote(str(fp))}")
        elif p.startswith("http"):
            urls.append(p)
    return urls


def get_agent() -> NanobotAgent:
    """Get the global agent instance."""
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    return _agent


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve local files (images) so the frontend can display them."""
    fp = Path("/") / file_path
    if not fp.is_file():
        return Response(status_code=404, content="Not found")
    mime, _ = mimetypes.guess_type(str(fp))
    return FileResponse(fp, media_type=mime or "application/octet-stream")


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
        for msg in history:
            if isinstance(msg.get("content"), str):
                msg["content"] = _rewrite_local_images(msg["content"])
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


@app.get("/sessions")
async def list_sessions():
    """List all sessions."""
    agent = get_agent()
    try:
        sessions = await agent.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error("List sessions error: {}", e)
        return {"sessions": [], "error": str(e)}


@app.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    agent = get_agent()
    try:
        success = await agent.delete_session(session_id)
        return {"success": success}
    except Exception as e:
        logger.error("Delete session error: {}", e)
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
    # Map (channel, chat_id) → session_key so bus_outbound_loop can tag
    # outbound messages with the correct session_key.
    chat_to_session: dict[tuple[str, str], str] = {}

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
        chat_to_session[("gateway", "direct")] = session_id

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
                    "type": "delta", "delta": _rewrite_local_images(content),
                    "run_id": run_id, "session_key": session_id,
                })

        _enqueue({
            "type": "lifecycle", "phase": "start",
            "run_id": run_id, "session_key": session_id,
        })
        orphans: list = []
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
            orphans = checker.drain_all()
            _enqueue({
                "type": "final", "content": _rewrite_local_images(result or ""),
                "run_id": run_id, "session_key": session_id,
                **({"has_more": True} if orphans else {}),
            })
            if not orphans:
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

            for orphan in orphans:
                logger.info("Steering: re-queuing orphaned message as new task for {}", session_id)
                new_run_id = str(uuid.uuid4())
                task = asyncio.create_task(_run_chat(orphan.content, session_id, new_run_id))
                active_tasks[session_id] = task

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

    async def bus_outbound_loop():
        """Forward OutboundMessage from the message tool (media) to WebSocket."""
        try:
            while True:
                msg = await agent.bus.consume_outbound()
                media = msg.media or []
                content = _rewrite_local_images(msg.content or "")
                image_urls = _file_paths_to_urls(media) if media else []
                session_key = chat_to_session.get(
                    (msg.channel, msg.chat_id),
                    next(iter(active_tasks), msg.chat_id or "cli:direct"),
                )
                if content or image_urls:
                    _enqueue({
                        "type": "message",
                        "content": content,
                        "images": image_urls,
                        "session_key": session_key,
                    })
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("bus_outbound_loop crashed: {}", e)

    recv = asyncio.create_task(receive_loop())
    send = asyncio.create_task(send_loop())
    bus_out = asyncio.create_task(bus_outbound_loop())

    try:
        done, _pending = await asyncio.wait(
            [recv, send, bus_out], return_when=asyncio.FIRST_COMPLETED,
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
        bus_out.cancel()


def main():
    """CLI entry point."""
    import argparse
    global _gateway_port

    parser = argparse.ArgumentParser(description="nanobot gateway bridge")
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _gateway_port = args.port

    uvicorn.run(
        "nanobot.gateway_bridge.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
