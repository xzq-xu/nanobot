# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for nanobot-gateway — standalone binary for the
shellClaw ↔ nanobot gateway bridge.

Build:
    pyinstaller nanobot-gateway.spec

The output binary name includes platform/arch so electron-builder
can pick the right one at package time.
"""

import platform
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
_os_map = {"Darwin": "darwin", "Linux": "linux", "Windows": "win"}
_arch_map = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}
plat = _os_map.get(platform.system(), platform.system().lower())
arch = _arch_map.get(platform.machine(), platform.machine())
binary_name = f"nanobot-gateway-{plat}-{arch}"

# ---------------------------------------------------------------------------
# Hidden imports — modules loaded lazily / conditionally at runtime
# ---------------------------------------------------------------------------
hiddenimports = [
    # Providers (conditionally imported in _make_provider)
    "nanobot.providers.custom_provider",
    "nanobot.providers.litellm_provider",
    "nanobot.providers.openai_codex_provider",
    "nanobot.providers.transcription",
    # Agent internals
    "nanobot.agent.tools.mcp",
    "nanobot.agent.messages",
    "nanobot.agent.steering",
    # Channels (lazy-loaded in channels/manager.py — include all for completeness)
    "nanobot.channels.telegram",
    "nanobot.channels.whatsapp",
    "nanobot.channels.discord",
    "nanobot.channels.feishu",
    "nanobot.channels.mochat",
    "nanobot.channels.dingtalk",
    "nanobot.channels.email",
    "nanobot.channels.slack",
    "nanobot.channels.qq",
    # Gateway bridge
    "nanobot.gateway_bridge",
    "nanobot.gateway_bridge.server",
    "nanobot.gateway_bridge.agent",
    "nanobot.gateway_bridge.gateway_types",
    # Third-party: gateway stack
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "sse_starlette",
    # Third-party: MCP
    "mcp",
    "mcp.client.stdio",
    "mcp.client.streamable_http",
    "mcp.types",
    # Third-party: misc
    "oauth_cli_kit",
    "json_repair",
]

# litellm has extensive dynamic imports and ships data files (JSON, tokenizers)
# that it loads via importlib.resources at runtime — collect everything.
_litellm_datas, _litellm_binaries, _litellm_hiddenimports = collect_all("litellm")
hiddenimports += _litellm_hiddenimports

# ---------------------------------------------------------------------------
# Data files — non-Python resources that must ship with the binary
# ---------------------------------------------------------------------------
datas = [
    ("nanobot/templates", "nanobot/templates"),
    ("nanobot/skills", "nanobot/skills"),
]

# litellm data: JSON configs, tokenizer files, etc.
datas += _litellm_datas

# ---------------------------------------------------------------------------
# Excludes — large packages not needed by the gateway
# ---------------------------------------------------------------------------
excludes = [
    "tkinter",
    "matplotlib",
    "PIL",
    "scipy",
    "numpy",
    "pandas",
    "IPython",
    "notebook",
    "test",
    "unittest",
    # Matrix channel (optional dependency, large native deps)
    "nio",
    "matrix_nio",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["gateway_entry.py"],
    pathex=["."],
    binaries=_litellm_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=binary_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=(plat != "win"),
    upx=(plat != "win"),
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
