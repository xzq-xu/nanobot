from pathlib import Path

from nanobot.session.manager import SessionManager


def test_list_sessions_skips_corrupt_preview_lines_and_keeps_metadata_key(tmp_path: Path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("agent:main:desk-9af3997d")
    session.metadata["title"] = "语音调试"
    session.add_message("assistant", "我正在处理语音转文字")
    manager.save(session)

    with open(manager._get_session_path(session.key), "a", encoding="utf-8") as f:
        f.write("{ broken json\n")

    sessions = manager.list_sessions()

    assert sessions[0]["key"] == "agent:main:desk-9af3997d"
    assert sessions[0]["title"] == "语音调试"
    assert sessions[0]["preview"] == "我正在处理语音转文字"


def test_repair_uses_stored_metadata_key_when_called_with_filename_fallback(tmp_path: Path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("agent:main:desk-9af3997d")
    session.add_message("user", "调试语音转文字")
    manager.save(session)

    with open(manager._get_session_path(session.key), "a", encoding="utf-8") as f:
        f.write("{ broken json\n")

    repaired = manager._repair("agent:main_desk-9af3997d")

    assert repaired is not None
    assert repaired.key == "agent:main:desk-9af3997d"
    assert repaired.messages[0]["content"] == "调试语音转文字"
