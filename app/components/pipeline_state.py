# app/components/pipeline_state.py
#
# Module-level progress store shared between streamlit_app.py and page scripts.
# Python's GIL makes dict access from multiple threads safe.
from __future__ import annotations

_sessions: dict[str, dict] = {}


def get_progress(sid: str) -> dict:
    return _sessions.get(
        sid,
        {"running": False, "step": 0, "msg": "", "result": None, "error": None},
    )


def set_progress(sid: str, **kwargs: object) -> None:
    if sid not in _sessions:
        _sessions[sid] = {}
    _sessions[sid].update(kwargs)


def is_running(sid: str) -> bool:
    return bool(_sessions.get(sid, {}).get("running", False))
