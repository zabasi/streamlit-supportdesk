"""Pull environment facts out of free text. A fact that is not stated stays absent (never guessed)."""
from __future__ import annotations

import re

_VERSION = r"v?(\d+\.\d+(?:\.\d+)?(?:[a-z]+\d*)?)"
_PATTERNS = {
    "streamlit_version": re.compile(r"streamlit(?:\s+version)?\s*[:=]?\s*" + _VERSION, re.IGNORECASE),
    "python_version": re.compile(r"python(?:\s+version)?\s*[:=]?\s*" + _VERSION, re.IGNORECASE),
}
_OS = (("windows", "Windows"), ("macos", "macOS"), ("mac os", "macOS"), ("ubuntu", "Ubuntu"),
       ("debian", "Debian"), ("linux", "Linux"))
_BROWSERS = (("chrome", "Chrome"), ("firefox", "Firefox"), ("safari", "Safari"), ("edge", "Edge"))


def extract_environment(text: str) -> dict:
    """Return only the fields that are actually present in ``text``."""
    env: dict = {}
    for key, pattern in _PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            env[key] = match.group(1)
    lowered = (text or "").lower()
    for needle, label in _OS:
        if re.search(rf"\b{needle}\b", lowered):
            env["operating_system"] = label
            break
    for needle, label in _BROWSERS:
        if re.search(rf"\b{needle}\b", lowered):
            env["browser"] = label
            break
    return env


UNKNOWN_FIELDS = ("streamlit_version", "python_version", "operating_system", "browser")


def missing_environment(env: dict) -> list[str]:
    return [name for name in UNKNOWN_FIELDS if not env.get(name)]
