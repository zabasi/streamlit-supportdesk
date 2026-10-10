"""The answer prompt. Wording lives in the files named by ``prompts.*`` in the configuration;
this module only lays the case and the evidence out around it."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ConfigError, Settings
from ..data.environment import missing_environment
from ..types import CaseInput
from .packet import Packet


@dataclass(frozen=True)
class Prompts:
    system: str
    output_format: str


def load_prompts(settings: Settings) -> Prompts:
    def read(path, key):
        if not path.is_file():
            raise ConfigError(f"prompt file not found: {path} (setting prompts.{key})")
        return path.read_text(encoding="utf-8").strip()
    return Prompts(system=read(settings.prompts.system, "system"),
                   output_format=read(settings.prompts.output_format, "output_format"))


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"[{h.get('role', 'user')}] {h.get('body', '').strip()}" for h in history)


def _format_env(env: dict) -> str:
    known = ", ".join(f"{k}={v}" for k, v in sorted(env.items()) if v) or "(nothing stated)"
    unknown = ", ".join(missing_environment(env)) or "(none)"
    return f"Known: {known}\nNot stated (unknown, do not guess): {unknown}"


def _format_evidence(packet: Packet) -> str:
    if not packet.chunks:
        return "(no evidence retrieved)"
    parts = []
    label_of = {cid: label for label, cid in packet.labels.items()}
    for chunk in packet.chunks:
        attrs = f'type="{chunk.source_type}" url="{chunk.url}"'
        if chunk.version:
            attrs += f' version="{chunk.version}"'
        state = chunk.meta.get("state")
        if state:
            attrs += f' state="{state}"'
        parts.append(f'<evidence id="{label_of[chunk.chunk_id]}" {attrs}>\n{chunk.text}\n</evidence>')
    return "\n".join(parts)


def build_messages(case: CaseInput, env: dict, packet: Packet, prompts: Prompts, *,
                   max_body_chars: int) -> list[dict]:
    user = (
        f"## Report\nTitle: {case.title}\n\n{case.body[:max_body_chars]}\n\n"
        f"## Conversation so far\n{_format_history(case.history)}\n\n"
        f"## Environment\n{_format_env(env)}\n\n"
        f"## Evidence\n{_format_evidence(packet)}\n\n"
        f"## Output format\n{prompts.output_format}"
    )
    return [{"role": "system", "content": prompts.system}, {"role": "user", "content": user}]
