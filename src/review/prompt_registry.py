"""
YAML-backed prompt template registry.

Each entry maps ``fixed_prompt_id`` → text template, allowed labels,
and isolation token.  Isolation tokens are always prepended even
under three-model isolation, for forward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptEntry:
    """One fixed-prompt template."""
    fixed_prompt_id: str
    text_template: str
    allowed_labels: tuple[str, ...]
    isolation_token: str


class PromptRegistry:
    """Look up prompt templates by ``fixed_prompt_id``."""

    def __init__(self, entries: dict[str, PromptEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_yaml(cls, path: Path | str) -> PromptRegistry:
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        entries: dict[str, PromptEntry] = {}
        for item in raw.get("prompts", []):
            pe = PromptEntry(
                fixed_prompt_id=item["fixed_prompt_id"],
                text_template=item["text_template"],
                allowed_labels=tuple(item["allowed_labels"]),
                isolation_token=item["isolation_token"],
            )
            entries[pe.fixed_prompt_id] = pe
        return cls(entries)

    def get(self, prompt_id: str) -> PromptEntry:
        try:
            return self._entries[prompt_id]
        except KeyError:
            raise KeyError(
                f"Unknown fixed_prompt_id {prompt_id!r}; "
                f"available: {sorted(self._entries)}"
            )

    def render(self, prompt_id: str) -> str:
        """Return the fully-rendered prompt text (token + template)."""
        pe = self.get(prompt_id)
        return f"{pe.isolation_token} {pe.text_template}"

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._entries
