"""Shared QC result containers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QCMessage:
    """One validation message."""

    level: str
    code: str
    message: str
    location: str | None = None


@dataclass
class QCReport:
    """Structured validation report."""

    name: str
    messages: list[QCMessage] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, location: str | None = None) -> None:
        self.messages.append(QCMessage("error", code, message, location))

    def add_warning(self, code: str, message: str, location: str | None = None) -> None:
        self.messages.append(QCMessage("warning", code, message, location))

    @property
    def ok(self) -> bool:
        return not any(message.level == "error" for message in self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "metrics": self.metrics,
            "messages": [asdict(message) for message in self.messages],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
