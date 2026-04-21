from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JSONLStore(Generic[T]):
    def __init__(self, path: Path, model_type: type[T]) -> None:
        self.path = path
        self.model_type = model_type
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: T) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(item.model_dump_json())
            handle.write("\n")

    def write_all(self, items: Iterable[T]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(item.model_dump_json())
                handle.write("\n")

    def read_all(self) -> list[T]:
        if not self.path.exists():
            return []
        records: list[T] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                records.append(self.model_type.model_validate(data))
        return records
