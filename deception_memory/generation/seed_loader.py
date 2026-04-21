from __future__ import annotations

import json
from pathlib import Path

from deception_memory.schemas import TruthAnchor


def load_truth_anchors(path: Path) -> list[TruthAnchor]:
    anchors: list[TruthAnchor] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            anchors.append(TruthAnchor.model_validate(json.loads(line)))
    return anchors


def normalize_truth_anchors(anchors: list[TruthAnchor]) -> list[TruthAnchor]:
    seen: set[str] = set()
    deduped: list[TruthAnchor] = []
    for anchor in anchors:
        if anchor.anchor_id in seen:
            continue
        seen.add(anchor.anchor_id)
        deduped.append(anchor)
    return deduped
