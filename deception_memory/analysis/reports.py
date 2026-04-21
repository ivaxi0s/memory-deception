from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deception_memory.analysis.metrics import compute_summary
from deception_memory.schemas import SampleRecord


def export_reports(samples: list[SampleRecord], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(samples)
    rows = [
        {
            "sample_id": sample.sample_id,
            "accepted": sample.accepted,
            "judge_score": sample.judge_result.score,
            "deception_family": sample.memory_bundle.motive_family.value,
            "distortion_type": sample.truth_anchor.distortion_type.value,
            "query_surface": sample.query.surface_type.value,
        }
        for sample in samples
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = [
        "# Deception Memory Report",
        "",
        f"- Generated candidates: {summary['generated_candidates']}",
        f"- Accepted samples: {summary['accepted_samples']}",
        f"- Acceptance rate: {summary['acceptance_rate']:.2%}",
        f"- Mean judge score: {summary['mean_judge_score']:.2f}",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary
