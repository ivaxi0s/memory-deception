from pathlib import Path

from deception_memory.config import PipelineConfig
from deception_memory.llm.client import MockLLMClient
from deception_memory.pipeline.orchestrator import Orchestrator


def build_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "seeds": {"raw_path": "data/seeds/demo_seeds.jsonl"},
            "generation": {"deception_families": ["reputational"], "max_iterations_per_anchor": 1},
            "evaluation": {"baseline_runs_per_candidate": 1, "min_judge_score": 4},
            "output": {"data_dir": str(tmp_path), "pretty_logs": False},
            "artifacts": {
                "prepared_seeds_path": str(tmp_path / "runs" / "prepared_seeds.jsonl"),
                "candidates_path": str(tmp_path / "runs" / "candidates.jsonl"),
                "baselines_path": str(tmp_path / "runs" / "baselines.jsonl"),
                "memory_evals_path": str(tmp_path / "runs" / "memory_evals.jsonl"),
                "judgments_path": str(tmp_path / "runs" / "judgments.jsonl"),
                "dataset_path": str(tmp_path / "outputs" / "dataset.jsonl"),
                "reports_dir": str(tmp_path / "outputs" / "reports"),
            },
        }
    )


def test_generate_candidates_requires_prepared_seeds(tmp_path: Path) -> None:
    orchestrator = Orchestrator(build_config(tmp_path), MockLLMClient(scripted_responses=[]), output_dir=tmp_path)
    try:
        orchestrator.generate_candidates()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError when prepared seeds artifact is missing")


def test_resume_uses_saved_artifacts(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    orchestrator = Orchestrator(config, MockLLMClient(scripted_responses=["not-json", "not-json"]), output_dir=tmp_path)
    orchestrator.prepare_seeds()
    result = orchestrator.resume()
    assert result["resumed_stage"] == "generate-candidates"
