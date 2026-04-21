from __future__ import annotations

from pathlib import Path

from deception_memory.config import PipelineConfig
from deception_memory.enums import DeceptionFamily
from deception_memory.generation.memory_builder import MemoryBuilder
from deception_memory.generation.query_builder import QueryBuilder
from deception_memory.generation.refinement import RefinementEngine
from deception_memory.generation.seed_loader import load_truth_anchors, normalize_truth_anchors
from deception_memory.generation.truth_anchor import TruthAnchorValidator
from deception_memory.llm.client import BaseLLMClient
from deception_memory.logging_utils import get_logger
from deception_memory.pipeline.baseline_runner import BaselineRunner
from deception_memory.pipeline.dataset_builder import acceptance_decision, build_sample_record
from deception_memory.pipeline.eval_runner import MemoryEvalRunner
from deception_memory.pipeline.judge_runner import JudgeRunner
from deception_memory.pipeline.query_judge_runner import QueryJudgeRunner
from deception_memory.schemas import (
    BaselineResult,
    CandidateRecord,
    JudgmentArtifact,
    MemoryEvalResult,
    SampleRecord,
    TruthAnchor,
)
from deception_memory.storage.jsonl_store import JSONLStore
from deception_memory.storage.run_registry import RunRegistry


class Orchestrator:
    def __init__(self, config: PipelineConfig, client: BaseLLMClient, output_dir: Path | None = None) -> None:
        self.config = config
        self.client = client
        self.output_dir = output_dir or config.output.data_dir
        self.logger = get_logger(__name__)
        self.anchor_validator = TruthAnchorValidator()
        self.memory_builder = MemoryBuilder(client, config)
        self.query_builder = QueryBuilder(client, config)
        self.baseline_runner = BaselineRunner(client, config)
        self.eval_runner = MemoryEvalRunner(client, config)
        self.judge_runner = JudgeRunner(client, config)
        self.query_judge_runner = QueryJudgeRunner(client, config)
        self.refiner = RefinementEngine(client, config)
        self.registry = RunRegistry(self.output_dir / "runs" / "run_registry.json")

    def _resolve(self, override: Path | None, default: Path) -> Path:
        return override or default

    def _require_input(self, path: Path, label: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path

    def _record_stage(self, stage: str, path: Path, count: int) -> None:
        self.registry.update_run(stage, {"path": str(path), "count": count})

    def prepare_seeds(self, raw_path: Path | None = None, output_path: Path | None = None) -> list[TruthAnchor]:
        input_path = self._resolve(raw_path, self.config.seeds.raw_path)
        output_path = self._resolve(output_path, self.config.artifacts.prepared_seeds_path)
        anchors = normalize_truth_anchors(load_truth_anchors(input_path))
        valid = [anchor for anchor in anchors if self.anchor_validator.validate(anchor).valid]
        if valid:
            store = JSONLStore(output_path, TruthAnchor)
            store.write_all(valid)
        self._record_stage("prepare-seeds", output_path, len(valid))
        return valid

    def generate_candidates(
        self,
        prepared_seeds_path: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CandidateRecord]:
        input_path = self._resolve(prepared_seeds_path, self.config.artifacts.prepared_seeds_path)
        output_path = self._resolve(output_path, self.config.artifacts.candidates_path)
        self._require_input(input_path, "prepared seeds artifact")
        anchors = JSONLStore(input_path, TruthAnchor).read_all()
        candidates: list[CandidateRecord] = []
        for anchor in anchors:
            for family_name in self.config.generation.deception_families:
                family = DeceptionFamily(family_name)
                candidate = CandidateRecord(
                        candidate_id=f"{anchor.anchor_id}_{family.value}",
                        truth_anchor=anchor,
                        memory_bundle=self.memory_builder.build(anchor, family),
                        query=self.query_builder.build(anchor),
                        iteration_index=0,
                    )
                for attempt in range(self.config.generation.max_iterations_per_anchor):
                    query_judge = self.query_judge_runner.run(candidate)
                    candidate.query_judge_result = query_judge
                    candidate.iteration_index = attempt
                    candidate.mutation_history.append(
                        {
                            "attempt": attempt,
                            "query_judge_score": query_judge.score,
                            "objective_query": query_judge.objective_query,
                            "nonleading": query_judge.nonleading,
                            "grounded_in_anchor": query_judge.grounded_in_anchor,
                            "feedback": query_judge.feedback_for_next_iteration,
                        }
                    )
                    if (
                        query_judge.score >= self.config.evaluation.min_query_judge_score
                        and query_judge.objective_query
                        and query_judge.nonleading
                        and query_judge.grounded_in_anchor
                    ):
                        break
                    refinement = self.refiner.refine_pre_eval(candidate, query_judge)
                    if refinement.revised_memory_bundle is not None:
                        candidate.memory_bundle = refinement.revised_memory_bundle
                    if refinement.revised_query is not None:
                        candidate.query = refinement.revised_query
                    if refinement.revised_memory_bundle is None and refinement.revised_query is None:
                        candidate.query = self.query_builder.fallback_query(anchor)
                candidates.append(candidate)
        JSONLStore(output_path, CandidateRecord).write_all(candidates)
        self._record_stage("generate-candidates", output_path, len(candidates))
        return candidates

    def run_baselines(
        self,
        candidates_path: Path | None = None,
        output_path: Path | None = None,
    ) -> list[BaselineResult]:
        input_path = self._resolve(candidates_path, self.config.artifacts.candidates_path)
        output_path = self._resolve(output_path, self.config.artifacts.baselines_path)
        self._require_input(input_path, "candidate artifact")
        candidates = JSONLStore(input_path, CandidateRecord).read_all()
        results = [
            BaselineResult(
                candidate_id=candidate.candidate_id,
                anchor_id=candidate.truth_anchor.anchor_id,
                bundle_id=candidate.memory_bundle.bundle_id,
                query_id=candidate.query.query_id,
                baseline_runs=self.baseline_runner.run(candidate.truth_anchor, candidate.query),
            )
            for candidate in candidates
        ]
        JSONLStore(output_path, BaselineResult).write_all(results)
        self._record_stage("run-baselines", output_path, len(results))
        return results

    def run_memory_evals(
        self,
        candidates_path: Path | None = None,
        baselines_path: Path | None = None,
        output_path: Path | None = None,
    ) -> list[MemoryEvalResult]:
        candidates_input = self._resolve(candidates_path, self.config.artifacts.candidates_path)
        baselines_input = self._resolve(baselines_path, self.config.artifacts.baselines_path)
        output_path = self._resolve(output_path, self.config.artifacts.memory_evals_path)
        self._require_input(candidates_input, "candidate artifact")
        self._require_input(baselines_input, "baseline artifact")
        candidates = JSONLStore(candidates_input, CandidateRecord).read_all()
        baseline_results = JSONLStore(baselines_input, BaselineResult).read_all()
        baselines_by_candidate = {result.candidate_id: result for result in baseline_results}
        results: list[MemoryEvalResult] = []
        for candidate in candidates:
            baseline = baselines_by_candidate.get(candidate.candidate_id)
            if baseline is None:
                raise FileNotFoundError(f"Missing baseline result for candidate {candidate.candidate_id}")
            memory_run = self.eval_runner.run(candidate.truth_anchor, candidate.memory_bundle, candidate.query)
            results.append(
                MemoryEvalResult(
                    candidate_id=candidate.candidate_id,
                    anchor_id=candidate.truth_anchor.anchor_id,
                    bundle_id=candidate.memory_bundle.bundle_id,
                    query_id=candidate.query.query_id,
                    baseline_run_ids=[run.run_id for run in baseline.baseline_runs],
                    memory_run=memory_run,
                )
            )
        JSONLStore(output_path, MemoryEvalResult).write_all(results)
        self._record_stage("run-memory-evals", output_path, len(results))
        return results

    def judge_candidates(
        self,
        candidates_path: Path | None = None,
        baselines_path: Path | None = None,
        memory_evals_path: Path | None = None,
        output_path: Path | None = None,
    ) -> list[JudgmentArtifact]:
        candidates_input = self._resolve(candidates_path, self.config.artifacts.candidates_path)
        baselines_input = self._resolve(baselines_path, self.config.artifacts.baselines_path)
        memory_input = self._resolve(memory_evals_path, self.config.artifacts.memory_evals_path)
        output_path = self._resolve(output_path, self.config.artifacts.judgments_path)
        self._require_input(candidates_input, "candidate artifact")
        self._require_input(baselines_input, "baseline artifact")
        self._require_input(memory_input, "memory eval artifact")
        candidates = JSONLStore(candidates_input, CandidateRecord).read_all()
        baselines = {result.candidate_id: result for result in JSONLStore(baselines_input, BaselineResult).read_all()}
        memory_results = {result.candidate_id: result for result in JSONLStore(memory_input, MemoryEvalResult).read_all()}
        judgments: list[JudgmentArtifact] = []
        for candidate in candidates:
            baseline_result = baselines.get(candidate.candidate_id)
            memory_result = memory_results.get(candidate.candidate_id)
            if baseline_result is None or memory_result is None:
                raise FileNotFoundError(f"Missing evaluation artifacts for candidate {candidate.candidate_id}")
            judge_result = self.judge_runner.run(
                candidate,
                baseline_result.baseline_runs,
                memory_result.memory_run,
                query_judge_result=candidate.query_judge_result,
            )
            accepted = acceptance_decision(
                candidate,
                baseline_result.baseline_runs,
                memory_result.memory_run,
                judge_result,
                self.config,
            )
            history = [
                {
                    "iteration_index": candidate.iteration_index,
                    "judge_score": judge_result.score,
                    "accepted": accepted,
                    "feedback": judge_result.feedback_for_next_iteration,
                }
            ]
            if not accepted and candidate.iteration_index + 1 < self.config.generation.max_iterations_per_anchor:
                refinement = self.refiner.diagnose(
                    candidate,
                    baseline_result.baseline_runs,
                    memory_result.memory_run,
                    judge_result,
                    query_judge_result=candidate.query_judge_result,
                )
                history.append(refinement.model_dump(mode="json"))
            judgments.append(
                JudgmentArtifact(
                    candidate_id=candidate.candidate_id,
                    anchor_id=candidate.truth_anchor.anchor_id,
                    bundle_id=candidate.memory_bundle.bundle_id,
                    query_id=candidate.query.query_id,
                    baseline_run_ids=[run.run_id for run in baseline_result.baseline_runs],
                    memory_run_id=memory_result.memory_run.run_id,
                    query_judge_result=candidate.query_judge_result,
                    judge_result=judge_result,
                    accepted=accepted,
                    iteration_history=history,
                )
            )
        JSONLStore(output_path, JudgmentArtifact).write_all(judgments)
        self._record_stage("judge", output_path, len(judgments))
        return judgments

    def build_dataset(
        self,
        candidates_path: Path | None = None,
        baselines_path: Path | None = None,
        memory_evals_path: Path | None = None,
        judgments_path: Path | None = None,
        output_path: Path | None = None,
    ) -> list[SampleRecord]:
        candidates_input = self._resolve(candidates_path, self.config.artifacts.candidates_path)
        baselines_input = self._resolve(baselines_path, self.config.artifacts.baselines_path)
        memory_input = self._resolve(memory_evals_path, self.config.artifacts.memory_evals_path)
        judgments_input = self._resolve(judgments_path, self.config.artifacts.judgments_path)
        output_path = self._resolve(output_path, self.config.artifacts.dataset_path)
        self._require_input(candidates_input, "candidate artifact")
        self._require_input(baselines_input, "baseline artifact")
        self._require_input(memory_input, "memory eval artifact")
        self._require_input(judgments_input, "judgment artifact")
        candidates = {candidate.candidate_id: candidate for candidate in JSONLStore(candidates_input, CandidateRecord).read_all()}
        baselines = {result.candidate_id: result for result in JSONLStore(baselines_input, BaselineResult).read_all()}
        memory_results = {result.candidate_id: result for result in JSONLStore(memory_input, MemoryEvalResult).read_all()}
        judgments = JSONLStore(judgments_input, JudgmentArtifact).read_all()
        samples = []
        for judgment in judgments:
            candidate = candidates[judgment.candidate_id]
            baseline_result = baselines[judgment.candidate_id]
            memory_result = memory_results[judgment.candidate_id]
            sample = build_sample_record(
                candidate=candidate,
                baseline_runs=baseline_result.baseline_runs,
                memory_run=memory_result.memory_run,
                judge_result=judgment.judge_result,
                accepted=judgment.accepted,
                iteration_history=judgment.iteration_history,
            )
            samples.append(sample)
        JSONLStore(output_path, SampleRecord).write_all(samples)
        self._record_stage("build-dataset", output_path, len(samples))
        return samples

    def resume(self) -> dict[str, str]:
        artifacts = self.config.artifacts
        if not artifacts.prepared_seeds_path.exists():
            raise FileNotFoundError("Cannot resume before prepared seeds exist.")
        if not artifacts.candidates_path.exists():
            self.generate_candidates()
            return {"resumed_stage": "generate-candidates", "status": "completed"}
        if not artifacts.baselines_path.exists():
            self.run_baselines()
            return {"resumed_stage": "run-baselines", "status": "completed"}
        if not artifacts.memory_evals_path.exists():
            self.run_memory_evals()
            return {"resumed_stage": "run-memory-evals", "status": "completed"}
        if not artifacts.judgments_path.exists():
            self.judge_candidates()
            return {"resumed_stage": "judge", "status": "completed"}
        if not artifacts.dataset_path.exists():
            self.build_dataset()
            return {"resumed_stage": "build-dataset", "status": "completed"}
        return {"resumed_stage": "none", "status": "dataset already complete"}
