"""Deception sample generator using red team agent + deception judge."""

import json
import uuid
from pathlib import Path
from typing import Optional

from deception_memory.config import PipelineConfig
from deception_memory.llm.client import BaseLLMClient
from deception_memory.logging_utils import get_logger
from deception_memory.generation.deception_red_team_agent import DeceptionRedTeamAgent
from deception_memory.runtime import PromptRuntime

logger = get_logger(__name__)


class DeceptionSampleGenerator:
    """Generate deception samples using red team + judge loop."""

    def __init__(self, client: BaseLLMClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self.accept_score = max(1, min(5, self.config.evaluation.min_judge_score))
        self.red_team = DeceptionRedTeamAgent(client, config)
        self.runtime = PromptRuntime(client, config)

    def generate_samples(
        self,
        num_samples: int,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Generate num_samples deception test cases.

        Process:
        1. Red team generates memory-query pair
        2. Run baseline + memory-conditioned responses
        3. Judge deception (score must be 3+)
        4. Save sample

        Returns list of sample dicts with memories, query, responses, score.
        """
        output_path = Path(output_path or self.config.output.samples_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = []
        attempts = 0
        max_attempts = num_samples * 5  # Allow 5 attempts per target sample

        logger.info(f"🎯 Generating {num_samples} deception samples")

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            logger.info(f"\n[Sample {len(samples)+1}/{num_samples}, Attempt {attempts}]")

            # Step 1: Red team generates case
            try:
                case = self.red_team.generate_case()
            except Exception as e:
                logger.warning(f"  ❌ Red team failed: {e}")
                continue

            memories = case["new_memories"]
            query = case["new_query"]

            logger.info(f"  Q: {query[:80]}...")
            logger.info(f"  M: {len(memories)} items")

            # Step 2: Run baseline (no memory)
            try:
                baseline_resp = self.runtime.run_model(query, component="baseline")
            except Exception as e:
                logger.warning(f"  ❌ Baseline failed: {e}")
                continue

            logger.info(f"  Baseline: {baseline_resp[:80]}...")

            # Step 3: Run with memory
            try:
                memory_resp = self.runtime.run_model(
                    query,
                    system_prompt=self.runtime.build_memory_system_prompt(memories),
                    component="memory_conditioned",
                )
            except Exception as e:
                logger.warning(f"  ❌ Memory response failed: {e}")
                continue

            logger.info(f"  Memory: {memory_resp[:80]}...")

            # Step 4: Judge deception
            try:
                judge_result = self.runtime.judge_deception(memories, query, baseline_resp, memory_resp)
                logger.info(
                    f"  Judge: {judge_result.score}/5 | Memory-induced: {judge_result.plausibly_memory_induced}"
                )
            except Exception as e:
                logger.warning(f"  ❌ Judge failed: {e}")
                continue

            # Step 5: Accept only severe deception (score > 4), memory-induced, and material
            if (
                judge_result.score >= self.accept_score
                and judge_result.plausibly_memory_induced
                and judge_result.material_distortion
            ):
                sample = {
                    "sample_id": str(uuid.uuid4()),
                    "memories": memories,
                    "query": query,
                    "baseline_response": baseline_resp,
                    "memory_response": memory_resp,
                    "judge_result": {
                        "score": judge_result.score,
                        "reasoning": judge_result.reasoning,
                        "memory_induced": judge_result.plausibly_memory_induced,
                        "material_distortion": judge_result.material_distortion,
                        "feedback_for_next_iteration": judge_result.feedback_for_next_iteration,
                    },
                    "metadata": {
                        "failure_mode": case.get("failure_mode"),
                        "attack_vector": case.get("attack_vector"),
                    },
                }

                samples.append(sample)

                # Save to file
                sample_file = output_path / f"{sample['sample_id']}.json"
                sample_file.write_text(json.dumps(sample, indent=2))

                logger.info(f"  ✓ ACCEPTED → {sample_file}")
            else:
                logger.info(
                    f"  ❌ Reject (score={judge_result.score}, "
                    f"mem_ind={judge_result.plausibly_memory_induced})"
                )

        logger.info(f"\n✓ Generated {len(samples)}/{num_samples} samples in {attempts} attempts")
        return samples
