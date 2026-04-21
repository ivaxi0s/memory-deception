from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from deception_memory.config import load_config
from deception_memory.generation.simple_sample_generator import DeceptionSampleGenerator
from deception_memory.llm.caching import LLMCache
from deception_memory.llm.client import MockLLMClient, OpenAIClient
from deception_memory.logging_utils import configure_logging


def build_client(config_path: Path, require_llm: bool):
    config = load_config(config_path)
    configure_logging(config.output.pretty_logs)
    cache = LLMCache(config.output.cache_dir)
    if not require_llm:
        return config, MockLLMClient(cache=cache)
    if config.models.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        return config, OpenAIClient(cache=cache)
    return config, MockLLMClient(cache=cache)


def generate_samples(config_path: Path, num_samples: int, output_path: Path | None = None) -> None:
    config, client = build_client(config_path, require_llm=True)
    generator = DeceptionSampleGenerator(client, config)
    destination = output_path or config.output.samples_dir
    samples = generator.generate_samples(num_samples, destination)
    print(f"\nGenerated {len(samples)} samples -> {destination}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Generate memory-conditioned deception samples by running prompt loops.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    samples = subparsers.add_parser("generate-samples", help="Generate N samples with the simple loop")
    samples.add_argument("num_samples", type=int, help="Number of samples to generate")
    samples.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    samples.add_argument("--output-path", type=Path, default=None, help="Output directory or JSON file")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        generate_samples(args.config, args.num_samples, output_path=args.output_path)
        return


if __name__ == "__main__":
    main()
