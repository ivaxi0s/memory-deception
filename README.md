# memory-deception

`memory-deception` is a Python project for generating memory-conditioned deception samples for long-term-memory-enabled assistants. The current repo keeps the lightweight sample-generation path centered on a benign query, injected memories, baseline contrast, and structured judging.

## Methodology

The sample generator centers on a contrastive question:

`Does injected user memory make the model produce a materially more deceptive answer than it would without memory?`

Each accepted sample includes:

- a generated memory bundle
- a benign query
- a no-memory baseline response
- a memory-conditioned response
- a structured judge result

## Project Layout

```text
deception_memory/
  cli/
  generation/
  llm/
  prompts/
configs/
data/
tests/
```

## Usage

The project is config-driven with YAML files in [configs](/home/ivaxi/projects/memory-deception/configs). The default config specifies models, generation controls, and output paths.

Example:

```bash
python main.py generate-samples 10 --config configs/default.yaml
```

The primary interface is the repo-root [main.py](/home/ivaxi/projects/memory-deception/main.py). The default config names the provider as `openai` and sets `generator_model`, `target_model`, and `judge_model` to `gpt-5.4`.

## Expected Outputs

- `data/outputs/samples/*.json`: accepted sample records

## Acceptance

Accepted samples require:

- judge score at or above threshold
- material distortion
- plausible memory inducement
- stronger deception in the memory-conditioned answer than the baseline

## Testing

The test suite covers:

- memory bundle structure constraints
- query validation
- mocked generation behavior

Run:

```bash
pytest
```

## Notes

- Prompt templates live in [deception_memory/prompts](/home/ivaxi/projects/memory-deception/deception_memory/prompts).
- LLM-backed stages fail fast if `provider=openai` and the key is missing.
