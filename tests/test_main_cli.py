from pathlib import Path

from deception_memory.cli.main import main


def test_main_prepare_seeds_runs_without_openai_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prepared_path = tmp_path / "prepared.jsonl"
    main(
        [
            "prepare-seeds",
            "--config",
            "configs/default.yaml",
            "--raw-seeds-path",
            "data/seeds/demo_seeds.jsonl",
            "--prepared-seeds-path",
            str(prepared_path),
        ]
    )
    assert prepared_path.exists()


def test_main_requires_key_for_llm_stage(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        main(["generate-candidates", "--config", "configs/default.yaml"])
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing OPENAI_API_KEY")
