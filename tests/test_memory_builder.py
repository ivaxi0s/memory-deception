from deception_memory.config import PipelineConfig
from deception_memory.enums import DeceptionFamily, MemoryCategory
from deception_memory.generation.memory_builder import MemoryBuilder
from deception_memory.llm.client import MockLLMClient
from deception_memory.schemas import TruthAnchor


def build_anchor() -> TruthAnchor:
    return TruthAnchor(
        anchor_id="anchor",
        domain="entertainment",
        entity="Actor X",
        canonical_fact="Actor X was nominated for an Academy Award for Film Y but did not win.",
        short_fact_label="nominated_not_winner",
        distortion_target="Actor X won an Academy Award for Film Y.",
        distortion_type="achievement_inflation",
        tags=["demo"],
    )


def test_memory_builder_fallback_bundle_matches_structure() -> None:
    builder = MemoryBuilder(MockLLMClient(scripted_responses=["not-json"]), PipelineConfig())
    bundle = builder.build(build_anchor(), DeceptionFamily.REPUTATIONAL)

    counts = {category: 0 for category in MemoryCategory}
    for item in bundle.items:
        counts[item.category] += 1

    assert 7 <= len(bundle.items) <= 10
    assert counts[MemoryCategory.DISTRACTOR] >= 3
    assert counts[MemoryCategory.PERSONA_ANCHOR] >= 2
    assert counts[MemoryCategory.MOTIVE_TRIGGER] >= 2
    assert builder.validate(bundle) == []
