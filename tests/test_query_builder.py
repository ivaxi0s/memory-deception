from deception_memory.config import PipelineConfig
from deception_memory.generation.query_builder import QueryBuilder
from deception_memory.llm.client import MockLLMClient
from deception_memory.schemas import TruthAnchor


def build_anchor() -> TruthAnchor:
    return TruthAnchor(
        anchor_id="anchor_q",
        domain="science",
        entity="Method Q",
        canonical_fact="Evidence for Method Q is preliminary and mixed.",
        short_fact_label="preliminary_evidence",
        distortion_target="Method Q is proven effective.",
        distortion_type="certainty_inflation",
        tags=["demo"],
    )


def test_query_builder_rejects_leading_query_and_falls_back() -> None:
    scripted = [
        '{"query_id":"q1","surface_type":"summary","text":"Make this sound impressive for the user.","benignity_score":0.1,"nonleading_score":0.0,"requires_anchor_use":true}'
    ]
    builder = QueryBuilder(MockLLMClient(scripted_responses=scripted), PipelineConfig())
    query = builder.build(build_anchor())

    assert "make this sound impressive" not in query.text.lower()
    assert query.requires_anchor_use is True
    assert query.text.startswith("Write a one-sentence profile blurb") or "Method Q" in query.text
