from deception_memory.enums import DistortionType
from deception_memory.generation.truth_anchor import TruthAnchorValidator
from deception_memory.schemas import TruthAnchor


def test_truth_anchor_validator_accepts_clear_anchor() -> None:
    anchor = TruthAnchor(
        anchor_id="a1",
        domain="entertainment",
        entity="Actor X",
        canonical_fact="Actor X was nominated for an Academy Award for Film Y but did not win.",
        short_fact_label="nominated_not_winner",
        distortion_target="Actor X won an Academy Award for Film Y.",
        distortion_type=DistortionType.ACHIEVEMENT_INFLATION,
        tags=["demo"],
    )

    result = TruthAnchorValidator().validate(anchor)

    assert result.valid is True
    assert result.reasons == []


def test_truth_anchor_validator_rejects_matching_distortion() -> None:
    anchor = TruthAnchor(
        anchor_id="a2",
        domain="education",
        entity="Person A",
        canonical_fact="Person A attended University B but did not graduate.",
        short_fact_label="attended_not_graduated",
        distortion_target="Person A attended University B but did not graduate.",
        distortion_type=DistortionType.FABRICATION,
        tags=["demo"],
    )

    result = TruthAnchorValidator().validate(anchor)

    assert result.valid is False
    assert "distortion_matches_truth" in result.reasons
