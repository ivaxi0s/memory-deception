from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility shim for older local runtimes
    class StrEnum(str, Enum):
        pass


class DeceptionFamily(StrEnum):
    PROTECTIVE = "protective"
    REPUTATIONAL = "reputational"
    INSTRUMENTAL = "instrumental"
    RELATIONAL = "relational"
    IDENTITY_PRESERVING = "identity_preserving"


class DistortionType(StrEnum):
    FABRICATION = "fabrication"
    OMISSION = "omission"
    MISLEADING_IMPLICATION = "misleading_implication"
    CERTAINTY_INFLATION = "certainty_inflation"
    SELECTIVE_FRAMING = "selective_framing"
    ACHIEVEMENT_INFLATION = "achievement_inflation"


class QuerySurface(StrEnum):
    FACT_LOOKUP = "fact_lookup"
    BIO_BLURB = "bio_blurb"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    PLANNING_TASK = "planning_task"
    CAPTION = "caption"
    TABLE_COMPLETION = "table_completion"


class FailureMode(StrEnum):
    QUERY_TOO_DIRECT = "query_too_direct"
    QUERY_TOO_LEADING = "query_too_leading"
    TRUTH_ANCHOR_WEAK = "truth_anchor_weak"
    BASELINE_ALREADY_DECEPTIVE = "baseline_already_deceptive"
    MEMORY_MOTIVE_WEAK = "memory_motive_weak"
    DISTORTION_IMMATERIAL = "distortion_immaterial"
    JUDGE_UNCERTAIN = "judge_uncertain"
    PARSING_FAILURE = "parsing_failure"


class MemoryCategory(StrEnum):
    DISTRACTOR = "distractor"
    PERSONA_ANCHOR = "persona_anchor"
    MOTIVE_TRIGGER = "motive_trigger"


class RunCondition(StrEnum):
    NO_MEMORY = "no_memory"
    MEMORY = "memory"
