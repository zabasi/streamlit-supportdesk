"""retrieval.concept_weight: the concept queries' rankings can count less than the report's in the fusion."""
import dataclasses

import pytest

from supportdesk.config import ConfigError, Settings
from supportdesk.pipelines.advanced import WeightedFusionPipeline
from supportdesk.registry import make_pipeline
from supportdesk.retrieval.fusion import rrf, weighted_rrf
from supportdesk.types import CaseInput

REPORT = ["r1", "r2", "r3", "shared"]
CONCEPT = ["shared", "c1", "c2", "c3"]


def test_weights_of_one_are_plain_rrf():
    assert weighted_rrf([REPORT, CONCEPT], [1.0, 1.0], k=60) == rrf([REPORT, CONCEPT], k=60)


def test_a_lower_concept_weight_keeps_the_report_ahead_but_still_admits_concept_pages():
    equal = [cid for cid, _ in weighted_rrf([REPORT, CONCEPT], [1.0, 1.0], k=60)]
    lower = [cid for cid, _ in weighted_rrf([REPORT, CONCEPT], [1.0, 0.5], k=60)]
    assert equal.index("c1") < equal.index("r2")  # equal: a concept-only page overtakes the report's matches
    assert lower.index("r3") < lower.index("c1")  # lower: the report's matches keep the lead
    assert "c1" in lower[:6]  # and the concept-only page is still among the candidates
    with pytest.raises(ValueError):
        weighted_rrf([REPORT, CONCEPT], [1.0], k=60)


def test_advanced_runs_with_a_concept_weight(index, clients, settings):
    s = dataclasses.replace(settings,
                            pipeline=dataclasses.replace(settings.pipeline, name="advanced", scope_gate="off"),
                            retrieval=dataclasses.replace(settings.retrieval, concept_weight=0.5,
                                                          keep_report_query=True, max_queries=3))
    pipeline = make_pipeline("advanced", "dense_doc_slots", index, clients, s)
    assert isinstance(pipeline, WeightedFusionPipeline)
    assert pipeline.run(CaseInput(case_id="c", title="Session state lost", body="st.session_state lost")).retrieved


def test_concept_weight_is_a_checked_setting():
    assert Settings.load(env={}).retrieval.concept_weight == 1.0  # default: plain RRF, advanced unchanged
    with pytest.raises(ConfigError, match="concept_weight"):
        Settings.load(env={}, overrides=["retrieval.concept_weight=1.5"])
