from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "evals" / "build_semantic_adjudication_report.py"
SPEC = spec_from_file_location("semantic_adjudication_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
report_builder = module_from_spec(SPEC)
SPEC.loader.exec_module(report_builder)


def test_double_annotation_packet_has_complete_coverage_and_explicit_human_boundary() -> None:
    report = report_builder.build_report()

    assert report["scope"]["case_count"] == 16
    assert report["scope"]["fact_count"] == 147
    assert report["scope"]["annotation_type"] == "independent_ai_double_annotation"
    assert report["scope"]["human_annotation_claimed"] is False
    assert report["provisional_adjudication"]["status"] == "human_signoff_complete"
    signoff = report["provisional_adjudication"]["human_signoff"]
    assert signoff["decision"] == "approved"
    assert signoff["human_reannotation_claimed"] is False


def test_provisional_gold_includes_the_jointly_identified_correction() -> None:
    report = report_builder.build_report()
    change = report["provisional_adjudication"]["gold_changes"][0]

    assert change["case_id"] == "rv_semantic_normal_006"
    assert change["before"] == "evidence_performance"
    assert change["after"] == "configuration_maintenance"
    assert report["agreement"]["decision_exact_rate"] == 1.0
