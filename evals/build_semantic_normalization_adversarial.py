from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

EVAL_ROOT = Path(__file__).parent
VALIDATOR_PATH = EVAL_ROOT / "validate_semantic_normalization_adversarial.py"
SPEC = importlib.util.spec_from_file_location("semantic_adversarial_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-adversarial-v1"
CASES_PATH = ROOT / "adversarial-cases.json"
DEFAULT_MANIFEST = ROOT / "dataset-manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and seal adversarial semantic cases")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    cases = validator.base._load(CASES_PATH)
    summary = validator.validate_adversarial(cases)
    manifest = {
        "schema_version": "robot-vacuum-semantic-normalization-adversarial-manifest-v1",
        "dataset_status": "synthetic_adversarial_fixture",
        "observed_attack_data": False,
        "purpose": (
            "명령처럼 보이는 승인 데이터와 여러 위치의 위조 참조를 분리해 경계·의미 "
            "처리를 검증한다."
        ),
        "summary": summary,
        "limitations": [
            "합성 공격 문자열이며 실제 공격 로그나 완전한 보안 평가가 아니다.",
            "도구 실행 권한 탈취, 인코딩 우회, 다중 턴 간접 주입은 포함하지 않는다.",
            "accepted injection 사례의 제어 문자열은 의미 gold에서 제외한다.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
