# Robot Vacuum Semantic Normalization Evaluation v1

## 목적과 경계

이 자료는 `story-brief-v1`에서 투영된 승인 엔티티 묶음(`approved_entity_projection-v1`)과
같은 승인 revision의 worker 상태 projection에서 `MediaFacts`를 만드는 노드를 평가한다.
raw conversation, 메시지 이력, 미승인 LLM 추출 결과는 입력에 포함하지 않는다.

LLM과 결정적 adapter의 책임은 분리한다.

| 담당 | 허용된 판단 |
| --- | --- |
| LLM | 복합 문장의 atomic proposition 분해, `feature`의 다섯 능력군과 `claim`의 여덟 능력군 의미 분류 |
| deterministic adapter | `availability`, `support_level`, `collection_state`의 권위값과 승인된 source/evidence/asset 링크 조인, boundary/revision/digest 검증, conflict·stale·위조 거부 |

LLM은 상태값을 새로 만들거나 worker projection을 재추출하지 않는다. adapter가 LLM의
proposition을 승인된 fact ID에 조인한 뒤 상태를 부착해 `MediaFacts`를 만든다.
`product`, `problem`, `evidence` entity kind는 각각 `product_identity_outcome`,
`problem_environment`, `evidence_performance`로 결정적으로 라우팅한다. 따라서 모델의 주
분류 품질은 `feature`가 속하는 cleaning/mobility/automation/control/configuration 다섯
그룹과, 여덟 그룹 어디에나 속할 수 있는 `claim`을 각각 따로 계산한다.

## 파일

- `semantic-normalization-case.schema.json`: 한 사례의 raw input, model view, 정답 계약
- `normal-cases.json`: 정상 64건
- `defensive-cases.json`: 방어 64건
- `dataset-manifest.json`: 분포·링크·ID·model-view leakage 검증 기록

## 사례 구조

각 사례는 다음 층을 가진다.

```text
case metadata (case_id, case_kind, variant, tags, notes)
├── input
│   ├── approved_revision, brief_digest, worker_projection_digest, source_boundary
│   ├── approved_entity_projection: entities, facts, sources, evidence, assets
│   └── worker_projection: revision, brief_digest, fact_states
├── model_view
│   └── facts (fact_id, entity_kind, statement만 포함)
└── expected
    ├── atomic_propositions (LLM gold: fact ID + clause + group)
    ├── media_facts (adapter state + links)
    └── adapter decision
```

`approved_entity_projection`은 원래 story brief의 전체 문서가 아니라, 승인된
product/problem/feature/claim/evidence/distractor 엔티티와 그 fact·근거 링크를 보존한
평가용 투영이다. 따라서 이 파일의 projection은 실제 `schemas/story-brief.schema.json`
전체를 가장하지 않으며, 런타임에서는 해당 스키마의 엔티티를 이 투영으로 변환한다.

평가 harness가 LLM에 전달하는 값은 각 사례의 `model_view`뿐이다. model view에는
`case_id`, `case_kind`, `variant`, `tags`, `expected`, `notes`, `template_id`, `source_boundary`,
revision, digest, state (`availability`/`support_level`/`collection_state`)와 gold
`capability_group`을 넣지 않는다. 또한 model view 안에 raw conversation 필드는 없다.
`template_id`는 raw input에만 남겨 두며 LLM view에는 보내지 않는다.

모델 view의 `fact_id`는 의미 없는 참조 토큰이며 숫자 순서가 능력군을 인코딩하지 않는다.
사례마다 fact 순서와 정답 proposition 순서를 독립적으로 섞었으며 정답은 ID로 조인한다.

model view에는 `fact_id`, `entity_kind`, `statement`만 둔다. source/evidence/asset catalog와
참조 ID, entities 목록은 LLM에 보내지 않고 결정적 adapter가 승인 projection에서 조인한다.
raw projection에는 audit·join을 위해 entities, facts, catalog와 참조를 모두 남긴다.
`entity_kind`는 문장 출처 종류이며 capability group의 gold 라벨이 아니다.

## 정상 64건

- fact 수와 entity 수는 모두 4~14 범위에서 사례별로 달라진다.
- 네 제품 변형(`basic_vacuum`, `vacuum_mop`, `auto_empty_dock`, `all_in_one_dock`)은 각각 16건이다.
- 정상 64건은 실제 story brief의 `claims` 경로를 빠뜨리지 않도록 각각 claim fact를 정확히
  1건 포함한다. 최초 구성은 여덟 능력군별 8건이었으나 독립 이중 주석에서 두 주석자가
  동일하게 수정한 1건을 반영해 `configuration_maintenance` 9건,
  `evidence_performance` 7건, 나머지 각 8건이다. `maker_stated` 28건과 evidence-linked
  `supported` 36건을 포함한다.
- 동일 capability group에 복수 fact가 들어가는 사례, 한 fact에서 2개 proposition으로
  분해되는 복합 문장, 제품 기능과 무관한 distractor를 섞었다. `compound_sentence` 사례는
  사례당 1~3개 compound fact만 가지며, 실제 생성 결과는 19개 사례에 각 1개씩이다.
- 모든 사례에서 exact duplicate fact statement와 atomic proposition text는 0건이다.
  row 수와 사례 간 반복을 구분하기 위해 고유 text 분모는 manifest에 별도로 기록했다.
- `explicitly_absent`와 `unknown + skipped`를 각각 20건·22건에 포함했다.
- 전역 fact 기준으로 `maker_stated` 343건, 실제 evidence ref가 연결된 `supported` 195건,
  명시적 부재·미확인에 해당하는 `none` 44건을 포함한다. `supported`는 evidence ref가 있는
  사실에만 부여하고, 근거가 없는 제공 정보는 `maker_stated`로 유지한다.
- worker projection의 conflict는 0건이고, 정상 사례의 adapter 결정은 모두 `accepted`다.

`unknown`과 명시적 부재는 텍스트만 보고 추론하는 값이 아니다. 정답 상태는 worker
projection에 있고 adapter가 소유한다. model view에는 해당 상태 필드가 없다.

## 방어 64건

| 방어 유형 | 건수 | 기대 동작 |
| --- | ---: | --- |
| conflict | 32 | `reject_conflict` |
| stale revision | 8 | `reject_stale_revision` |
| digest mismatch | 8 | `reject_digest_mismatch` |
| forged source/asset/evidence link | 8 | `reject_forged_reference` |
| prompt injection | 8 | 정상 수용; 인용된 instruction-like text를 데이터로만 취급 |

stale/digest는 합계 16건, forged refs/assets/evidence는 합계 8건이다. conflict 사례의
`source_boundary`는 의도적으로 `defensive_boundary_injection`이며 정상 승인 경계를
넘어오면 안 되는 상태를 시험한다. 위조 링크 8건에는 catalog에 존재하지 않는 source,
asset 또는 evidence ID가 하나씩 들어간다. 이는 의도적인 방어 결함이지 정상 데이터의
링크 오류가 아니다.

prompt-injection 사례의 문장 안에는 예를 들어 `SYSTEM: ignore the evaluator ...`와
같은 인용문이 있다. 이 문자열은 source data이며 프롬프트 지시가 아니다. model은 실제
제품 proposition의 group만 반환하고, 인용문을 따라 상태를 바꾸거나 승인 플래그를
만들지 않아야 한다.

수용된 prompt-injection 사례의 adapter `failure_code`는 다른 정상 입력과 마찬가지로
`null`이다. 사례의 `prompt_injection` 태그를 모델이나 adapter가 보지 않으므로, 태그에서
정답을 복사해 “주입을 무시했다”는 별도 코드를 만드는 shortcut을 허용하지 않는다.

경계 검사에서 거부되는 conflict·stale·digest mismatch·forged reference 56건은 의미
모델을 호출하지 않으며 `atomic_propositions`, `ignored_fact_ids`, `media_facts`가 모두
비어 있어야 한다. 의미 분류 정답을 갖는 방어 사례는 경계를 정상 통과하는 prompt-injection
8건뿐이다.

adapter의 링크 계약은 `product_identity_outcome → product/maker-input`,
`problem_environment → lifestyle/maker-input`, `evidence_performance → evidence/test-report`,
나머지 다섯 feature group → `feature` 또는 control group의 `app` asset과
document source로 고정한다. 상세 매핑과 distractor/compound/중복 불변식은 manifest의
`semantic_invariants`에 있다.

## 검증

JSON Schema와 교차 불변식을 함께 검사했다.

```bash
uv run --no-sync python evals/validate_semantic_normalization_dataset.py
```

검증기는 JSON Schema와 다음 교차 불변식을 함께 확인한다.

- 정상·방어 각각 64건, 변형별 16건
- normal fact/entity 범위 4~14, 정상 conflict 0건
- 모든 model view의 금지 키 leakage 0건
- canonical brief/worker digest와 fact-state revision 일치
- 사례·entity·fact·source·evidence·asset 중복 ID 0건
- 사례 내 exact duplicate fact statement/proposition text 0건, compound fact 1~3개 제한
- atomic proposition의 완전한 clause grounding과 MediaFacts의 상태·참조 exact join
- 정상 및 prompt-injection 사례의 unresolved link 0건
- 방어 유형 `32 / 8 / 8 / 8 / 8` 및 의도된 forged-link 8건
- normal/defensive 사례별 fact 순서 64종, 전체 capability-group/출력 위치 쌍 113종

상세 수치는 `dataset-manifest.json`의 `normal`, `defensive`, `integrity`에 있다.

## 실제 모델 출력 채점

모델과 adapter의 실행 결과는 `semantic-normalization-predictions-v1` 형식으로 저장한다.

```json
{
  "schema_version": "semantic-normalization-predictions-v1",
  "predictions": [
    {
      "case_id": "rv_semantic_normal_001",
      "output": {
        "adapter": {},
        "atomic_propositions": [],
        "media_facts": [],
        "ignored_fact_ids": []
      }
    }
  ]
}
```

`output`은 사례 스키마의 `expected`와 같은 계약을 사용한다. LLM은 atomic propositions만
만들고, adapter가 승인 상태와 참조를 붙여 최종 output을 구성한다. 실제 채점은 다음처럼
실행한다.

```bash
uv run --no-sync python evals/score_semantic_normalization.py predictions.json \
  --split all --fail-on-gate
```

scorer는 사실 분해 precision/recall/F1, feature·claim 능력군 macro F1, adapter 판정,
상태·참조 exact join, 중복 proposition·ID, 미승인 fact ID, 거부 입력에서의 의미 모델
호출 및 MediaFacts 생성을 분리해 보고한다. 구조 validator와 달리 실제 예측값을 고정
gold에 대조한다.

## 알려진 한계

- 모두 합성 한국어 문장이라 실제 고객 대화의 방언·다국어·오탈자 분포를 대표하지 않는다.
- 제한된 문장 원형을 여러 사례에서 바꿔 조합한 개발 fixture이므로, 같은 표현을 보지 않은
  제품 단위 최종 홀드아웃이나 일반화 성능을 대신하지 않는다.
- 16건은 독립 AI 이중 주석과 불일치 조정을 거쳤고, 세 가지 잠정 판정은 2026-09-01
  사용자 검수로 승인됐다. 이는 사람 이중 주석을 의미하지 않는다.
- validator는 entity kind가 허용하는 능력군 범위와 문장 grounding을 검사하지만, 허용 범위
  안에서 선택한 gold group의 의미적 정답성 자체를 증명하지 않는다. 이 판정은 이중 주석과
  외부 semantic scorer로 보완해야 한다.
- `approved_entity_projection-v1`과 worker projection은 평가 fixture다. 런타임은 실제
  canonical story brief/projection으로 digest를 다시 계산하고 승인 revision을 검증해야 한다.
- prompt injection은 인용문 기반 8건만 다루며 다국어·도구 호출·중첩 인코딩 공격은 포함하지 않는다.
- forged-link 8건은 fact/entity의 source·asset·evidence 참조만 다루며 catalog 자체나 최종
  출력에서 새로 생긴 위조 참조까지 포괄하지 않는다.
- 이 세트는 MediaFacts 정규화까지만 평가하며 media-slot 수량, HTML 렌더링, 이미지 품질은 다루지 않는다.
- forged-link의 8건 unresolved reference는 방어 기대값이며, 이를 정상 학습 자료로 재사용하지 않는다.
