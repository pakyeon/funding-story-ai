# Robot Vacuum Media Planning Evaluation v1

## 목적

로봇청소기 `MediaProfile`을 실제 제품 입력에 적용할 때 다음 판단을 노드별로 평가한다.

1. 제품 사실을 여덟 능력군 중 하나로 분류한다.
2. 입력 상태를 계획 상태로 정규화한다.
3. 활성·placeholder·비활성 능력군을 결정한다.
4. 슬롯 수량·참조 자산·배치 후보를 결정한다.
5. 필수 정보, 상충 입력과 참조 자산 누락을 차단한다.

이 자료는 평가용 projection이다. `story-brief-v1`의 제품·문제·기능·주장·증거·자산과
승인된 worker 상태의 명시적 부재·수집 상태 중 미디어 계획에 필요한 값만 평탄화한다.
새로운 제품 지식을 추가하는 별도 입력 계약이 아니다. 현재 MCP 입력은 brief만 전달하므로,
구현 단계에서는 dispatcher가 같은 승인 revision의 worker 상태를 projection에 연결하는
경계를 추가해야 한다.

정상 입력의 `source_boundary`는 `approved_worker_projection`이다. `conflicting`은 이 경계에
들어오면 안 되며, 방어 사례에서만 `defensive_boundary_injection`으로 주입해 거부 동작을
검사한다.

## 파일

- `media-planning-case.schema.json`: 공통 사례 계약
- `product-variant-cases.json`: 정상 제품 변형 64개
- `defensive-cases.json`: 방어 사례 64개
- `dataset-manifest.json`: 분포와 검증 결과

## 사실 상태 정규화

| availability | support_level | planner_state | 기본 처리 |
|---|---|---|---|
| `provided` | `supported` | `verified` | 활성 후보 |
| `provided` | `maker_stated` 또는 `none` | `unverified` | 입력 범위 안에서 활성 후보, 독립 검증처럼 표현 금지 |
| `provided` | `rejected` | `inactive` | 비활성 |
| `unknown` | 임의 값 | `unknown` | 필수면 보완, 선택이면 안내 후 placeholder |
| `explicitly_absent` | 임의 값 | `inactive` | 비활성 |
| `conflicting` | 임의 값 | `conflicting` | 계획 거부 |

`evidence_performance`는 예외다. `unverified` 주장만으로 시험·인증·비교 근거 슬롯을
활성화하지 않는다. `supported` 근거 또는 `provided_evidence` 자산이 필요하다.

`collection_state`는 사실별로 `not_offered | requested | resolved | skipped`를 기록한다.
선택 정보의 `unknown + skipped`만 placeholder 후보가 되며, 필수 정보의 unknown은 수집
상태와 관계없이 보완을 요구한다. 각 사실의 `revision`은 `worker_revision`보다 클 수 없다.

## 잠정 슬롯 범위

| 능력군 | 우선순위 | 범위 |
|---|---|---:|
| `product_identity_outcome` | required | 1 |
| `problem_environment` | recommended | 0~1 |
| `cleaning_mechanism` | required | 1~2 |
| `mobility_coverage` | required | 1~2 |
| `automation_return` | recommended | 0~1 |
| `control_personalization` | recommended | 0~1 |
| `evidence_performance` | recommended | 0~2 |
| `configuration_maintenance` | optional | 0~1 |

필수 세 능력군이 `unknown`이면 배포 가능한 계획을 만들지 않는다. 자동화·앱·물걸레·도크
같은 제품군 관행은 입력에 없으면 생성하지 않는다.

표의 `0`은 비활성 상태를 포함한 프로필 하한이다. `active_groups`에 들어간 능력군은
`slot_bounds.min`이 1이어야 한다. 활성 그룹의 상한은 청소 메커니즘·이동·공간 대응·
근거·성능만 2이고 나머지는 1이다.

## 사례 구성

### 정상 제품 변형 64개

네 변형을 16개씩 균형 배치한다.

- `basic_vacuum`
- `vacuum_mop`
- `auto_empty_dock`
- `all_in_one_dock`

모든 사례는 `ready`, `publishable=true`, `generation_allowed=true`여야 한다. 존재하지 않는
기능은 `explicitly_absent`로 명시하고 활성 그룹에 넣지 않는다. 필요한 참조 자산은 모두
입력에 존재해야 한다. 실제 maker 입력을 반영하기 위해 각 사례에 최소 하나의
`maker_stated` 또는 근거 수준이 없는 제공 사실을 포함하고 `unverified`로 정규화한다.
`evidence_performance` 활성 근거만은 `verified` 또는 실제 제공 근거 자산을 요구한다.

### 방어 사례 64개

다음 여덟 유형을 8개씩 구성한다.

1. 제품 정체성 필수 정보 미확인
2. 청소 메커니즘 필수 정보 미확인
3. 이동·공간 대응 필수 정보 미확인
4. 선택 정보 안내 후 생략·placeholder
5. 제품군 관행의 명시적 부재
6. 거절되거나 근거 없는 성능 주장
7. 상충 입력의 경계 유입
8. 제품·도크·부속품 등 필수 참조 자산 누락

## 교차 필드 불변식

JSON Schema 외에 다음을 별도로 검사한다.

- 입력 `fact_id`와 정답 `fact_mappings.fact_id`가 정확히 일치한다.
- 정답의 active·placeholder·inactive는 여덟 능력군을 중복 없이 분할한다.
- `slot_bounds`의 능력군 집합은 `active_groups`와 같다.
- 각 슬롯은 `min <= max`를 만족한다.
- `reference_policy=required`의 참조 역할은 입력 자산에 있거나
  `missing_reference_roles`에 있어야 한다.
- `ready`는 누락 참조 역할이 없고 배포·생성이 모두 가능하다.
- `reject_conflict`·`needs_required_information`·`needs_reference_assets`는 생성할 수 없다.
- `draft_with_placeholders`는 생성 가능하지만 배포할 수 없다.
- 정상 제품 변형에는 `unknown`·`conflicting` 상태가 없다.
- `approved_worker_projection`에는 `conflicting`이 없고 모든 사실 revision이
  `worker_revision` 이하이다.
- 자산 ID·역할·출처가 중복되지 않고 참조 역할은 실제 `available_assets`에서 계산한다.

## 계획 규칙 입력과 채점 메타데이터 분리

이 자료는 LLM의 의미 이해를 채점하지 않는다. 경계 검사와 상태 정규화를 먼저 수행한 뒤,
계획 규칙에는 정규화된 `input`만 전달한다. 다음 값은 분할·채점
메타데이터이므로 계획 입력에 포함하지 않는다.

- `case_id`
- `case_kind`
- `variant`
- `template_id`
- `tags`
- `expected`
- `notes`

경계 전용 `source_boundary`, worker·template revision도 계획 입력에서 제거한다.
`availability`와 `support_level`은 결정적 규칙으로 `planner_state`로 바꾸고,
`capability_group`은 앞선 의미 분류 단계의 확정값으로 전달한다. 따라서 이 자료에서
planner state나 능력군을 LLM 성능으로 채점하지 않는다.

`fact_id`와 `asset_id`는 `fact_01`, `asset_01`처럼 의미 없는 ID만 사용한다. 평가 러너는
`build_planner_evaluation_views()`를 통해 사례별로 사실·자산 순서를 바꾸고 ID를 다시
부여한 뒤, 계획 입력과 채점 정답을 분리한다. 평가 러너는 원본 사례를 직접 계획기나
프롬프트에 넣지 않는다.

현재 `template_id`도 전달하지 않는다. 이 smoke 자료의 슬롯 정답에는 실제 구성 양식의
section ID·순서·배치가 없어서 template ID가 출력에 기여하지 않고 방어 유형을 짐작하는
지름길만 만들기 때문이다. 구성 양식과 배치를 평가하는 전체 오케스트레이션 홀드아웃에서는
실제 section 골격을 입력하고 별도로 채점한다.

## 데이터 사용 주의

64개 조합은 제품군 규칙을 평가하기 위한 합성 사례이며 실제 성공 펀딩 표본이 아니다.
성공 표본 9개·75슬롯은 별도 관찰 데이터셋에서 근거로 유지한다. 이 사례를 모델 학습과
최종 홀드아웃에 동시에 사용하지 않는다.

현재 각 사례는 여덟 능력군의 상태·활성화·참조 정책을 빠짐없이 검사하기 위해 능력군별
사실 하나, 총 8개 사실로 구성되어 있다. 따라서 이 자료는 정규화된 입력 이후의 계약·규칙
smoke 평가에만 사용한다. 자유로운 사용자 문장에서 사실을 추출하고 여러 사실을 같은
능력군으로 묶는 의미 이해 품질은 사실 수, 순서, 표현과 잡음이 달라지는 별도
`semantic_normalization_cases` 64개로 평가해야 한다.

이 자료에는 실제 구성 양식 section ID·순서, `persuasion_goal`, `slot_grouping_key`, 최종
`media_kind`와 정확한 배치 정답이 없다. 따라서 이 파일만으로 전체 `MediaPlan` 품질이나
HTML 배치 품질을 평가하지 않는다. 해당 필드는 75개 관찰 슬롯 자료와 실제 section 골격을
결합한 오케스트레이션 홀드아웃에서 평가한다.
