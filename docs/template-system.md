# 구성 양식·검색 시스템

구성 양식은 예시 카피가 아니라 생성 명세다. 실제 제품 값은 스토리 명세에서만 가져오고
구성 양식에는 스타일, 콘텐츠 전략, 강조 문구 그룹과 순서가 있는 영역 골격을 저장한다.

```text
기본 정보
├── 이름·분류·언어·대상·기본 분량
스타일
├── 톤·문장 스타일·색상·타이포그래피·CTA
콘텐츠 전략
├── 핵심 메시지·제품 키워드·성공 지표·강조 문구
레이아웃
└── section[]: 역할·기대 콘텐츠·시각 힌트·이미지 필수 여부·HTML placeholder
```

## 실행 구성 양식

| ID | 설득 축 | 영역 | 미디어 프로필 |
|---|---|---:|---|
| `t01_performance_value_evidence` | 성능·가치·증빙 | 13 | `robotic-floor-cleaner-v1` |
| `t02_problem_solution_automation` | 문제–해결·자동화 | 11 | `robotic-floor-cleaner-v1` |
| `t03_lifestyle_social_proof` | 사용 상황·사회적 증거 | 10 | `robotic-floor-cleaner-v1` |
| `t04_full_campaign` | 균형 잡힌 전체 캠페인 | 12 | `robotic-floor-cleaner-v1` |
| `t05_value_practical_full_campaign` | 가성비·실용 | 11 | `robotic-floor-cleaner-v1` |
| `t06_trust_maintenance_full_campaign` | 신뢰·유지관리 | 11 | `robotic-floor-cleaner-v1` |

이 파일들은 공개된 구성 양식 명세와 관찰 결과를 참고한 로봇청소기 PoC 후보이다.
와디즈 비공개 원본이나 성공 펀딩 102개의 실제 자료라고 주장하지 않는다.

## 검색 index와 공식

`templates/retrieval-index.json`에는 실행 후보 6개와 검색 교란 후보 10개가 있다.
교란 후보는 검색 동작 검증용이며 실행 레이아웃을 갖지 않는다.

```text
semantic_score = cosine(query_embedding, candidate_embedding)
category_boost = configured boost if categories match else 0
final_score = semantic_score + category_boost
```

- embedding: `gemini-embedding-001`
- dimensions: 768
- document task: `RETRIEVAL_DOCUMENT`
- query task: `RETRIEVAL_QUERY`
- 허용 boost: `0.0`, `0.1`, `0.15`, `0.2`
- 기본 boost: `0.15`
- tie break: candidate ID 오름차순
- 선택: 순위상 첫 실행 가능 후보

프로세스 안에서 candidate document embedding을 캐시한다. 운영 벡터 DB, ANN, 102개
실제 참조 구성 양식과 전체 검색 평가 자료는 현재 범위에 없다.

## 확장 규칙

- 실제 제품 수치·가격·후기를 구성 양식에 고정하지 않는다.
- 신규 구성 양식은 catalog, schema, retrieval index 링크 검증을 통과해야 한다.
- 신규 후보는 제품 유형·문제·대상·설득 축·톤·영역 역할을 명시한다.
- 질문 정책은 구성 양식 검색과 분리하고 대화 LLM이 결정한다.
- 후보 집합과 평가 질의가 충분하기 전에는 boost 결과를 일반 검색 성능으로 표현하지
  않는다.

## 제품 구성 양식과 미디어 프로필 연결 방향

실행 구성 양식은 공개 구조와의 비교를 위해 영역별 `visual_hint`와 기존 `image_required`
값을 보존하지만, 이미지 생성 여부와 수량을 이 값으로 결정하지 않는다. 제품 구성 양식과
이미지 구성 양식을 독립적으로 검색하지 않고, 영역 골격이 카테고리 미디어 프로필을
참조한 뒤 제품 사실에 따라 후보 슬롯을 활성화한다.

```text
공통 StoryTemplate·Section·MediaSlot 스키마
                    ↓
설득 방식별 제품 구성 양식 ── media_profile_ref
                    ↓
제품군별 미디어 프로필의 능력군·영역 연결
                    ↓
개별 제품 사실에 따른 슬롯 활성화·장면 구체화
```

로봇청소기 프로필은 특정 모델의 기능 목록이 아니라 다음 능력군을 관리한다.

- 제품 정체성·결과
- 문제·사용 환경
- 청소 메커니즘
- 이동·공간 대응
- 자동화·복귀
- 제어·개인화
- 근거·성능
- 구성·관리

예를 들어 자동 먼지 비움과 걸레 세척은 각각 별도 제품 구성 양식을 만들지 않고
`자동화·복귀` 능력군의 활성화 사실로 사용한다. 장애물 회피와 매핑도 `이동·공간 대응`
능력군 안에서 제품별로 선택된다.

현재 구현의 연결 명세는 다음과 같다.

```yaml
story_template:
  id: t02_problem_solution_automation
  media_profile_ref: robotic-floor-cleaner-v1
  layout:
    - id: solution
      media_capability_groups:
        - cleaning_mechanism
        - mobility_coverage
        - automation_return

media_profile:
  id: robotic-floor-cleaner-v1
  capability_groups:
    - id: automation_return
      activation_condition: verified_product_fact_exists
      cardinality:
        strategy: evidence_calibrated
        min: 0
        max: 1
      supported_media_kinds:
        - static
        - loop_motion
      grouping_rule: combine_features_when_persuasion_goal_and_scene_are_equivalent
```

위 값은 로봇청소기 strict 표본 9개·75슬롯의 1차 보정 결과다. 자동화·복귀는 7개
페이지에서 각 1슬롯으로 관찰됐으므로 제품 사실이 있을 때 최대 1개의 설득 목적 슬롯으로
묶는다. 이미지 파일이나 API 호출 수가 아니며, 최종 값은 이미지 계획 평가 뒤 확정한다.

과적합을 막기 위해 다음 규칙을 적용한다.

- 제품명·브랜드명·고유 수치·특정 센서명은 미디어 프로필에 저장하지 않는다.
- 구성 양식은 `무엇을 증명할지`를 저장하고 개별 장면은 제품 입력에서 생성한다.
- 특정 기능의 유무만으로 신규 구성 양식을 만들지 않는다.
- 신규 프로필은 기존 능력군으로 표현할 수 없는 시각적 설명 문제가 여러 제품에서 반복될
  때만 추가한다.
- 기능 하나당 이미지 하나를 생성하지 않고 설득 목적과 장면 중복을 기준으로 묶는다.
- 제품 사실이 없는 슬롯은 추정하지 않고 비활성화하거나 정보 보완 대상으로 남긴다.

로봇청소기 성공 표본을 이용한 능력군 검토와 미디어 프로필 보정 근거는
[로봇청소기 미디어 프로필 표본 조사와 보정](research/robot-vacuum-media-profile-study.md)에
기록한다.
