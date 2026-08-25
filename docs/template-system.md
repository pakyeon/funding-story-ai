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

| ID | 설득 축 | 영역 | 이미지 |
|---|---|---:|---:|
| `t01_performance_value_evidence` | 성능·가치·증빙 | 13 | 5 |
| `t02_problem_solution_automation` | 문제–해결·자동화 | 11 | 5 |
| `t03_lifestyle_social_proof` | 사용 상황·사회적 증거 | 10 | 5 |
| `t04_full_campaign` | 균형 잡힌 전체 캠페인 | 12 | 6 |
| `t05_value_practical_full_campaign` | 가성비·실용 | 11 | 5 |
| `t06_trust_maintenance_full_campaign` | 신뢰·유지관리 | 11 | 5 |

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
