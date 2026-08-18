# 템플릿·검색 시스템

템플릿은 예시 카피가 아니라 생성 명세다. 스타일·콘텐츠 전략과 순서가 있는 섹션
골격을 관리하고 실제 제품 값은 브리프에서만 가져온다.

## 템플릿 명세

```text
기본 정보
├── 이름·카테고리·언어·타깃
├── 기본 분량·내레이션 모드
스타일
├── 톤·문장 스타일·컬러 팔레트
├── 타이포그래피·긴급성·CTA 방식
콘텐츠 전략
├── 핵심 메시지·제품 키워드
├── 성공 지표·강조 문구 그룹
레이아웃
└── 순서가 있는 section[]
```

각 section에는 역할, 기대 콘텐츠, 시각 힌트, 콘텐츠 타입, 이미지 필수 여부와 편집
가능한 HTML placeholder가 있다. 6개 실행 템플릿은 모두 아래 12개 결과 역할을 쓴다.

```text
hero → problem → solution → comparison → features → social_proof
→ funding_plan → platform_choice → timeline → team → risks → cta
```

공통 결과 역할은 출력 비교와 편집기를 단순화하고, 템플릿 차이는 콘텐츠 전략·강조
문구·톤·시각 무드로 표현한다.

## 실행 템플릿

| ID | 설득 축 | 섹션 |
|---|---|---:|
| `t01_performance_value_evidence` | 성능·가치·증빙 | 12 |
| `t02_problem_solution_automation` | 문제–해결·자동화 | 12 |
| `t03_lifestyle_social_proof` | 사용 상황·사회적 증거 | 12 |
| `t04_full_campaign` | 균형 잡힌 전체 캠페인 | 12 |
| `t05_value_practical_full_campaign` | 가성비·실용 | 12 |
| `t06_trust_maintenance_full_campaign` | 신뢰·유지관리 | 12 |

카탈로그의 프로젝트·관찰 ID는 구조 연구 provenance다. 해당 구성이 실제 성과를
만들었다거나 외부 서비스의 내부 템플릿 원본이라는 뜻이 아니다.

## 검색 index

`templates/retrieval-index.json`에는 16후보가 있다.

- 실행 템플릿 6개
- 같은 카테고리 hard negative 6개
- 다른 카테고리 negative 4개

negative는 축소된 데이터에서 category boost가 의미 검색을 어떻게 보정하거나 왜곡하는지
확인하기 위한 검색 후보이며 생성할 템플릿 파일은 없다.

## 선택 공식

```text
semantic_score = cosine(query_embedding, candidate_embedding)
category_boost = configured boost if categories match else 0
final_score = semantic_score + category_boost
```

- embedding: `gemini-embedding-001`
- dimensions: 768
- document task: `RETRIEVAL_DOCUMENT`
- query task: `RETRIEVAL_QUERY`
- 허용 boost: `0.0`, `0.1`, `0.2`
- tie break: candidate ID 오름차순

프로세스 안에서는 candidate document embedding을 캐시한다. 운영 벡터 DB, ANN,
오프라인 embedding artifact와 자동 파라미터 학습은 현재 범위 밖이다.

## 직접 선택기와의 차이

`funding-story generate --dry-run`은 API를 호출하지 않기 위해 기존의 설명 가능한 규칙
선택기를 쓴다. 전체 MCP 서버 경로는 Gemini embedding KNN을 사용한다. 두 선택기는
동일한 `StoryTemplateSelector` protocol을 구현하므로 실행 파이프라인은 선택 방식에
의존하지 않는다.

## 확장 규칙

- 실제 제품 수치·가격·후기를 템플릿에 고정하지 않는다.
- 신규 템플릿은 catalog, schema와 retrieval index 링크 검증을 통과해야 한다.
- 신규 candidate는 제품 유형·문제·타깃·설득 축·톤·섹션 역할을 모두 명시한다.
- 제품군 profile을 추가해도 질문 라우팅 코어를 제품명에 맞춰 분기하지 않는다.
- 후보 집합과 평가 질의가 충분해지기 전에는 boost를 일반 성능으로 표현하지 않는다.
