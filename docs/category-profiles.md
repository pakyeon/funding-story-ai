# 제품군 프로필

제품군 프로필은 공통 입력 그래프를 바꾸지 않고 질문 문구와 카테고리 신호를
전문화한다.

## 공통 의미 슬롯

- `product_identity`
- `key_strengths`
- `target_supporters`
- `problem_context`
- `trust_elements`
- `maker_team_intro`

각 슬롯은 `provided`, `explicitly-absent`, `unknown` 상태와 값, 출처 턴을 가진다.
제품군별 기능명을 코어 상태 이름으로 사용하지 않는다.

## 프로필이 제공하는 값

```json
{
  "profile_id": "robot-vacuum-ko-v1",
  "match_terms": ["로봇청소기", "흡입력", "물걸레", "LiDAR"],
  "semantic_slot_guidance": {
    "key_strengths": {
      "extraction_hints": ["흡입력", "센서", "도크 자동화"],
      "question_examples": ["흡입·물걸레·주행·도크에서 핵심 특장점은 무엇인가요?"]
    }
  },
  "template_soft_boosts": {
    "t02_problem_solution_automation": 3
  }
}
```

질문 그래프는 어떤 예시가 로봇청소기용인지 알지 못한다. `question_prompt()`가 선택된
stage의 슬롯과 profile을 조합한다. `template_soft_boosts`는 API 호출 없는 직접
선택기에서만 사용한다. 전체 MCP 경로의 KNN은 profile의 `category`와 별도 환경 변수
`TEMPLATE_CATEGORY_BOOST`를 사용한다.

## 새 제품군 추가 절차

1. 공통 슬롯으로 표현되지 않는 필수 의미가 있는지 확인한다.
2. 제품명과 유형 match term을 정의한다.
3. 슬롯별 추출 힌트와 중립적인 질문 예시를 작성한다.
4. 실제 템플릿 근거가 있을 때만 soft boost를 추가한다.
5. 합성 브리프와 질문·선택 회귀 테스트를 작성한다.
6. 별도 홀드아웃으로 실제 동작을 확인하기 전에는 검증 완료로 표시하지 않는다.

현재 포함된 실제 검증 프로필은 `robot-vacuum-ko-v1` 하나다.
