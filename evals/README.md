# 평가 자료 안내

이 디렉터리는 Funding Story AI의 대화 worker, 의미 정규화, 미디어 계획을 작은 단위부터
전체 흐름 순서로 검증하기 위한 데이터셋과 실행 도구를 보관한다. 실제 사용자 데이터가 아닌
합성 자료와 공개 페이지에서 관찰한 파생 메타데이터를 사용하며, 모델 실행 결과는
`evals/results/`에 저장하고 Git에는 포함하지 않는다.

## 자료 구성

### Story Maker Worker

`evals/datasets/story-worker-*-evaluation-v1.json`은 다음 책임을 각각 평가한다.

- `field`: 사용자 입력을 16개 제품 정보 필드로 반영하는지
- `collection`: 필수·선택 정보의 그룹별 수집과 생략 처리가 올바른지
- `question`: 이미 답한 내용을 불필요하게 반복하지 않고 적절한 후속 질문을 만드는지
- `summary`: 생성 직전 요약이 사용자 입력에 근거하는지
- `approval`: 명시적 승인 전 생성을 차단하고 수정 후 재승인을 요구하는지
- `flow`: 질문 → 보완 → 요약 → 승인 전체 흐름이 안정적인지

### MediaFacts 의미 정규화

| 자료 | 용도 | 사례 수 |
| --- | --- | ---: |
| [`robot-vacuum-semantic-normalization-v1`](datasets/robot-vacuum-semantic-normalization-v1/README.md) | 개발용 정상·방어 자료 | 64 + 64 |
| [`robot-vacuum-semantic-normalization-holdout-v1`](datasets/robot-vacuum-semantic-normalization-holdout-v1/README.md) | 문장 원형이 다른 봉인 홀드아웃 | 64 |
| [`robot-vacuum-semantic-normalization-adversarial-v1`](datasets/robot-vacuum-semantic-normalization-adversarial-v1/README.md) | 지시문 형태 입력·위조 참조 방어 | 64 |
| [`semantic-normalization-v1`](adjudication/semantic-normalization-v1/README.md) | AI 이중 주석과 사용자 승인 기록 | 16 |

각 사례는 LLM이 보는 `model_view`, 결정적 adapter가 사용하는 승인 projection, 기대 결과를
분리한다. 홀드아웃은 결과 확인 후 프롬프트 조정에 재사용하지 않는다.

### MediaPlan

| 자료 | 용도 | 사례 수 |
| --- | --- | ---: |
| [`robot-vacuum-media-planning-v1`](datasets/robot-vacuum-media-planning-v1/README.md) | 능력군 활성화·슬롯·참조·차단 규칙 | 64 + 64 |
| [`robot-vacuum-media-slots-v1`](datasets/robot-vacuum-media-slots-v1/README.md) | 성공 페이지의 영역–미디어 슬롯 관찰 자료 | README 참조 |

미디어 슬롯 자료는 원본 이미지나 영상을 포함하지 않고 공개 페이지 식별자와 관찰 가능한
파생 메타데이터만 저장한다. 성공 성과와 특정 구성의 인과관계를 주장하는 자료가 아니다.

## 검증 순서

먼저 정적 계약과 데이터셋 무결성을 검사한다.

```bash
uv run python evals/validate_semantic_normalization_dataset.py
uv run python evals/validate_semantic_normalization_holdout.py
uv run python evals/validate_semantic_normalization_adversarial.py
uv run python evals/validate_media_planning_dataset.py
```

그다음 제품 코드와 평가 도구의 회귀 테스트를 실행한다.

```bash
uv run pytest
```

실제 Gemini 의미 정규화 평가는 별도 비용 승인을 받은 실행에서만 수행한다.

```bash
uv run python evals/run_semantic_normalization.py \
  --split normal \
  --output evals/results/semantic-normalization-v1/normal-predictions.json \
  --errors-output evals/results/semantic-normalization-v1/normal-errors.json

uv run python evals/score_semantic_normalization.py \
  evals/results/semantic-normalization-v1/normal-predictions.json \
  --split normal \
  --fail-on-gate
```

각 하위 README와 `dataset-manifest.json`이 해당 자료의 세부 분포, 한계와 판정 기준의
권위 있는 설명이다.
