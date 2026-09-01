# Robot Vacuum Semantic Normalization Holdout v1

이 자료는 개발 fixture와 다른 문장 원형에서 승인 엔티티 의미 정규화를 평가하는 합성 봉인
홀드아웃이다. 실제 사용자 대화나 사람 주석 자료는 아니다.

## 구성

- 정상 승인 사례 64건, 로봇청소기 변형 4종 각 16건
- fact 수 6~11개, claim fact 사례당 1개
- claim 능력군 8종 각 8건
- distractor 사례 16건, 복합 fact 18건
- 개발 normal/defensive와 fact·proposition exact overlap 0건
- 개발 세트와 ID overlap 0건
- 개발 문구와 최대 문자 4-gram Jaccard 0.0883

이 자료는 프롬프트 조정에 사용하지 않는다. 결과를 확인한 뒤 프롬프트나 라벨을 바꾸면
새 홀드아웃을 만들어야 한다.

```bash
uv run --no-sync python evals/validate_semantic_normalization_holdout.py
```

분포와 분리 검증 결과는 `dataset-manifest.json`에 기록한다.
