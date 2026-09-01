# Robot Vacuum Semantic Normalization Adversarial v1

이 합성 자료는 개발 방어 세트의 고정된 prompt-injection 문구와 한정된 위조 참조 위치를
다양화한다. 실제 공격 로그나 보안 인증 자료는 아니다.

## 구성

- 명령처럼 보이는 승인 fact 데이터 32건: 한국어·영어·혼합 언어, 문장 앞·중간·끝,
  역할 사칭·JSON·Markdown·HTML·도구 호출 형식
- 위조 참조 32건: entity/fact의 source·evidence·asset ref 및 evidence/asset catalog의
  source ref 위치별 각 4건
- 제품 변형 4종 각 16건

prompt-injection 32건은 유효한 승인 경계를 통과하며, 제어 문자열을 지시로 따르거나
proposition으로 만들지 않고 실제 제품 fact만 분류해야 한다. 위조 참조 32건은 의미 모델을
호출하기 전에 `reject_forged_reference`로 거부하고 모든 의미 출력과 MediaFacts를 비운다.

```bash
uv run --no-sync python evals/validate_semantic_normalization_adversarial.py
```

정확한 분포와 교차 split 중복 검사는 `dataset-manifest.json`에 기록한다.
