# 적응형 story-maker-worker 평가 결과

- 평가 데이터셋: `adaptive-worker-evaluation-v1`
- 평가 범위: 구조화 계약, LangGraph 노드, 세션 상태, 승인 guard와 FastMCP 호출 경계
- 데이터: 한국어 로봇청소기 대화 4개 시나리오, 총 12턴
- 실행 방식: 고정된 LLM 구조화 출력을 사용하는 오프라인 회귀 평가
- 저장소 회귀 검증: 핵심 브랜치 65개, UI 브랜치 71개 테스트 통과

## 1. 평가 해석 범위

이번 평가는 LLM 공급자의 자연어 품질 점수가 아니라, 주어진 구조화 LLM 출력이 각 노드의
계약과 상태 전이를 통해 의도대로 처리되는지를 검증한다. 따라서 입력 이해·질문·요약·승인
수치는 **노드 계약과 오케스트레이션 평가 결과**이며 Gemini 자체의 실서비스 정확도로
해석하면 안 된다.

실제 Gemini 품질 평가는 모델 접근 환경에서 동일 데이터셋을 실행한 뒤 별도로 기록해야
한다. 유료 API를 호출하지 않는 기본 테스트에서는 수행하지 않는다.

## 2. 데이터셋 구성

| 시나리오 | 검증 흐름 | 턴 수 |
|---|---|---:|
| `partial-then-approve` | 즉시 생성 요청 → 필수 정보 질문 → 보완 → 요약 → 승인 | 3 |
| `ambiguous-approval` | 충분한 입력 → 요약 → 모호한 응답 차단 → 명시적 승인 | 3 |
| `revision-invalidates-approval` | 요약 → 타깃 수정 → 새 요약 버전 → 재승인 | 3 |
| `reject-without-generation` | 요약 → 생성 거절 → 수집 상태 복귀 → 세션 종료 | 3 |

원본 데이터는
[`tests/fixtures/adaptive-worker-evaluation.json`](../tests/fixtures/adaptive-worker-evaluation.json),
평가 러너는
[`tests/test_adaptive_worker_evaluation.py`](../tests/test_adaptive_worker_evaluation.py)에 있다.

## 3. 노드별 평가 결과

| 평가 대상 | 결과 | 판정 |
|---|---:|---|
| 사실 변경안 반영 | 21/21, 100% | 통과 |
| 필수 정보 누락 검사 | 6/6, 100% | 통과 |
| 질문의 누락 필드 관련성 | 1/1, 100% | 통과 |
| 이미 제공된 필드 반복 질문 | 0건 | 통과 |
| 최신 수정값 반영 | 1/1, 100% | 통과 |
| 요약의 구조화 사실 일치 | 5/5, 100% | 통과 |
| 변조된 요약 사실 차단 | 1/1, 100% | 통과 |
| 승인 의도별 라우팅 | 6/6, 100% | 통과 |
| 허용되지 않은 상태 전이 | 0건 | 통과 |
| 승인 없는 FastMCP 호출 | 0건 | 통과 |
| 승인된 FastMCP 호출 | 3건 | 통과 |

`변조된 요약 사실 차단`은 `confirmed_facts`가 현재 구조화 사실과 달라진 한 개의 공격적
회귀 사례를 의미한다. 이 결과를 실제 자연어 출력의 일반적인 환각 현상 발생률 0%로
해석하지 않는다. 현재 수치가 보장하는 것은 구조화 사실 불일치가 승인 화면으로 넘어가지
않았다는 점이다.

## 4. 전체 흐름 평가 결과

| 지표 | 결과 | 판정 |
|---|---:|---|
| 기대 단계 전이 | 12/12, 100% | 통과 |
| 기대 최종 상태 도달 | 4/4, 100% | 통과 |
| 필수 정보 충족 뒤 요약 도달 | 4/4, 100% | 통과 |
| 수정된 최신 사실의 최종 반영 | 1/1, 100% | 통과 |
| 질문 단계에서 제공 완료 필드 반복 | 0건 | 통과 |
| 새 worker 인스턴스의 SQLite 세션 복구 | 1/1, 100% | 통과 |
| 서로 다른 `thread_id`의 상태 혼합 | 0건 | 통과 |
| 승인 전 도구 호출 | 0건 | 통과 |

고정 시나리오의 평균 대화 길이는 3턴이다. 이는 제품 경험의 목표 턴 수가 아니라 현재
평가 fixture의 구성값이다.

## 5. 원래 요구사항 대조

| 원래 요구사항 | 구현 | 검증 결과 |
|---|---|---|
| LangGraph 기반 질문 → 보완 → 요약 → 승인 | `conversation.py` 조건부 그래프 | 단계 전이 12/12 통과 |
| 필수 필드 작성 여부만 규칙으로 판단 | `check_required_fields` | 누락 검사 6/6 통과 |
| 변경 가능한 의미 판단은 LLM이 담당 | `understand_turn`, `plan_next_questions`, `build_summary`, `classify_approval` | 각 노드의 구조화 계약과 라우팅 통과 |
| 이전 입력과 답변을 세션 문맥으로 유지 | `StoryWorkerState`와 SQLite Checkpointer | 인스턴스 간 복구 1/1 통과 |
| 최신 수정값을 기준으로 사용 | `apply_fact_patch`, `fact_history`, `facts_revision` | 최신값 반영 1/1 통과 |
| 생성 직전 결정 내용 요약 | `build_summary`, `request_approval` | 근거 일치 5/5 통과 |
| 사용자 명시적 승인 필수 | `classify_approval`, `approval_guard` | 비승인 호출 0건 |
| 자동 생성과 질문 건너뛰기 생성 없음 | 불리언 승인·skip 경로 제거 | 즉시 생성 요청이 질문으로 전환됨 |
| 요약 수정 시 기존 승인 무효화 | `summary_facts_revision`, `summary_version` | 수정 후 버전 1→2 및 재승인 요구 |
| 장기 메모리는 초기 MVP에서 제외 | thread 범위 Checkpointer만 사용 | 사용자·프로젝트 간 장기 기억 없음 |
| 노드별·전체 흐름 평가 | 고정 fixture와 자동 회귀 평가 | 본 문서의 모든 오프라인 지표 통과 |

## 6. 남은 검증

- 실제 Gemini 출력의 사실 추출 정확도와 수정 감지율
- 다양한 표현에서 질문 관련성, 질문 묶음과 자연스러움
- 자연어 요약의 누락률과 환각 현상 발생률
- 실제 승인·조건부 승인·반문 표현의 분류 정확도
- 로봇청소기 이외 제품군의 필수 정보와 질문 흐름
- 장시간 대화에서 토큰 사용량과 요약 도입 임계값
- 여러 프로세스가 같은 SQLite 파일을 사용하는 경우의 운영 안전성

이 항목은 현재 구조의 통과 결과와 분리해 후속 live-model 평가로 관리한다.

## 7. 재현 명령

```bash
uv run pytest tests/test_adaptive_worker_evaluation.py -q -s
uv run pytest tests/test_conversation.py tests/test_worker.py -q
```
