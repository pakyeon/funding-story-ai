# 의미 정규화 독립 주석·잠정 조정 v1

## 범위

개발용 정상 사례 앞 16건, 147개 fact를 두 개의 독립 `gpt-5.6-luna(max)` 실행으로
주석했다. 두 주석자는 `model_view`만 보았으며 기존 정답, 원본 projection, source/evidence/
asset catalog와 서로의 출력을 보지 않았다.

이는 **AI 이중 주석**이며 사람 이중 주석으로 간주하지 않는다. 세 가지 잠정 판정은
2026-09-01 사용자 검수로 승인됐고 `human-signoff.json`에 승인 범위를 보존한다.

## 결과

| 지표 | 결과 |
|---|---:|
| 판단(`classify`/`ignore`) 일치 | 147/147 (100%) |
| 능력군 label set 일치 | 137/147 (93.20%) |
| 절 분해·문구·능력군 전체 일치 | 116/147 (78.91%) |
| 주석 A와 잠정 gold 전체 일치 | 147/147 (100%) |
| 주석 B와 잠정 gold 전체 일치 | 116/147 (78.91%) |

두 주석자가 동일하게 기존 gold와 달랐던 1건은 `evidence_performance`에서
`configuration_maintenance`로 잠정 수정했다. 문장의 출처가 후기라는 사실보다 “브러시
관리 편의”라는 주장 내용이 분류 대상이라는 판단이다. 연결된 source/evidence 설명도 실제
문장과 일치하도록 후기 자료로 바로잡았다.

나머지 불일치는 주로 다음 두 정책 차이다.

1. 같은 능력군·상태·근거를 공유하는 한 문장 안의 병렬 동작을 한 proposition으로 둘지,
   동작마다 나눌지
2. 방 선택·금지 영역·지도 기반 재청소 지정을 사용자 설정으로 볼지, 이동·공간 대응으로
   다시 나눌지

잠정 gold는 같은 능력군·상태·근거의 병렬 동작을 하나로 유지하고, `그리고`로 연결된 독립
서술이나 서로 다른 능력군·상태·근거만 나눈다. 방 선택·금지 영역·지도 기반 지정은
`control_personalization`으로 유지한다.

## 사용자 검수 결과

- [x] gold 1건의 `evidence_performance → configuration_maintenance` 수정
- [x] 동일 능력군 병렬 동작을 합쳐 두는 절 경계 정책
- [x] 방 선택·금지 영역·지도 기반 지정의 `control_personalization` 분류

승인 상태는 `AI 독립 이중 주석 + 사용자 검수 승인 완료`다. 사람 주석자가 147개 fact를
새로 이중 주석했다는 의미는 아니다.

전체 31개 불일치와 두 주석, 잠정 gold는 `adjudication-report.json`에 보존한다.
`build_semantic_adjudication_report.py`로 원본 주석에서 보고서를 재생성할 수 있다.
