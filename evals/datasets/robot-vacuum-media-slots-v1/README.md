# Robot Vacuum Media Slot Annotations v1

## 목적

성공한 와디즈 로봇청소기 스토리에서 사람이 확인한 `page → section → media_slot` 관계를
기록한다. 이 자료는 다음 작업에 사용한다.

- 제품군 미디어 프로필의 능력군 검토
- 영역별 후보 슬롯과 참조 이미지 정책 보정
- 이미지 계획 노드의 영역·능력군·배치 분류 평가
- 특정 제품의 기능 목록을 고정 규칙으로 복제하는 과적합 방지

원본 이미지·영상은 포함하지 않는다. 공개 페이지 ID와 관찰 가능한 파생 메타데이터만
저장한다.

## 데이터 계층

- `strict_mixed`: 핵심 기능 설명에서 HTML 텍스트와 미디어가 교차하는 페이지의 슬롯 주석
- `visual_heavy_contrast`: 기능 설명이 이미지·GIF·반복 MP4 묶음에 집중된 페이지 메타데이터

`visual_heavy_contrast`는 이미지 내부 OCR·비전 주석 전까지 영역–슬롯 정답으로 사용하지
않는다.

## 주석 단위

하나의 슬롯은 파일 하나나 기능 하나가 아니라, 한 영역에서 하나의 설득 목적을 전달하는
미디어 묶음이다. 동일한 사용 장면과 목적을 공유하는 정적 이미지·GIF는 하나의 슬롯으로
묶을 수 있다.

예시:

```text
카펫 감지 + 흡입력 변화 + 브러시 작동
→ capability_group: cleaning_mechanism
→ slot_grouping_key: carpet_cleaning_adaptation
```

## 주요 라벨

### `capability_group`

- `product_identity_outcome`
- `problem_environment`
- `cleaning_mechanism`
- `mobility_coverage`
- `automation_return`
- `control_personalization`
- `evidence_performance`
- `configuration_maintenance`

### `evidence_status`

- `observed`: HTML 텍스트, DOM 미디어 또는 명시적 주변 문맥으로 확인
- `inferred`: 위치와 주변 문맥으로 추론했지만 미디어 내부 내용은 확인하지 못함
- `unverified`: 이미지 내부 내용·원본·크기를 확인하지 못함

### `reference_policy`

- `required`: 실제 제품·도크·부속품의 외형 유지가 필요한 장면
- `optional`: 제품이 등장할 수도 있지만 문제·환경 표현만으로 성립하는 장면
- `none`: 차트·표·텍스트 카드처럼 **제품 외형** 참조가 필요하지 않은 장면

이 값은 제품 외형 일관성 정책이다. `none`이어도 인증서·시험 결과·실제 제어 화면을
사용하려면 `reference_asset_roles`에 `provided_evidence` 또는 `control_interface`를 기록할
수 있다. 이는 생성 이미지의 외형 참조가 아니라 사실 근거·실제 화면 자산의 필요성을
뜻한다.

### `priority_candidate`

표본에서 관찰된 역할을 기반으로 한 후보 라벨이다. 아직 실행 프로필의 최종
`required/recommended/optional` 값이 아니다.

## 품질 규칙

- 관찰과 추론을 분리한다.
- 확인되지 않은 미디어 수·원본 규격은 `null` 또는 `unverified`로 남긴다.
- 브랜드명·제품명·고유 수치와 센서명을 일반 규칙으로 만들지 않는다.
- 같은 페이지의 반복 미디어를 서로 다른 평가 분할로 나누지 않는다.
- 이미지 내부 문구를 OCR하지 않은 경우 그 내용을 정답으로 작성하지 않는다.
- 성공 성과와 미디어 구성의 인과관계를 주장하지 않는다.

## 데이터 분할 주의

현재 v1 표본은 모두 능력군과 스키마 설계에 사용했으므로 최종 테스트 홀드아웃이 아니다.
추후 평가용 페이지는 새로운 프로젝트를 확보하고 **프로젝트 단위**로 분리한다. 동일 제품,
같은 메이커의 앵콜 프로젝트도 서로 다른 독립 홀드아웃으로 취급하지 않는다.

## 파일과 현재 규모

- `part-a.json`: strict 표본 3개, 슬롯 27개
- `part-b.json`: strict 표본 4개, 슬롯 32개
- `part-c.json`: strict 표본 2개·슬롯 16개와 대비 표본 6개
- `dataset-manifest.json`: 전체 표본·슬롯 수와 검증 결과
- `slot-annotation.schema.json`: strict 슬롯 계약
- `contrast-page.schema.json`: 대비 페이지 계약

현재 strict 데이터는 9개 페이지·75개 슬롯으로, 이 데이터 범주의 목표 분모인 50~200을
충족한다. 다만 슬롯별 실제 미디어 파일 수는 75개 중 12개만 확인됐고 원본 규격을 완전히
검증한 슬롯은 없다. 따라서 이 데이터로 능력군·배치·참조 정책은 평가할 수 있지만 실제
생성 이미지 개수나 최소 원본 해상도를 확정하지 않는다.

이 데이터셋은 관찰 결과의 분류 정답이다. 생성기의 실행 평가에는 구조화된 제품 사실,
영역 골격, 예상 활성·비활성 슬롯을 함께 가진 별도 `media-planning` 사례가 필요하다.

## 완료 기준

- strict 표본 9개의 각 핵심 설득 단계가 최소 한 번 주석됨
- 전체 슬롯 주석 50~200개
- annotation ID 중복 0건
- 열거형·필수 필드 누락 0건
- 모든 strict 표본이 둘 이상의 능력군을 포함
- 대비 표본이 슬롯 정답 집계에서 제외됨
