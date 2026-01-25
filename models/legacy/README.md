# Legacy Models

이 폴더의 파일들은 **레거시 2-Stage 설계** 기반으로 작성되었습니다.

## 왜 레거시인가?

### 기존 설계 (레거시)
```
Write Phase: z_i → D_i 전체 복원
Read Phase: 외부 Selector → z_i 선택 → LLM → Answer 직접 생성
```

### 새 설계 (현재)
```
Query → [LLM 내부 Routing over Z] → Evidence 텍스트 생성
외부 Selector 없음
```

## 파일 목록

| 파일 | 설명 | 레거시 이유 |
|------|------|-------------|
| `parametric_qa.py` | 기존 메인 모델 | 외부 selector 사용, Answer 직접 생성 |
| `router.py` | 문서 선택 모듈 | 외부 selector (새 설계에서 제거) |
| `write_phase.py` | Write Phase 학습기 | Write Phase 개념 변경 (문서 복원 → X) |
| `read_phase.py` | Read Phase 학습기 | Answer 생성 → Evidence 생성으로 변경 |

## 유용한 아이디어 (새 코드에 반영됨)

- `parametric_qa.py`의 LLM 초기화 (QLoRA) → `parametric_memory_llm.py`
- `parametric_qa.py`의 z_to_embedding → `parametric_memory_llm.py`
- `read_phase.py`의 optimizer 구조 → `evidence_trainer.py`

## 현재 사용할 파일

- `models/parametric_memory_llm.py` - 핵심 모델
- `models/evidence_trainer.py` - Evidence 생성 학습기
