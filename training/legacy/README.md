# Legacy Training Scripts

이 폴더의 파일들은 **레거시 2-Stage 설계** 기반입니다.

## 파일 목록

| 파일 | 설명 | 레거시 이유 |
|------|------|-------------|
| `train_write.py` | Write Phase 학습 | Write Phase 개념 변경 |
| `train_read.py` | Read Phase 학습 | Answer 생성 → Evidence 생성으로 변경 |
| `train_e2e.py` | End-to-end 학습 | 2-Stage → 1-Stage로 변경 |

## 현재 사용할 파일

- `training/train_evidence.py` - Evidence 생성 학습 스크립트
