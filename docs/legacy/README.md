# Legacy Documents

이 폴더의 문서들은 **더 이상 현재 연구 방향을 반영하지 않습니다**.

## 왜 레거시인가?

### 핵심 설계 오류

기존 문서들은 다음과 같은 구조를 제안했습니다:

```
Write Phase: z_i → D_i 전체 복원 (문서 재구성)
Read Phase: z_i (벡터) + Query → Answer (직접 생성)
            ↑
      외부 Select z_i 모듈
```

**문제점 3가지:**

1. **(A) Write Phase 목표가 잘못됨**
   - 기존: `z_i → D_i 전체 복원`
   - 올바른 방향: `(query, z) → evidence 생성`

2. **(B) 외부 Selector가 존재함**
   - 기존: 외부 모듈이 z_i를 선택
   - 올바른 방향: **LLM 내부 routing/attention**으로 z 선택 (외부 retriever 제거)

3. **(C) Local LLM이 Answer를 직접 생성함**
   - 기존: Local LLM → Answer
   - 올바른 방향: Local LLM → **Evidence (텍스트)** → ChatGPT → Answer

### 올바른 구조 (현재 방향)

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM 시스템                         │
│                                                             │
│   Memory Pool: Z = {z_i} (벡터/soft token)                 │
│                                                             │
│   Query ──→ [내부 routing/attention over Z] ──→ Evidence    │
│     │                                          (텍스트)     │
│     │                                              │        │
└─────┼──────────────────────────────────────────────┼────────┘
      │                                              │
      └──────────────┬───────────────────────────────┘
                     │
                     ▼
         Query + Evidence (텍스트) ──→ ChatGPT ──→ Answer
```

**핵심:**
- 내부 메모리(z): 벡터/soft token ✅
- 외부로 나가는 것: 텍스트(evidence) ✅
- 외부 retriever/selector: **없음** (내부 routing) ✅

## 파일 목록

| 파일 | 설명 | 레거시 이유 |
|------|------|-------------|
| `LLM as Memory 연구 종합 보고서(claude).md` | Claude 작성 보고서 | 2-Stage (Write/Read) 분리, 외부 selector |
| `LLM as Memory 연구 종합 보고서(chatgpt).md` | ChatGPT 작성 보고서 | evidence 생성은 맞으나 구조 불명확 |
| `Parametric QA 종합 실험 설계 계획서.md` | 실험 계획서 | z→D 복원 + 외부 selector + Answer 직접 생성 |
| `실험 설계 예시(claude).md` | Claude 실험 설계 | 위와 동일 |
| `실험 설계 예시(chatgpt).md` | ChatGPT 실험 설계 | evidence 방향은 맞으나 정리 안됨 |
| `RAG Experiment Design Methodology...md` | RAG 실험 방법론 | 일반적인 RAG 리뷰, 본 연구와 직접 관련 없음 |
| `RAG 관련 최신 실험 설계 동향.docx` | 동향 조사 | 참고 자료용 |

## 현재 문서

현재 연구 방향은 `docs/LLM-as-Memory 실험 설계서.md`를 참고하세요.
