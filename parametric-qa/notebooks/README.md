# Parametric QA Notebooks

Jupyter 노트북으로 구성된 Parametric QA 실험 환경입니다.

## 노트북 목록

| # | 노트북 | 설명 |
|---|--------|------|
| 00 | [00_setup_and_config.ipynb](00_setup_and_config.ipynb) | 환경 설정, Config 로드, Import 테스트 |
| 01 | [01_data_preparation.ipynb](01_data_preparation.ipynb) | 데이터 다운로드, 전처리, DataLoader 구성 |
| 02 | [02_model_exploration.ipynb](02_model_exploration.ipynb) | 모델 구조 이해, z_i 개념, Selection 방법 |
| 03 | [03_write_phase_training.ipynb](03_write_phase_training.ipynb) | Stage 1: Document reconstruction 학습 |
| 04 | [04_read_phase_training.ipynb](04_read_phase_training.ipynb) | Stage 2: QA generation 학습 |
| 05 | [05_evaluation.ipynb](05_evaluation.ipynb) | QA, Write, Efficiency 종합 평가 |
| 06 | [06_baselines.ipynb](06_baselines.ipynb) | 7개 Baseline 비교 |
| 07 | [07_ablation_study.ipynb](07_ablation_study.ipynb) | 하이퍼파라미터 영향 분석 |
| 08 | [08_analysis_and_visualization.ipynb](08_analysis_and_visualization.ipynb) | 결과 분석 및 시각화 |
| 09 | [09_multihop_extension.ipynb](09_multihop_extension.ipynb) | Multi-hop QA (HotpotQA) |

## 실험 Phase별 권장 순서

### Phase 1: Proof of Concept (3일)
1. `00_setup_and_config.ipynb` - 환경 확인
2. `01_data_preparation.ipynb` - NQ 데이터 준비
3. `02_model_exploration.ipynb` - 모델 이해
4. `03_write_phase_training.ipynb` - Write phase 학습
5. `04_read_phase_training.ipynb` - Read phase 학습
6. `05_evaluation.ipynb` - 기본 평가

### Phase 2: Ablation Studies (4일)
1. `07_ablation_study.ipynb` - 하이퍼파라미터 sweep
   - z_dim: {64, 128, 256, 512, 1024}
   - m_tokens: {1, 4, 8, 16, 32}
   - selection_method: {cosine, learned_router, attention}
   - LoRA rank: {frozen, 8, 32, 64}

### Phase 3: Full Evaluation (5일)
1. `06_baselines.ipynb` - 7개 Baseline 비교
2. `05_evaluation.ipynb` - 3-seed 평가
3. `08_analysis_and_visualization.ipynb` - 결과 시각화

### Phase 4: Multi-hop Extension (3일)
1. `09_multihop_extension.ipynb` - HotpotQA 실험

## 요구사항

```bash
pip install -r ../requirements.txt
```

## GPU 메모리

- 최소: 16GB (QLoRA 4-bit)
- 권장: 24GB (L4 GPU)
- Qwen3-8B + QLoRA 기준

## 주의사항

1. **순차 실행**: 노트북은 순서대로 실행하는 것을 권장합니다.
2. **Checkpoint**: 학습 후 checkpoint를 저장하여 다음 노트북에서 사용합니다.
3. **메모리**: GPU 메모리가 부족하면 batch_size를 줄이세요.
4. **실행 시간**: 모델 로드에 시간이 걸릴 수 있습니다.
