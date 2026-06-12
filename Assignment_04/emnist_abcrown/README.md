# EMNIST Letters 검증: α,β-Crown

α,β-Crown (alpha-beta-CROWN)을 사용하여 EMNIST Letters 완전연결 신경망의 로버스트니스를 검증한다. Assignment 3 (Marabou 검증)의 후속 실험으로, 동일 모델과 동일 샘플을 사용하여 두 툴의 결과 및 속도를 직접 비교한다.

## 개요

- **모델**: 784 → 256 → 128 → 26 FC 네트워크 (EMNIST Letters, a–z)
- **검증 속성**: L∞ 로버스트니스, ε = 0.01 (Assignment 3와 동일)
- **검증 도구**: α,β-Crown — bound propagation + branch-and-bound (BaB)
- **비교 대상**: Assignment 3에서 Marabou(SMT 기반)로 얻은 결과

## 디렉토리 구조

```
Assignment_04/
├── alpha-beta-CROWN/       # α,β-Crown 클론 (저장소)
└── emnist_abcrown/
    ├── models/
    │   └── emnist_fc.onnx          # FC 모델 (Assignment 3에서 복사)
    ├── specs/                      # 자동 생성 VNNlib 스펙 (260개)
    ├── results/
    │   ├── verification_results.csv
    │   ├── comparison_results.csv
    │   ├── fig1_result_bar.png
    │   ├── fig2_time_comparison_scatter.png
    │   ├── fig3_time_distribution.png
    │   └── fig4_agreement_heatmap.png
    ├── instances.csv               # VNN-COMP 포맷 인스턴스 목록
    ├── abcrown_config.yaml         # α,β-Crown 설정 파일
    ├── test.py                     # 메인 검증 파이프라인
    ├── generate_specs.py           # VNNlib 스펙 생성 모듈
    ├── parse_results.py            # abcrown 출력 파서
    ├── visualize.py                # 시각화 모듈
    ├── environment.yml             # conda 환경 스펙
    └── requirements.txt            # pip 의존성
```

## 환경 설정

### 1. α,β-Crown 클론

```bash
cd Assignment_04
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
git submodule update --init --recursive
```

### 2. conda 환경 생성

```bash
conda create -n abcrown python=3.11 -y
conda activate abcrown

# PyTorch 설치 (CUDA 드라이버에 맞게 선택)
# CUDA 12.1 드라이버 (GPU 권장):
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 전용 또는 최신 CUDA:
pip install torch==2.11.0 torchvision

# α,β-Crown 의존성 설치
pip install -e alpha-beta-CROWN/auto_LiRPA
pip install -r emnist_abcrown/requirements.txt
```

> **CUDA 호환성 참고**: 시스템 드라이버가 535.x(CUDA 12.2)인 경우 torch 2.11.0+cu130 빌드와 호환되지 않음. 이 경우 CPU 전용으로 실행되며, 실험 결과에는 영향 없음.

### 3. 설치 확인

```bash
conda activate abcrown
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

## 실행 방법

### 전체 파이프라인 실행

```bash
conda activate abcrown
cd emnist_abcrown
python test.py
```

옵션:
```
--samples-per-class N   클래스당 샘플 수 (기본값: 10)
--epsilon E             L∞ perturbation 반지름 (기본값: 0.01)
--timeout T             인스턴스당 타임아웃(초) (기본값: 30)
```

### 스모크 테스트 (클래스당 1개, 총 26개)

```bash
python test.py --samples-per-class 1
```

## 모델 구조

```
입력 (28×28 이미지)
  └→ Flatten → (784,)
  └→ Linear(784, 256) + ReLU
  └→ Linear(256, 128) + ReLU
  └→ Linear(128, 26)   [a–z 로짓]
```

- 학습: EMNIST Letters, 10 에폭, Adam (lr=1e-3)
- 테스트 정확도: ~90.64%
- 포맷: ONNX opset 11

## 검증 방법

### VNNlib 스펙 형식

각 샘플에 대해 `.vnnlib` 파일이 생성됨:
- **입력 제약**: 정규화된 픽셀 값 x_i를 기준으로 `[x_i − ε, x_i + ε]` 구간
- **출력 속성 (부정 형식)**: `(or (and (>= Y_j Y_true)) ...)` — 임의의 오답 클래스 j가 정답 클래스를 이기면 위반
  - `unsat` → 검증됨: 반례 없음, 모델이 로버스트
  - `sat` → 위반됨: 반례 발견, 모델이 로버스트하지 않음
  - `timeout` → 시간 초과, 미결정

### abcrown_config.yaml 주요 설정

```yaml
general:
  device: cpu               # GPU 가능 시 'cuda'로 변경

bab:
  timeout: 30               # 인스턴스당 타임아웃(초)
  branching:
    method: kfsb            # FC 네트워크에 최적

attack:
  pgd_order: before         # UNSAT 탐색 전 PGD 선공격으로 SAT 조기 탐지
```

### Marabou (Assignment 3)와 비교

| 항목 | Marabou (Assignment 3) | α,β-Crown (Assignment 4) |
|------|----------------------|-------------------------|
| 핵심 방법 | SMT / LP 솔버 (symbolic) | Bound propagation + BaB (numeric) |
| SAT 탐지 | SMT 열거 | PGD 공격 (gradient 기반) |
| UNSAT 증명 | SMT 반박 | 선형 완화(CROWN) 경계 계산 |
| 쿼리 수 | 클래스 쌍당 25개 서브쿼리 | 1개 DNF 쿼리 (전체 클래스 동시) |
| GPU 지원 | 없음 | 지원 (CUDA) |
| 인터페이스 | Python API | YAML + VNNlib |

## 결과 요약

`test.py` 실행 후 `results/` 디렉토리에서 확인:

| 파일 | 내용 |
|------|------|
| `verification_results.csv` | 260개 샘플 검증 결과 (result, time_s) |
| `comparison_results.csv` | α,β-Crown과 Marabou 결과 병합 |
| `fig1_result_bar.png` | 글자별 verified/falsified/timeout 막대그래프 |
| `fig2_time_comparison_scatter.png` | 속도 비교: 로그-로그 산점도 + 속도 향상 히스토그램 |
| `fig3_time_distribution.png` | 검증 시간 분포 히스토그램 |
| `fig4_agreement_heatmap.png` | 두 툴 결과 일치 히트맵 |

**주요 결과**: 260개 샘플 중 233개 verified (89.6%), 27개 falsified (10.4%), Marabou와 100% 일치. α,β-Crown의 중앙값 속도 향상: **356×** (CPU 기준).
