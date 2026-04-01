# Assignment 01 — Adversarial Attacks on Neural Networks

Implementation of adversarial attack methods (FGSM, PGD) on MNIST and CIFAR-10 datasets.

---

## Project Structure

```
Assignment_01/
├── Attack/
│   ├── fgsm.py          # FGSM targeted / untargeted
│   └── pgd.py           # PGD targeted / untargeted
├── Model/
│   ├── mnist_cnn.py     # Custom CNN for MNIST
│   ├── cifar_dla.py     # Custom Simple DLA for MNIST (NOT USED)
│   ├── resnet18.py      # Pretrained ResNet-18 for CIFAR-10
│   ├── utils.py         # Training / evaluation utilities
│   └── training_result/ # Saved model weights (.pt) and training accuracy curve
├── data/                # Downloaded datasets (auto-generated)
├── results/             # Output images and CSV (auto-generated)
├── test.py              # Main experiment script
└── requirements.txt
```

---

## Requirements

```bash
pip install -r requirements.txt
```

> GPU 사용 시 CUDA 버전에 맞는 PyTorch를 설치해야 합니다.
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## How to Run
Assignment_01 폴더로 이동 후, test.py 파일 실행

```bash
python test.py
```
파일 실행 시, 아래 순서로 실험 코드가 수행됨
1. MNIST / CIFAR-10 데이터셋 자동 다운로드 (`./data/`)
2. 저장된 모델 weight 로드 (`./Model/training_result/`)
   - 저장된 모델이 없을 경우 자동으로 학습 후 저장
3. 실험 A: Epsilon별 ASR 평가 (Targeted, target class=8 고정)
4. 실험 B: Epsilon별 ASR 평가 (Untargeted)
5. 실험 C: Target class별 ASR 평가 (eps=0.3 고정, target 0~9 순회)
6. 시각화 결과 저장 (`./results/*.png`)
7. 실험 결과 CSV 저장 (`./results/results.csv`)

---

## Experiments

| 실험 | 설명 | 고정 조건 |
|------|------|-----------|
| A | Epsilon별 ASR (Targeted) | target class = 8 |
| B | Epsilon별 ASR (Untargeted) | - |
| C | Target class별 ASR | eps = 0.3 |

Epsilon 범위: `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`

---

## Attack Methods

| 방법 | 설명 |
|------|------|
| FGSM Targeted | 단일 gradient 스텝으로 특정 클래스로 유도 |
| FGSM Untargeted | 단일 gradient 스텝으로 오분류 유도 |
| PGD Targeted | iterative FGSM + ε-ball projection (k=40, step=0.01) |
| PGD Untargeted | iterative FGSM + ε-ball projection (k=40, step=0.01) |

---

## Output

```
results/
├── results.csv                      # 전체 실험 결과
├── mnist_fgsm_targeted.png          # 시각화 (MNIST, FGSM Targeted)
├── mnist_fgsm_untargeted.png
├── mnist_pgd_targeted.png
├── mnist_pgd_untargeted.png
├── cifar_fgsm_targeted.png          # 시각화 (CIFAR-10, FGSM Targeted)
├── cifar_fgsm_untargeted.png
├── cifar_pgd_targeted.png
└── cifar_pgd_untargeted.png
```

---

## Models

- **MNIST**: 2개의 합성곱 레이어(32, 64 채널)와 완전 연결 레이어(128→10)로 구성된 Custom CNN
- **CIFAR-10**: ImageNet pretrained ResNet-18을 CIFAR-10에 맞게 fine-tuning
