# Assignment 2: DeepXplore

## 파일 구조
```bash
Assignment_02/   
├── CIFAR10/   
│   ├── models/   
│   │   ├── model.py          # ResNet50 model definition   
│   │   ├── train.py          # Model training script   
│   │   ├── gen_diff.py       # DeepXplore main script (modified for CIFAR-10)   
│   │   ├── utils.py          # Utility functions (modified for TF2)   
│   │   ├── test.py           # Demo script   
│   │   ├── visualize.py      # Visualization script   
│   │   └── requirements.txt
│   └── results/              # Visualization outputs
└── deepxplore/               # Original DeepXplore repository
```

## 환경 설정 및 모델 학습
```bash
# 환경 설정
pip install -r requirements.txt

# 모델 학습
cd Assignment_02/CIFAR10/models
python train.py

# 커스텀 설정으로 학습
python train.py --img_size 64 --epochs 10 --batch_size 32 --weights imagenet
```

## DeepXplore 실행
```bash
cd Assignment_02/CIFAR10/models

# 단일 transformation 실행
python gen_diff.py light 1.0 0.1 0.05 100 50 0.5

#데모 실행
python test.py
```
### Arguments
| Argument | Description | Example |
|---|---|---|
| transformation | Image transformation type | light, occl, blackout |
| weight_diff | Weight for differential behavior | 1.0 |
| weight_nc | Weight for neuron coverage | 0.1 |
| step | Gradient descent step size | 0.05 |
| seeds | Number of seed inputs | 100 |
| grad_iterations | Gradient descent iterations | 50 |
| threshold | Neuron activation threshold | 0.5 |

## Model Configuration
- **Model A**: ResNet50, random seed=111, SGD optimizer (lr=0.01, momentum=0.9), basic augmentation
- **Model B**: ResNet50, random seed=222, AdamW optimizer (lr=0.001, weight_decay=0.01), strong augmentation

## DeepXplore 수정 사항
Python = 3.11 와 Tensorflow = 2.21.0에 맞게 조정되었으며, CIFAR-10과 ResNet50에 맞춰 일부 코드 수정함.

1. 데이터셋: MNIST → CIFAR-10 (64×64로 리사이즈)
2. 모델: 소형 모델 3개 → ResNet50 2개
3. `from keras` → `from tensorflow.keras`
4. `xrange` → `range` (Python 3 호환)
5. `scipy.misc.imsave` → `imageio.imwrite`
6. `K.function` + `K.gradients` → `tf.GradientTape` (TF2 호환)
7. `layer.output_shape` → `layer.output.shape` (TF2 호환)
8. 4D conv 레이어만 뉴런 커버리지 추적 대상으로 설정

## 모델 성능
| 모델 | 옵티마이저 | Test Accuracy |
|---|---|---|
| Model A | SGD (lr=0.01, momentum=0.9) | 74.66% |
| Model B | AdamW (lr=0.001, weight_decay=0.01) | 84.75% |

## 실험 결과
| Transformation | Disagreements | Model A Coverage | Model B Coverage |
|---|---|---|---|
| light | 32 | 70.5% | 61.9% |
| occl | 57 | 71.2% | 63.1% |
| blackout | 51 | 71.3% | 63.2% |