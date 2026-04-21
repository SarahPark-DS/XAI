import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import imageio

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

model1 = load_model('../models/model_a_initialized.h5')
model2 = load_model('../models/model_b_initialized.h5')

os.makedirs('./results', exist_ok=True)

def get_prediction(model, img_array):
    img = np.expand_dims(img_array.astype('float32') / 255.0, axis=0)
    pred = model.predict(img, verbose=0)
    return CIFAR10_CLASSES[np.argmax(pred[0])], np.max(pred[0])

def visualize_disagreements(input_dir, transformation, n=3):
    # gradient-induced만 선택 (_orig 있는 것만)
    files = [f for f in os.listdir(input_dir)
             if f.endswith('.png') and '_orig' not in f
             and not f.startswith('already_disagree')]
    
    # _orig 파일이 있는 것만 필터링
    valid_files = []
    for f in files:
        orig_path = os.path.join(input_dir, f.replace('.png', '_orig.png'))
        if os.path.exists(orig_path):
            valid_files.append(f)
    
    # n개 부족하면 already_disagree에서 보충
    if len(valid_files) < n:
        already = [f for f in os.listdir(input_dir)
                   if f.startswith('already_disagree') and f.endswith('.png')]
        valid_files += already[:n - len(valid_files)]
    
    selected = valid_files[:n]

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, fname in enumerate(selected):
        img = imageio.imread(os.path.join(input_dir, fname))
        pred1, conf1 = get_prediction(model1, img)
        pred2, conf2 = get_prediction(model2, img)

        parts = fname.replace('.png', '').split('_')
        orig_class = parts[2] if fname.startswith('already_disagree') else parts[1]

        orig_path = os.path.join(input_dir, fname.replace('.png', '_orig.png'))

        if os.path.exists(orig_path):
            orig_img = imageio.imread(orig_path)
            axes[i][0].imshow(orig_img)
            axes[i][0].set_title(f'Original\nTrue: {orig_class}', fontsize=11)
            axes[i][1].imshow(img)
            axes[i][1].set_title(
                f'Modified ({transformation})\nModel A: {pred1} ({conf1:.2f})\nModel B: {pred2} ({conf2:.2f})',
                fontsize=11
            )
        else:
            axes[i][0].imshow(img)
            axes[i][0].set_title(f'True: {orig_class}', fontsize=11)
            axes[i][1].text(0.5, 0.5,
                f'Model A: {pred1}\nconf: {conf1:.2f}\n\nModel B: {pred2}\nconf: {conf2:.2f}',
                ha='center', va='center', fontsize=13,
                transform=axes[i][1].transAxes
            )
            axes[i][1].axis('off')

        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle(f'Disagreement-inducing inputs ({transformation})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = f'../results/disagreements_{transformation}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

# 각 transformation별 3개씩 시각화
for transformation in ['light', 'occl', 'blackout']:
    input_dir = f'../models/generated_inputs/{transformation}'
    if os.path.exists(input_dir):
        visualize_disagreements(input_dir, transformation, n=3)
        print(f'{transformation} done!')

print('All visualizations saved to results/')