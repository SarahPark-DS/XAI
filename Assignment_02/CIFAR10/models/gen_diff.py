'''
usage: python gen_diff.py -h
'''

# from __future__ import print_function
# import argparse
# from keras.datasets import mnist
# from keras.layers import Input
# from scipy.misc import imsave
# from Model1 import Model1
# from Model2 import Model2
# from Model3 import Model3
# from configs import bcolors
# from utils import *

import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 # change the dataset to cifar10
from tensorflow.keras.models import load_model
import imageio # scipy.misc.imsave is deprecated, use imageio.imwrite instead
import os
import json
from tqdm import tqdm

from utils import *

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in CIFAR10 dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1], default=0, type=int) # change the target model to 0 and 1, since we only have two models in cifar10
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# random seed 고정
random.seed(42)
np.random.seed(42)


# input image dimensions
img_rows, img_cols, img_channels = 64, 64, 3
# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = tf.image.resize(x_test, (img_rows, img_cols)).numpy() # resize the input images to 64x64
x_test = x_test.astype("float32") / 255.0 # normalize the input images to [0, 1]

# load pre-trained models for cifar10
model1 = load_model('./model_a_initialized.h5')
model2 = load_model('./model_b_initialized.h5')

# init coverage table
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)
print("model_layer_dict1 size:", len(model_layer_dict1))
print("model_layer_dict2 size:", len(model_layer_dict2))

output_dir = f'./generated_inputs/{args.transformation}'
os.makedirs(output_dir, exist_ok=True)  # create the directory to save generated inputs

# ==============================================================================================
# start gen inputs
disagree_count = 0


for _ in tqdm(range(args.seeds), desc = "Processing seeds"):
    # gen_img = np.expand_dims(random.choice(x_test), axis=0)
    idx = random.randint(0, len(x_test) - 1)
    gen_img = np.expand_dims(x_test[idx], axis=0)
    orig_label_name = CIFAR10_CLASSES[y_test[idx][0]]

    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2 = np.argmax(model1.predict(gen_img, verbose=0)[0]), np.argmax(model2.predict(gen_img, verbose=0)[0])

    if label1 != label2:
        print(f'Already disagree: model1 = {CIFAR10_CLASSES[label1]}, model2 = {CIFAR10_CLASSES[label2]}')
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        disagree_count += 1

        img_deprocessed = deprocess_image(gen_img.copy())
        imageio.imwrite(f'{output_dir}/already_disagree_{orig_label_name}_{CIFAR10_CLASSES[label1]}_{CIFAR10_CLASSES[label2]}_{idx}.png', img_deprocessed)

        continue


    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    intermediate_model1 = tf.keras.Model(inputs=model1.input, outputs=model1.get_layer(layer_name1).output)
    intermediate_model2 = tf.keras.Model(inputs=model2.input, outputs=model2.get_layer(layer_name2).output)

    for iters in tqdm(range(args.grad_iterations), desc=f"Seed {orig_label_name} - Iterations", leave=False):
        gen_img_tensor = tf.Variable(gen_img, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(gen_img_tensor)
            pred1 = model1(gen_img_tensor)
            pred2 = model2(gen_img_tensor)

            # differential loss
            if args.target_model == 0:
                loss_diff = -args.weight_diff * tf.reduce_mean(pred1[:, orig_label]) + tf.reduce_mean(pred2[:, orig_label])
            else:
                loss_diff = tf.reduce_mean(pred1[:, orig_label]) - args.weight_diff * tf.reduce_mean(pred2[:, orig_label])

            # neuron coverage loss (shape에 따라 동적으로 슬라이싱)
            out1 = intermediate_model1(gen_img_tensor)
            out2 = intermediate_model2(gen_img_tensor)
            loss_nc1 = tf.reduce_mean(out1[..., index1]) if len(out1.shape) == 4 else tf.reduce_mean(out1[:, index1])
            loss_nc2 = tf.reduce_mean(out2[..., index2]) if len(out2.shape) == 4 else tf.reduce_mean(out2[:, index2])
            loss_nc = args.weight_nc * (loss_nc1 + loss_nc2)

            total_loss = loss_diff + loss_nc

        grads = tape.gradient(total_loss, gen_img_tensor)
        grads_value = normalize(grads).numpy()

        # apply transformation
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        gen_img += grads_value * args.step
        gen_img = np.clip(gen_img, 0, 1)

        pred1_new = np.argmax(model1.predict(gen_img, verbose=0)[0])
        pred2_new = np.argmax(model2.predict(gen_img, verbose=0)[0])

        if pred1_new != pred2_new:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            disagree_count += 1

            nc1 = neuron_covered(model_layer_dict1)
            nc2 = neuron_covered(model_layer_dict2)
            print(f'Disagree! model1={CIFAR10_CLASSES[pred1_new]}, model2={CIFAR10_CLASSES[pred2_new]}, '
                  f'coverage: model1={nc1[2]:.3f}, model2={nc2[2]:.3f}')

            img_deprocessed = deprocess_image(gen_img.copy())
            orig_deprocessed = deprocess_image(orig_img.copy())

            imageio.imwrite(f'{output_dir}/{args.transformation}_{orig_label_name}_{CIFAR10_CLASSES[pred1_new]}_{CIFAR10_CLASSES[pred2_new]}_{idx}_{iters}.png', img_deprocessed)
            imageio.imwrite(f'{output_dir}/{args.transformation}_{orig_label_name}_{CIFAR10_CLASSES[pred1_new]}_{CIFAR10_CLASSES[pred2_new]}_{idx}_{iters}_orig.png', orig_deprocessed)
            break


print(f'\nTotal disagreements: {disagree_count}')
nc1 = neuron_covered(model_layer_dict1)
nc2 = neuron_covered(model_layer_dict2)
print(f'Final coverage: model1={nc1[2]:.3f}, model2={nc2[2]:.3f}') 

results = {
    "transformation": args.transformation,
    "total_disagreements": disagree_count,
    "total_seeds": args.seeds,
    "final_coverage_model1": nc1[2],
    "final_coverage_model2": nc2[2],
    "hyperparameters": {
        "weight_diff": args.weight_diff,
        "weight_nc": args.weight_nc,
        "step": args.step,
        "grad_iterations": args.grad_iterations,
        "threshold": args.threshold,
        "target_model": args.target_model,
        "start_point": args.start_point,
        "occlusion_size": args.occlusion_size
    }
}

with open(f'{output_dir}/results_summary.json', 'w') as f:
    json.dump(results, f, indent=4)




