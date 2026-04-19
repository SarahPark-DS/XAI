#%%
import argparse
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.keras import TqdmCallback
from model import resnet

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# argparse 설정
parser = argparse.ArgumentParser(description='Train two ResNet models on CIFAR-10')
parser.add_argument('--img_size',   type=int,   default=64,         help='Input image size')
parser.add_argument('--epochs',     type=int,   default=10,         help='Number of epochs')
parser.add_argument('--batch_size', type=int,   default=8,          help='Batch size')
parser.add_argument('--weights',    type=str,   default=None,       help='Pretrained weights (None or imagenet)')
parser.add_argument('--seed_a',     type=int,   default=111,        help='Random seed for Model A')
parser.add_argument('--seed_b',     type=int,   default=222,        help='Random seed for Model B')
parser.add_argument('--lr_a',       type=float, default=0.01,       help='Learning rate for Model A')
parser.add_argument('--lr_b',       type=float, default=0.001,      help='Learning rate for Model B')
args = parser.parse_args()


# CIFAR-10 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.image.resize(x_train, (args.img_size, args.img_size)).numpy() / 255.0
x_test  = tf.image.resize(x_test,  (args.img_size, args.img_size)).numpy() / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

# Model A: seed_a + SGD + basic augmentation
print("Start training model A")
tf.keras.utils.set_random_seed(args.seed_a)
model_a = resnet(input_shape=(args.img_size, args.img_size, 3), weights=args.weights)
model_a.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr_a, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
datagen_a = ImageDataGenerator(horizontal_flip=True)
checkpoint_a = ModelCheckpoint(
    filepath='./model_a_initialized.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
model_a.fit(
    datagen_a.flow(x_train, y_train, batch_size=args.batch_size),
    epochs=args.epochs,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_a, TqdmCallback(verbose=1)],
    verbose=0
)
print("Model A Saved!")

# Model B: seed_b + AdamW + strong augmentation
print("Start training model B")
tf.keras.utils.set_random_seed(args.seed_b)
model_b = resnet(input_shape=(args.img_size, args.img_size, 3), weights=args.weights)
model_b.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr_b, weight_decay=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
datagen_b = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
checkpoint_b = ModelCheckpoint(
    filepath='./model_b_initialized.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
model_b.fit(
    datagen_b.flow(x_train, y_train, batch_size=args.batch_size),
    epochs=args.epochs,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_b, TqdmCallback(verbose=1)],
    verbose=0
)
print("Model B Saved!")