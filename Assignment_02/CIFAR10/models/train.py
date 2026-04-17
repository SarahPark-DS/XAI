#%%
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.keras import TqdmCallback
from model import resnet

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %%
# CIFAR-10 로드 및 전처리
IMG_SIZE = 64
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0
x_test = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Model A: ImageNet pretrained + seed = 111 + SGD + basig augmentation
print("Start training model A")
tf.random.set_seed(111)
model_a = resnet(input_shape = (IMG_SIZE, IMG_SIZE, 3))
model_a.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
## Data Augmentation: horizontal flip만 간단하게 수행
datagen_a = ImageDataGenerator(horizontal_flip = True)

checkpoint_a = ModelCheckpoint(
    filepath = "./model_a.h5",
    monitor = "val_accuracy",
    save_best_only = True,
    verbose = 1
)
model_a.fit(
    datagen_a.flow(x_train, y_train, batch_size = 8),
    epochs = 10,
    validation_data = (x_test, y_test),
    callbacks = [checkpoint_a, TqdmCallback(verbose = 1)]
)

print("Model A Saved!")


# Model 5: ImageNet pretrained + seed = 222 + AdamW + strong augmentation
print("Start training model B")
tf.random.set_seed(222)
model_b = resnet(input_shape = (IMG_SIZE, IMG_SIZE, 3))
model_b.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, weight_decay = 0.01),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

## Data Augmentation: model A보다 강력하게 데이터 증강 수행. 
datagen_b = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 15, # 이미지 최대 15도 회전
    width_shift_range = 0.1, # 가로로 10% 이동
    height_shift_range = 0.1, # 세로로 10% 이동
    zoom_range = 0.1) # 10% 확대/축소

checkpoint_b = ModelCheckpoint(
    filepath = "./model_b.h5",
    monitor = "val_accuracy",
    save_best_only = True,
    verbose = 1
)

model_b.fit(
    datagen_b.flow(x_train, y_train, batch_size = 8),
    epochs = 10,
    validation_data = (x_test, y_test),
    callbacks = [checkpoint_b, TqdmCallback(verbose = 1)]
)

print("Model B Saved!")
# %%
