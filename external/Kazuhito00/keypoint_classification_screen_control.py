# -*- coding: utf-8 -*-
# Train gestures.
# Created by wengc on 04/08/2022

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier_screen_control.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier_screen_control.tflite'

try:
    with open(dataset, 'r', encoding='utf-8'):
        pass
except FileNotFoundError:
    import os

    os.chdir('./external/Kazuhito00')

# cweng: The coordinates are post-processed according to
# https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png
RANDOM_SEED = 42
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0,))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
plt.plot(np.sort(y_dataset))


def get_compiled_model():
    NUM_CLASSES = len(set(y_dataset))
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return _model


def get_callback_list():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_best_only=True,
                                                     save_weights_only=True)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.00001,
                                                     patience=5, mode='min',
                                                     verbose=1)
    return [cp_callback, es_callback, reduce_lr]


def plot_history(_history):
    plt.figure(111)
    plt.plot(_history.history['accuracy'], label='accuracy')
    plt.plot(_history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.figure(112)
    plt.plot(_history.history['loss'], label='loss')
    plt.plot(_history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')


model = get_compiled_model()
history = model.fit(X_train, y_train,
                    epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=get_callback_list())

plot_history(history)

"""
misc
"""


def do_test_on_data():
    model.evaluate(X_test, y_test, batch_size=128)

    # Test skewness on each class
    for class_id in range(len(set(y_dataset))):
        print(f"##############  On id {class_id} ##############")
        val_loss, val_acc = model.evaluate(X_dataset[np.where(y_dataset == class_id)],
                                           y_dataset[np.where(y_dataset == class_id)],
                                           batch_size=128)


do_test_on_data()

# Convert to tflite which may speed up the inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb', encoding='utf-8').write(tflite_quantized_model)
