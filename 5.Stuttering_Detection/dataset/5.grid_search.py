# %%
import os
import random
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from sklearn.metrics import f1_score, confusion_matrix


from time import time
from glob import glob
from preprocessing import compute_linear_matrix, get_mfccs_training, get_mfcc, LABELS
from itertools import product
from functools import partial
from typing import Iterable, Any

from keras.callbacks import Callback,ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


TEAM_FOLDER = os.path.join('.', 'TEAM10')

if not os.path.exists(TEAM_FOLDER):
    os.makedirs(TEAM_FOLDER)


# %%
# train_ds_pure = tf.data.Dataset.list_files(['train_sep28_7/*'])
# val_ds_pure = tf.data.Dataset.list_files(['val_sep28_7/*'])
# test_ds_pure = tf.data.Dataset.list_files(['test_sep28_7/*'])
# test_ds_pure1 = tf.data.Dataset.list_files(['fluency_balanced_7/*'])

# # Shuffle the dataset
# train_ds_shuffled = train_ds_pure.shuffle(buffer_size=len(train_ds_pure), seed=seed)
# val_ds_shuffled = val_ds_pure.shuffle(buffer_size=len(val_ds_pure), seed=seed)
# test_ds_shuffled = test_ds_pure.shuffle(buffer_size=len(test_ds_pure), seed=seed)
# test_ds_shuffled1 = test_ds_pure1.shuffle(buffer_size=len(test_ds_pure1), seed=seed)

# %%
    # true_labels = []
    # directory = './train_sep28_7'
    # for filename in os.listdir(directory):
    #     label = tf.strings.split(tf.strings.split(filename, '/')[-1], '_')[0]
    #     true_labels.append(label.numpy())

    
    # from sklearn.utils import class_weight

    # # Assuming you have the true labels in an array called 'true_labels'
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(true_labels), y=true_labels)


    # class_weights_dict = dict(enumerate(class_weights))
    # class_weights_dict
class_weights_dict = {0: 0.9788884879105016, 1: 1.0220422004521477}

# %%
# true_labels

# %%
def preprocess(filename):
    signal, label = get_frozen_spectrogram(filename)
    signal.set_shape(SHAPE)
    signal = tf.expand_dims(signal, -1)
    signal = tf.image.resize(signal, [32,32])
    label_id = tf.argmax(label == LABELS)

    return signal, label_id

def get_model(alpha, model_filter, input_shape):
    '''Returns the model'''
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(filters=int(model_filter * alpha), kernel_size=[3, 3], strides=[2, 2], use_bias=False, padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=int(model_filter * alpha), kernel_size=[3, 3], strides=[1, 1], use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=int(model_filter * alpha), kernel_size=[3, 3], strides=[1, 1], use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=len(LABELS)),
        tf.keras.layers.Softmax()
    ])

def save_model(model, path):
    '''Saves the model'''
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path)

def convert_model(model, model_path, tflite_path, model_name):
    '''Converts the saved model into tflite model and saves it (also zip version)'''
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()

    if not os.path.exists(tflite_path):
        os.makedirs(tflite_path)
    tflite_model_path = os.path.join(tflite_path, f'{model_name}.tflite')

    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)

    with zipfile.ZipFile(f'{tflite_model_path}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(tflite_model_path, 'model10.tflite')
    
    sizes = {
        'tflite_model_size' : os.path.getsize(f'{tflite_model_path}') / 1024.0,
        'tflite_zip_model_size' : os.path.getsize(f'{tflite_model_path}.zip') / 1024.0
    }

    return tflite_model_path, sizes

def print_results(name, parameters, model_accuracies, sizes):
    '''saves results in a csv'''
    output_dict = {
                    'model_name': name,
                    **parameters,
                    **model_accuracies,
                    **sizes
            }
    print(output_dict.values())


def F1Score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# %%
def training(downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, num_coefficients, lower_frequency,
 upper_frequency, batch_size, initial_learning_rate, end_learning_rate, epochs, model_filter, alpha, initial_sparsity,
 final_sparsity):
    
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # %%
    train_ds_pure = tf.data.Dataset.list_files(['train_sep28_7/*'])
    val_ds_pure = tf.data.Dataset.list_files(['val_sep28_7/*'])
    test_ds_pure = tf.data.Dataset.list_files(['test_sep28_7/*'])
    test_ds_pure1 = tf.data.Dataset.list_files(['fluency_balanced_7/*'])

    # Shuffle the dataset
    train_ds_shuffled = train_ds_pure.shuffle(buffer_size=len(train_ds_pure), seed=seed)
    val_ds_shuffled = val_ds_pure.shuffle(buffer_size=len(val_ds_pure), seed=seed)
    test_ds_shuffled = test_ds_pure.shuffle(buffer_size=len(test_ds_pure), seed=seed)
    test_ds_shuffled1 = test_ds_pure1.shuffle(buffer_size=len(test_ds_pure1), seed=seed)

    # frame_step_in_s = frame_length_in_s
    # upper_frequency = downsampling_rate/2
    # num_coefficients = num_mel_bins
    print('inside', frame_step_in_s, upper_frequency, num_coefficients)
    # ******************* Parameter definition ***********************************************************
    PREPROCESSING_ARGS = {
        'downsampling_rate': downsampling_rate,
        'frame_length_in_s': frame_length_in_s,
        'frame_step_in_s': frame_step_in_s, 
        'num_mel_bins': num_mel_bins,
        'num_coefficients': num_coefficients,
        'lower_frequency': lower_frequency,
        'upper_frequency': upper_frequency,
    }
    
    # ******************* Preprocessing ***********************************************************
    global SHAPE, get_frozen_spectrogram

    get_frozen_spectrogram = partial(get_mfccs_training, **PREPROCESSING_ARGS)

    for mfcc, label in train_ds_pure.map(get_frozen_spectrogram).take(1):
        SHAPE = mfcc[:num_coefficients].shape
        print(label)


    train_ds = train_ds_shuffled.map(preprocess).batch(batch_size).cache()
    val_ds = val_ds_shuffled.map(preprocess).batch(batch_size)
    test_ds = test_ds_shuffled.map(preprocess).batch(batch_size)
    test_ds1 = test_ds_shuffled1.map(preprocess).batch(batch_size)
    



    #get shape for model input
    for example_batch, example_labels in train_ds.take(1):
        input_shape = example_batch.shape[1:]
    
    
    # ******************* Model And Training *******************************************************
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    # metrics = [tf.metrics.SparseCategoricalAccuracy()]
    # metrics = [tf.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    metrics=[tf.metrics.SparseCategoricalAccuracy()]
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=end_learning_rate,
        decay_steps=len(train_ds) * epochs,
    )

    optimizer = tf.optimizers.Adam(learning_rate=linear_decay)

    begin_step = int(len(train_ds_pure) * epochs * 0.2)
    end_step = int(len(train_ds_pure) * epochs)

    # Pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity= initial_sparsity,
            final_sparsity= final_sparsity,
            begin_step=begin_step,
            end_step=end_step
        )
    }


    model = tfmot.sparsity.keras.prune_low_magnitude(get_model(alpha, model_filter, input_shape), **pruning_params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, class_weight=class_weights_dict)

    # Compute F1 score on test set
    predictions = model.predict(test_ds)
    predicted_labels = np.argmax(predictions, axis=1)  # Assuming a classification task
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)  # Get true labels from test dataset

    f1_m = f1_score(true_labels, predicted_labels, average='macro')
    f1_w = f1_score(true_labels, predicted_labels, average='weighted')  # Compute F1 score
    f1_b = f1_score(true_labels, predicted_labels, average='binary')  # Compute F1 score

    cm =confusion_matrix(true_labels, predicted_labels)
    print(cm)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    print('tn, fp, fn, tp:',tn, fp, fn, tp )
    f1_cm = tp/(tp+(fn+fp)/2)
    print('********sep********f1_m, f1_w, f1_b,f1_cm:',f1_m, f1_w, f1_b,f1_cm)


    # Compute F1 score on test set1
    predictions1 = model.predict(test_ds1)
    predicted_labels1 = np.argmax(predictions1, axis=1)  # Assuming a classification task
    true_labels1 = np.concatenate([y for x, y in test_ds1], axis=0)  # Get true labels from test dataset

    f1_m1 = f1_score(true_labels1, predicted_labels1, average='macro')
    f1_w1 = f1_score(true_labels1, predicted_labels1, average='weighted')  # Compute F1 score
    f1_b1 = f1_score(true_labels1, predicted_labels1, average='binary')  # Compute F1 score

    cm1 =confusion_matrix(true_labels1, predicted_labels1)
    print(cm1)
    tn1, fp1, fn1, tp1 = confusion_matrix(true_labels1, predicted_labels1).ravel()
    print('tn, fp, fn, tp:',tn1, fp1, fn1, tp1 )
    f1_cm1 = tp1/(tp1+(fn1+fp1)/2)

    print('******fluency**********f1_m1, f1_w1, f1_b1,f1_cm1:',f1_m1, f1_w1, f1_b1,f1_cm1)
    
    # ******************* Results ***************************************************************
    _, training_accuracy= model.evaluate(train_ds)
    _, validation_accuracy = model.evaluate(val_ds)
    _, test_accuracy = model.evaluate(test_ds)
    _, test_accuracy1= model.evaluate(test_ds1)

    accuracies = {
        'training_accuracy_tf': training_accuracy*100,
        'validation_accuracy_tf': validation_accuracy*100,
        'test_accuracy_tf': test_accuracy*100,
        'fluency_accuracy': test_accuracy1*100,
        'sep28_f1': f1_cm*100,
        'fluency_f1': f1_cm1*100

    }

    # fig = plt.figure()
    # plt.plot(history.history['loss'], c='r')
    # plt.plot(history.history['val_loss'], c='b')
    # plt.title(f"Training and Validation Loss")
    # plt.legend(['train_loss', 'val_loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    return model, accuracies 



# %%
# ARGUMENTS = {
#         'downsampling_rate': [16000, 32000, 48000, 98000],
#         'frame_length_in_s': [0.08, 0.016, 0.032, 0.48, 0.98],
#         'frame_step_in_s': [0.008, 0.016, 0.032],
#         'num_mel_bins': [20, 40],
#         'num_coefficients': [20,40],
#         'lower_frequency': [20],
#         'upper_frequency': [16000],
#         'batch_size': [20],
#         'initial_learning_rate': [0.01],
#         'end_learning_rate': [1.e-5],
#         'epochs': [20],
#         'model_filter': [64, 128],
#         'alpha': [0.1, 0.2, 0.3],
#         'initial_sparsity': [0.2],
#         'final_sparsity': [0.5, 0.6, 0.7]
#     }

ARGUMENTS = {
        'downsampling_rate': [16000],
        'frame_length_in_s': [0.032],
        'frame_step_in_s': [0.008, 0.016, 0.032],
        'num_mel_bins': [40],
        'num_coefficients': [20, 40],
        'lower_frequency': [20],
        'upper_frequency': [8000],
        
        'batch_size': [20, 64, 128],
        'initial_learning_rate': [0.01],
        'end_learning_rate': [1.e-5],
        'epochs': [20, 30],
        'model_filter': [64, 128],
        'alpha': [0.2],
        'initial_sparsity': [0.2],
        'final_sparsity': [0.6]
    }


# %%
# from itertools import product
# from typing import Iterable, Any, Dict

# def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
#     for params in product(*parameters.values()):
#         yield dict(zip(parameters.keys(), params))
        
# def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
#     for params in product(*parameters.values()):
#         yield dict(zip(parameters.keys(), params))

import itertools
# Generate all combinations of hyperparameters
keys = ARGUMENTS.keys()
values = ARGUMENTS.values()
combinations = list(itertools.product(*values))

# Perform grid search
for params in combinations:
    # Create a dictionary with the current set of hyperparameters
    hyperparameters = dict(zip(keys, params))

# %%
# for parameters in grid_parameters(ARGUMENTS):
    model, accuracies = training(**hyperparameters)

    MODEL_NAME = 'model10'
    MODEL_PATH = os.path.join(TEAM_FOLDER, 'model', f'{MODEL_NAME}')
    save_model(model, MODEL_PATH)

    TFLITE_PATH = os.path.join(TEAM_FOLDER, 'tflite_model')
    TFLITE_NAME, sizes = convert_model(model, MODEL_PATH, TFLITE_PATH, MODEL_NAME)

    print('****************results*********************')
    print_results(MODEL_NAME, hyperparameters, accuracies, sizes)

    data = {**hyperparameters,**accuracies,**sizes}

    # import pdb
    # pdb.set_trace()

    # Create a DataFrame from the dictionary
    df_new = pd.DataFrame([data])

    # Check if the file exists data_preparation_7_all_noConfidence_balanced_target
    file_path = '1.data_preparation_7_all_noConfidence_balanced_target.csv' 
    if os.path.isfile(file_path):
        # Read the existing Excel file if it exists
        df_existing = pd.read_csv(file_path)

        # Append the new data to the existing DataFrame
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # If the file doesn't exist, use the new DataFrame as is
        df_updated = df_new

    # Save the updated DataFrame to the Excel file
    df_updated.to_csv(file_path, index=False)

# %%
# model, accuracies = training(**ARGUMENTS)

# MODEL_NAME = 'model10'
# MODEL_PATH = os.path.join(TEAM_FOLDER, 'model', f'{MODEL_NAME}')
# save_model(model, MODEL_PATH)

# TFLITE_PATH = os.path.join(TEAM_FOLDER, 'tflite_model')
# TFLITE_NAME, sizes = convert_model(model, MODEL_PATH, TFLITE_PATH, MODEL_NAME)

# print('****************results*********************')
# print_results(MODEL_NAME, ARGUMENTS, accuracies, sizes)


# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=5c74a7a1-d505-4b2b-a4fa-efea0c0f20da' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


