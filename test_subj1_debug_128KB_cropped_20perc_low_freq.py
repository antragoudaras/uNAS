import os
import pickle
import numpy as np
import tensorflow as tf
import argparse

from pathlib import Path

from sys import platform

if platform == "linux" or platform == "linux2":
    operating_sys = "linux"
else:
    operating_sys = "windows"

current_dir = os.getcwd()
if operating_sys == "windows":
    current_dir = current_dir + "\\configs\\cnn_egg_aging_subj1_128kb_cropped_20perc_low_freq.py"
else:
    current_dir = current_dir + "/configs/cnn_egg_aging_subj1_128kb_cropped_20perc_low_freq.py"    

parser = argparse.ArgumentParser("uNAS Search")
parser.add_argument("--config_file", type=str, default=current_dir, help="A config file describing the search parameters")
args = parser.parse_args()

configs = {}
exec(Path(args.config_file).read_text(), configs)

search_config = configs["search_config"]
dataset = configs["training_config"].dataset

objects = []
with (open("./artifacts/cnn_egg_cropped_low_freq/debug_subj1_agingevosearch_state.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

val_errors = []
test_errors = []
peak_mem_usage = []

for point in objects[0]:
    val_errors.append(point.val_error)
    test_errors.append(point.test_error)
    peak_mem_usage.append(point.resource_features[0])

minimum_arg = np.argmin(test_errors)
print("Minimum test error: {} in round: {}".format(min(test_errors), minimum_arg+1))
print("Minimum test error found: {} in round: {}".format(test_errors[minimum_arg], minimum_arg+1))
print("resource_features: {}, test_error: {}".format(objects[0][minimum_arg].resource_features, objects[0][minimum_arg].test_error))

new_curr_dir = os.getcwd()

best_model_keras = search_config.search_space.to_keras_model(objects[0][minimum_arg].point.arch, dataset.input_shape, dataset.num_classes)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

best_model_keras.compile(optimizer=configs["training_config"].optimizer(),
              loss=loss, metrics=[accuracy])

if operating_sys == "windows":
    best_model_keras.load_weights(f"{new_curr_dir}\\save_weights_dir_debug_subj1\\history_{minimum_arg}_model")
else:
    best_model_keras.load_weights(f"{new_curr_dir}/save_weights_dir_debug_subj1/history_{minimum_arg}_model")


test = dataset.test_dataset() \
    .batch(configs["training_config"].batch_size) \
    .prefetch(tf.data.experimental.AUTOTUNE)
_, test_acc = best_model_keras.evaluate(test, verbose=0)

print("Loaded test acc. : {}".format(1.0-test_acc))
print("Logged test acc. : {}".format(objects[0][minimum_arg].test_error))

# for value in test.take(100):
#     print("Dummy print")

def representative_dataset():
  for value in test.take(100):
    yield [value[0]]

converter = tf.lite.TFLiteConverter.from_keras_model(best_model_keras)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
tflite_best_model = converter.convert()

open("debug_subj1_toy_model.tflite", "wb").write(tflite_best_model)

# for round, point in enumerate(objects[0]):
#     model = search_config.search_space.to_keras_model(point.point.arch, dataset.input_shape, dataset.num_classes)
    
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    
#     model.compile(optimizer=configs["training_config"].optimizer(),
#                   loss=loss, metrics=[accuracy])

#     if operating_sys == "windows":
#         model.load_weights(f"{new_curr_dir}\\save_weights_dir_debug_subj1\\history_{round}_model")
#     else:
#         model.load_weights(f"{new_curr_dir}/save_weights_dir_debug_subj1/history_{round}_model")

    
#     test = dataset.test_dataset() \
#         .batch(configs["training_config"].batch_size) \
#         .prefetch(tf.data.experimental.AUTOTUNE)
#     _, test_acc = model.evaluate(test, verbose=0)
#     print("Loaded test acc. : {}".format(1.0-test_acc))
#     print("Logged test acc. : {}".format(point.test_error))
