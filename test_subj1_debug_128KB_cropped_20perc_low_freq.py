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

new_curr_dir = os.getcwd()
for round, point in enumerate(objects[0]):
    model = search_config.search_space.to_keras_model(point.point.arch, dataset.input_shape, dataset.num_classes)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    
    model.compile(optimizer=configs["training_config"].optimizer(),
                  loss=loss, metrics=[accuracy])

    if operating_sys == "windows":
        model.load_weights(f"{new_curr_dir}\\save_model_weights_in_history\\history_{round}_model")
    else:
        model.load_weights(f"{new_curr_dir}/save_model_weights_in_history/history_{round}_model")

    
    test = dataset.test_dataset() \
        .batch(configs["training_config"].batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    _, test_acc = model.evaluate(test, verbose=0)
    print("Loaded test acc. : {}".format(1.0-test_acc))
    print("Logged test acc. : {}".format(point.test_error))
    
    # val_errors.append(point.val_error)
    # test_errors.append(point.test_error)
    # peak_mem_usage.append(point.resource_features[0])

