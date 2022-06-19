from tensorflow_addons.optimizers import AdamW

from dataset import EggDatasetCropped
from config import TrainingConfig, BoundConfig
from configs.cnn_egg_aging_base_cropped import search_config, search_algorithm

training_config = TrainingConfig(
    dataset=EggDatasetCropped(subject_id=4, validation_split=0.3),
    epochs=75,
    batch_size=173,
    optimizer=lambda: AdamW(learning_rate=0.0005, weight_decay=1e-5),
    callbacks=lambda: [],
)

print('Tragos training config, train_values: {}, train_shape: {}'.format(training_config.dataset.train[0], training_config.dataset.train[0].shape), flush=True)

bound_config = BoundConfig(
    error_bound=0.05,
    peak_mem_bound=256000,
    model_size_bound=320000,
    mac_bound=400000000
)