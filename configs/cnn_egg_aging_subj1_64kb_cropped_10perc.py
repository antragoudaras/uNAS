from tensorflow_addons.optimizers import AdamW

from dataset import EggDatasetCropped
from config import TrainingConfig, BoundConfig
from configs.cnn_egg_aging_base_cropped import search_config, search_algorithm

training_config = TrainingConfig(
    dataset=EggDatasetCropped(subject_id=1, validation_split=0.1),
    epochs=130,
    batch_size=26,
    optimizer=lambda: AdamW(learning_rate=0.0005, weight_decay=1e-5),
    callbacks=lambda: [],
)

print('Tragos training config, train_values: {}, train_shape: {}'.format(training_config.dataset.train[0], training_config.dataset.train[0].shape), flush=True)

bound_config = BoundConfig(
    error_bound=0.05,
    peak_mem_bound=64000,
    model_size_bound=40000,
    mac_bound=50000000
)