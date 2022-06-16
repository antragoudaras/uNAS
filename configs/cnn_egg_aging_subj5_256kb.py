from tensorflow_addons.optimizers import AdamW

from dataset import EggDataset
from config import TrainingConfig, BoundConfig
from configs.cnn_egg_aging_base import search_config, search_algorithm

training_config = TrainingConfig(
    training_config = TrainingConfig(
    dataset=EggDataset(subject_id=5, validation_split=0.2),
    epochs=75,
    batch_size=26,
    optimizer=lambda: AdamW(learning_rate=0.0005, weight_decay=1e-5),
    callbacks=lambda: [],
)
)
print('Tragos training config, train_values: {}, train_shape: {}'.format(training_config.dataset.train[0], training_config.dataset.train[0].shape), flush=True)

bound_config = BoundConfig(
    error_bound=0.2,
    peak_mem_bound=256000,
    model_size_bound=160000,
    mac_bound=200000000
)