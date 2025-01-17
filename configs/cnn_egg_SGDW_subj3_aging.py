import tensorflow_addons as tfa

from cnn import CnnSearchSpace
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from dataset import EggDataset
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=EggDataset(subject_id=3),
    epochs=75,
    batch_size=26,
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.01, momentum=0.9, weight_decay=1e-5),
    callbacks=lambda: [],
)

print('Tragos training config, train_values: {}, train_shape: {}'.format(training_config.dataset.train[0], training_config.dataset.train[0].shape), flush=True)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    population_size=100,
    rounds=2000,
    checkpoint_dir="artifacts/cnn_egg"
)

bound_config = BoundConfig(
    error_bound=0.2,
    peak_mem_bound=60000,
    model_size_bound=40000,
    mac_bound=50000000
)