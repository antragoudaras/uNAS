from tensorflow_addons.optimizers import AdamW

from cnn import CnnSearchSpace
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from dataset import EggDataset
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=EggDataset(subject_id=3),
    epochs=30,
    batch_size=100,
    optimizer=lambda: AdamW(lr=0.0005, weight_decay=1e-5),
    callbacks=lambda: [],
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    population_size=50,
    rounds=1000,
    checkpoint_dir="artifacts/cnn_egg"
)

bound_config = BoundConfig(
    error_bound=20.0,
    peak_mem_bound=60000,
    model_size_bound=40000,
    mac_bound=40000000
)