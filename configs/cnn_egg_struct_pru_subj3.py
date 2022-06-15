from config import PruningConfig
from configs.cnn_egg_aging_subj3 import training_config, bound_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=10,
    finish_pruning_by_epoch=57,
    min_sparsity=0.05,
    max_sparsity=0.8
)