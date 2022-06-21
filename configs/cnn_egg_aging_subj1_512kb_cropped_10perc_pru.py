from config import PruningConfig
from configs.cnn_egg_aging_subj1_512kb_cropped_10perc import training_config, bound_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=58,
    finish_pruning_by_epoch=116,
    min_sparsity=0.10,
    max_sparsity=0.90
)
