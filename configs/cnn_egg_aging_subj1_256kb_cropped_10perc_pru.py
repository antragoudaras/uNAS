from config import PruningConfig
from configs.cnn_egg_aging_subj1_256kb_cropped_10perc import training_config, bound_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=13,
    finish_pruning_by_epoch=78,
    min_sparsity=0.05,
    max_sparsity=0.95
)
