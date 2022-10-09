from cnn import CnnSearchSpace
from config import AgingEvoConfig

from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    population_size=5,
    sample_size=1,
    rounds=10,
    checkpoint_dir="artifacts/cnn_egg_cropped_low_freq"
)