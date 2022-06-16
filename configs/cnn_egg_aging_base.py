from cnn import CnnSearchSpace
from config import AgingEvoConfig

from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    population_size=100,
    rounds=4000,
    checkpoint_dir="artifacts/cnn_egg"
)