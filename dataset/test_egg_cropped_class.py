from egg_dataset_cropped import EggDatasetCropped
import numpy as np 

dataset_egg = EggDatasetCropped(subject_id=3, validation_split=0.1)
train_data = dataset_egg.train_dataset()
val_data = dataset_egg.validation_dataset()
test_data = dataset_egg.test_dataset()
print("Dummy print")