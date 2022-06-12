from egg_dataset import EggDataset

dataset_egg = EggDataset(subject_id=2)
train_data = dataset_egg.train_dataset()
val_data = dataset_egg.validation_dataset()
test_data = dataset_egg.test_dataset()
print("Dummy print")