from braindecode.datasets import MOABBDataset
for i in range(9):
    subject_id = i + 1
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])