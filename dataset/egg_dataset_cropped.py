import tensorflow as tf
import numpy as np

from .dataset import Dataset

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events

class EggDataset(Dataset):
    def __init__(self, subject_id=None, validation_split=0.1, seed=0):
        self.subject_id = subject_id
        
        train_set, test_set = self.preprocess_dataset()
        #Split x and y values originating from train_set & test_set
        def split_x_y(set_list):
            x_list = []
            y_list = []
            for i in range(len(set_list)):
                x_list.append(set_list[i][0])
                y_list.append(set_list[i][1])
            return np.expand_dims(x_list, axis=-1), np.array(y_list)

        x_train, y_train = split_x_y(train_set)
        x_test, y_test = split_x_y(test_set)

        
        x_train, x_val, y_train, y_val = \
            self._train_test_split(x_train, y_train, split_size=validation_split, random_state=seed, stratify=y_train)
        self.train = (x_train, y_train)
        self.val = (x_val, y_val)
        self.test = (x_test, y_test)
    
    def preprocess_dataset(self):
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[self.subject_id])
        low_cut_hz = 4.  # low cut frequency for filtering
        high_cut_hz = 38.  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000

        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
        ]
        # Transform the data
        preprocess(dataset, preprocessors)
        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        test_set = splitted['session_E']
        return train_set, test_set

    def train_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.train)

    def validation_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.val)
        
    def test_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.test)

    @property
    def num_classes(self):
        return 4

    @property
    def input_shape(self):
        return (22, 1000, 1)
