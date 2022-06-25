import tensorflow as tf
import numpy as np

from .dataset import Dataset

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events

class EggDatasetCropped(Dataset):
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
        
        input_window_samples = 1000


        ######################################################################
        # Now we create the model. To enable it to be used in cropped decoding
        # efficiently, we manually set the length of the final convolution layer
        # to some length that makes the receptive field of the ConvNet smaller
        # than ``input_window_samples`` (see ``final_conv_length=30`` in the model
        # definition).
        #

        import torch
        from braindecode.util import set_random_seeds
        from braindecode.models import ShallowFBCSPNet

        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True
        # Set random seed to be able to roughly reproduce results
        # Note that with cudnn benchmark set to True, GPU indeterminism
        # may still make results substantially different between runs.
        # To obtain more consistent results at the cost of increased computation time,
        # you can set `cudnn_benchmark=False` in `set_random_seeds`
        # or remove `torch.backends.cudnn.benchmark = True`
        seed = 20200220
        set_random_seeds(seed=seed, cuda=cuda)

        n_classes = 4
        # Extract number of chans from dataset
        n_chans = dataset[0][0].shape[0]

        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=30,
        )

        # Send model to GPU
        if cuda:
            model.cuda()


        ######################################################################
        # And now we transform model with strides to a model that outputs dense
        # prediction, so we can use it to obtain predictions for all
        # crops.
        #

        from braindecode.models import to_dense_prediction_model, get_output_shape

        to_dense_prediction_model(model)


        ######################################################################
        # To know the models’ receptive field, we calculate the shape of model
        # output for a dummy input.
        #

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


        ######################################################################
        # Cut the data into windows
        # -------------------------
        #
        # In contrast to trialwise decoding, we have to supply an explicit window size and
        # window stride to the ``create_windows_from_events`` function.
        #

        from braindecode.preprocessing import create_windows_from_events

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
            window_size_samples=input_window_samples,
            window_stride_samples=n_preds_per_input,
            drop_last_window=False,
            preload=True
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
