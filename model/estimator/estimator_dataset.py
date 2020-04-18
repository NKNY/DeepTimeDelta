import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from six import iteritems
import tensorflow as tf
import numpy as np
import random
from model.utils.backwards_compatibility import merge_dicts

try:
    import itertools.izip as zip
except ImportError:
    pass

try:
    from sagemaker_tensorflow import PipeModeDataset
except ModuleNotFoundError:
    pass

try:
    # Tensorflow 1.13 (local and insy)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
except AttributeError:
    # Tensorflow 1.12 (sagemaker)
    AUTOTUNE = tf.contrib.data.AUTOTUNE

import multiprocessing
num_cpu = multiprocessing.cpu_count()

class RecommendationDataSet(object):

    def __init__(self, data_dir, subset, X_cols_to_use, input_data_format="npy", file_types=("X", "y", "seq_lens"),
                 shuffle_buffer_size=1, delta_t_mean=None, delta_t_std=None):

        """
        Object that is used to read the X.npy, y.npy, seq_lens.npy and user_ids.npy files
        that are assumed to be in the datadit/subset folder. These files are then combined,
        optionally shuffled and transformed a tf.data.Dataset iterator that returns a feature
        dict normally with keys {'uid', 'iid', 'delta_t' - optional, 'seq_lens', 'user_ids'}
        as well as a tensor of labels. All tensors are of length batch_size in the first
        dimension as specified in RecommendationDataSet.make_batch.
        Parameters
        ----------
        data_dir : str or Path
            Location of absolute path where data set's subset directories are located.
        subset : {'train', 'validation', 'test'}
            Subset of the data to use from the data set.
        X_cols_to_use : dict
            k:v pairs where k is the index of the column and v is the name of the feature that is
            assigned to the column.
        input_data_format : {"npy", "tfrecords"}
            Type of data to process - datasets currently saved as numpy arrays or tfrecords.
        file_types : iterable
            Names (without extension) for which files to include as the folder may contain other files.
        shuffle_buffer_size : int
            Number of samples coming from the queue from which the next sample is selected. If 1 no shuffling is
            performed. If above 1, files are also shuffled if there are multiple files.
            Note that shuffling the files is not supported via pipemode.
        delta_t_mean : float or None
            If provided the `delta_t` feature of the data (if used) will have `delta_t_mean` subtracted from every entry.
        delta_t_std : float or None
            If provided the `delta_t` feature of the data (if used) will be divided by `delta_t_mean` in every entry.
        """

        self.data_dir = data_dir
        self.subset = subset
        self.file_types = file_types
        self.X_cols_to_use = X_cols_to_use
        self.input_data_format = input_data_format
        self.all_keys_to_features = {
            "X": tf.FixedLenFeature((60), tf.int64),
            "y": tf.FixedLenFeature((20), tf.int64),
            "seq_lens": tf.FixedLenFeature((), tf.int64),
            "weights": tf.FixedLenFeature((20), tf.float32)
        }
        self.keys_to_features = {k:v for k,v in self.all_keys_to_features.items() if k in self.file_types}
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle = self.shuffle_buffer_size > 1
        self.delta_t_mean = delta_t_mean
        self.delta_t_std = delta_t_std

    def get_numpy_filenames(self):

        """
        Checks whether the subset is a supported name. If so it returns a list of file paths (str)
        that we are looking to load (data_dir/subset/file_type.npy).

        Returns
        -------
        list
            File paths (str) that we are looking to load (data_dir/subset/file_type.npy).
        """

        if self.subset in ["train", "validation", "test"]:
            subset_folder = os.path.join(self.data_dir, self.subset)
            return [os.path.join(subset_folder, x + '.npy') for x in self.file_types]
        else:
            raise ValueError("Invalid data subset '%s'" % self.subset)

    def numpy_parser(self, filename):
        return np.load(filename)

    def load_numpy_dataset(self):

        """
        Load dataset saved in dataset_name/subset/filename.npy

        Returns
        -------
        dataset : tf.data.Dataset
            Generator like pointer to dataset
        """

        # Load all filenames that the model is expecting as specified during initialisation
        filenames = self.get_numpy_filenames()

        if self.shuffle:
            random.shuffle(filenames)

        # k=[*file_types], v: loaded numpy array of the corresponding type
        files = dict(zip(self.file_types, [self.numpy_parser(x) for x in filenames]))
        dataset = tf.data.Dataset.from_tensor_slices(files)
        return dataset


    def load_tfrecords_dataset(self):

        """
        Load dataset saved in dataset_name/subset/subset.tfrecords

        Assumes a specific structure of tf records. Namely:
            "X" - flattened 1-dim length 60
            "y" - 1-dim length 20
            "seq_lens" - scalar
            "user_ids" - scalar
        Parameters
        ----------
        example_proto
            Pointer to a TFRecord
        Returns
        -------
        tf.data.Dataset
            Pointer to k:v dictionary where value is the tensor. Keys: X, y, seeq_lens, user_ids
        """

        if self.subset in ["train", "validation", "test"]:
            # List all the tfrecord files under the subset's path
            files = tf.data.Dataset.list_files(
                os.path.join(self.data_dir, self.subset, "*.tfrecords"),
                shuffle=self.shuffle
            )
            # Load the object pointing to TFRecords (multiple tfrecord files added in random order)
            dataset = files.apply(
                tf.contrib.data.parallel_interleave(
                    lambda filename: tf.data.TFRecordDataset(filename),
                    cycle_length=num_cpu,
                    buffer_output_elements=1,

                ),
            )

            # Process individual samples from tfrecords to rank 1-3 tensors
            return dataset.map(lambda x: self.example_to_features(x, self.keys_to_features), num_parallel_calls=num_cpu)

        else:
            raise ValueError("Invalid data subset '%s'" % self.subset)

    def load_tfrecords_pipemode_dataset(self):

        """
        STREAM dataset saved under the path "pipemode_SUBSET".

        Assumes a specific structure of tf records. Namely:
            "X" - flattened 1-dim length 60
            "y" - 1-dim length 20
            "seq_lens" - scalar
        Parameters
        ----------
        example_proto
            Pointer to a TFRecord
        Returns
        -------
        tf.data.Dataset-like SageMaker PipeModeDataset
            Pointer to k:v dictionary where value is the tensor. Keys: X, y, seq_lens
        """

        if self.subset in ["train", "validation", "test"]:

            dataset = PipeModeDataset(channel="pipemode_" + self.subset, record_format="TFRecord")
            # Process individual samples from tfrecords to rank 1-3 tensors
            return dataset.map(lambda x: self.example_to_features(x, self.keys_to_features))

        else:
            raise ValueError("Invalid data subset '%s'" % self.subset)


    def example_to_features(self, example_proto, keys_to_features):

        """
        Processes individual tf examples according to provided mapping. Reshapes a single X from
        a flattened array into rank 2 tensor. Currently assumes the shape of X to be (20,3).

        Parameters
        ----------
        example_proto
            A single entry in a TFRecordDataset, passed via dataset's map method.
        keys_to_features : dict
            Mapping from tensor names to tf.FixedLenFeatures of tf.int64.

        Returns
        -------
        features : tf.data.Dataset
            Generator like pointer to the data

        """

        features = tf.parse_single_example(example_proto, keys_to_features)
        features["X"] = tf.reshape(features["X"], (20, 3))
        return features

    def scale_dt(self, dt):

        """
        If `delta_t_mean` previously provided subtracts its value from every entry in the provided tensor.
        If `delta_t_std` previously provided divides every entry by it in the provided tensor.
        Providing both values (e.g. calculated from the training data) results it data standardisation.
        Parameters
        ----------
        dt : float
            Float tensor containing `delta_t` values that be rescaled.

        Returns
        -------
        dt : float
            Rescaled tensor of the same shape as the input.

        """

        if self.delta_t_mean is not None:
            dt -= self.delta_t_mean
        if self.delta_t_std is not None:
            dt /= self.delta_t_std
        return dt

    def preprocess_sample(self, features):

        """
        Splits input ? x timesteps x 3 tensor into 2 or 3 tensors corresponding to features used by the
        model and returns them together with the remaining inputs (usually y, seq_lens, user_ids)
        Parameters
        ----------
        features
            Pointer to dict with keys containing at least X and usually also y, seq_lens, user_ids
        Returns
        -------
            Pointer to k:v dictionary where value is the transformed tensor. Keys are currently expected
            to be uid, iid, (delta_t), y, seq_lens, user_ids. Presence or absence of delta_t is based
            on self.X_cols_to_use
        """
        # Remove X from features to keep
        features_to_retain = {k:v for k,v in features.items() if k != "X"}
        # Split X into uid, iid, (delta_t if specified)
        split_features = {v: features["X"][:,int(k)] for k,v in iteritems(self.X_cols_to_use)}

        # If using `delta_t` in the model and it should be rescaled apply the rescaling
        if "delta_t" in split_features:
            if self.delta_t_mean is not None or self.delta_t_std is not None:
                split_features["delta_t"] = self.scale_dt(tf.cast(split_features["delta_t"], tf.float64))

        # Combine the outputs
        merged_features = merge_dicts(features_to_retain, split_features)
        # Convert them to int32 from int64
        cast_features = {k:tf.cast(v, tf.int32) if v.dtype == tf.int64 else v for k,v in merged_features.items()}
        return cast_features

    def make_batch(self, batch_size):

        """
        Called in a peculiar tensorflow fashion whereby everything until make_one_shot_iterator is executed
        only once and pointers to the outputs of iterator.get_next are returned. Iterator.get_next is also
        explicitly called only once but every time an batch is requested by the main program the contents
        of the pointer are returned, removed from the pointer location and instead the next batch is put
        in the location.

        Parameters
        ----------
        batch_size : int
            Size of the batch that is to be shared between of the towers. Same as number of sequences.
        Returns
        -------
        features : dict
            Keys the requested columns e.g. 'uid' and 'iid' as well as 'seq_lens' and 'user_ids'. Each value
            is a tf.int32 tensor associated with the key, first dimension of length batch_size.
        labels : tf.int32
            Label tensor of shape batch_size x timesteps.
        """

        if self.input_data_format == "npy":
            dataset = self.load_numpy_dataset()
        elif self.input_data_format == "tfrecords":
            dataset = self.load_tfrecords_dataset()
        elif self.input_data_format == "tfrecords_pipemode":
            dataset = self.load_tfrecords_pipemode_dataset()
        else:
            raise ValueError("Invalid data format '%s'" % self.input_data_format)

        # Cache dataset
        dataset = dataset.cache()

        # For the value of parameter shuffle consult tf.Data.dataset.shuffle
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(self.shuffle_buffer_size))

        # Combine multiple samples into a batch
        # Apply the split into uid, iid, delta_t while maintaining seq_lens and user_ids
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=self.preprocess_sample, batch_size=batch_size, num_parallel_calls=num_cpu
            )
        )
        dataset = dataset.prefetch(AUTOTUNE)
        # Build iterator from dataset
        iterator = dataset.make_one_shot_iterator()
        # Op to retrieve the next batch
        next_batch = iterator.get_next()
        # Remove k:v associated with keys 'X' and 'y'
        features = {k:v for k,v in iteritems(next_batch) if k not in ["y"]}

        # Keeping as reminder if ever still need to use initializable iterator
        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        return features, next_batch["y"]

class EstimatorDatasetLastfm10pcFinal(object):
    num_users = 1001 # Actually 986 but users not reindexed - have to name max+1 for the embedding matrix
    num_items = 407305
    num_samples = {"train": 86667,
                   "validation": 4471,
                   "test": 4304}

class EstimatorDatasetML10M(object):
    num_users = 71568
    num_items = 10412
    num_samples = {"train": 478693,
                   "validation": 73590,
                   "test": 73561}

class EstimatorDatasetRTL(object):
    # Purposefully empty as dataset characteristics served to sagemaker via hyperparameters
    pass