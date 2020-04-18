import pandas as pd
import numpy as np
from time import time
import os
import sys


def determine_split_lens(seq_len: int, split=(0.9, 0.05, 0.05)):

    """

    Calculates the numbers of interactions as well as at which index each subset would start and end.
    The lengths are calculated on the assumption that there is one item overhang to account for
    subset 2's first item being used in the subset 1 but as a label. Akin to this:
    ------
         ---
           ---
    Note that both validation and test have to round up to a full item otherwise the remaining items go to train.
    Additionally as there are actually 19 sets of features (last on is only used for label) a subset percentage times
    seq_len - 1 has to equal a full item for the item to be added to a subset.
    For example using baseline split: if there are 20 items then all would go to train since (20-1)*0.05 < 1 for
    validation and test. If there are 21 items then 19 items would go to train (18 of them will be used as features),
    2 would go to validation (1 feature, 1 label) and 2 would go to test.

    Parameters
    ----------
    seq_len : int
        Number of total interactions by user in the whole dataset
    split : iterable
        (train, validation, test) percentage of total sequence that will go to each of the subsets

    Returns
    -------
    tuple
        Tuple of tuples, where first tuple contains values of lengths of three subsets. The second tuple
        also contains three tuples where each tuple belongs to a subset and contains two values: the start
        and the end of the subset in that user's history.
    """

    tr_pc, val_pc, ts_pc = split

    # Remove one item for label as data will have to be shifted, so have to have at least 21 items (not 20)
    # to have items go to test/val assuming 0.05 for either
    X_len = seq_len - 1

    # Lengths of outputs for a given user
    # Ts/val counts only incremented when a full item can be added
    ts_len = int(np.floor(X_len * ts_pc))
    val_len = int(np.floor(X_len * val_pc))
    # +1 as previously subtracted 1 - making sure tr_len even if others are 0 adds back up to seq_len
    tr_len = X_len - val_len - ts_len + 1

    # Add one more item to account for label if any items are at all added to the output df
    ts_len += 1 if ts_len > 0 else 0
    val_len += 1 if val_len > 0 else 0

    tr_start = 0
    tr_end = tr_start + tr_len
    # Val starts with the last item from train since that last train item is used as a label
    val_start = tr_end - 1
    val_end = val_start + val_len
    # As with val but if len(val)==0 then shift the pointer back up not to do -1 twice when not needed
    ts_start = val_end - 1 if val_len > 0 else val_end
    ts_end = ts_start + ts_len

    # Make sure that there is 1 item overlap between subsets unless subset length is 0
    assert tr_len + val_len + ts_len - 1 - (val_len > 0) - (ts_len > 0) == X_len

    return (tr_len, val_len, ts_len), ((tr_start, tr_end), (val_start, val_end), (ts_start, ts_end))


def train_val_test_split_train_overlapping(df, col_names=["uid", "iid", "time"], split=[0.9, 0.05, 0.05]):

    """

    Generates three DataFrames for train, val, test. The last item from train is also
    the first item in val as the item is used as a label in train but a set of
    features in val. The split is also performed in a way that items are added only to
    train until val_pc (or test_pc) would result a new full item - so in case of 20 items
    minus one label (so 19) all items go to train. But if we have 21-1 items for features
    they are split 18-1-1 (resulting df's are 19-2-2 due to a final label)

    Parameters
    ----------
    df : pd.DataFrame
        Original input, assumes only np.int32 values, 3 columns and user index being named 'uid'
    col_names : list
        Names of the columns given to the output. Note that order of the columns in maintained the same.
    split : list
        (train, validation, test) percentage of total sequence that will go to each of the subsets
    Returns
    -------
    train_df : pd.DataFrame
        Concatenated sequences belonging to train subset from all users with column names col_names
    val_df : pd.DataFrame
        Concatenated sequences belonging to validation subset from all users with column names col_names
    test_df : pd.DataFrame
        Concatenated sequences belonging to test subset from all users with column names col_names

    """

    # Ensure the total amount of data used is 100%
    assert np.sum(split) == 1

    train_lengths = {}
    val_lengths = {}
    test_lengths = {}

    train_idx = {}
    val_idx = {}
    test_idx = {}

    # Split df by user id
    grouped = df.groupby("uid")
    # iterate over users and determine lengths for each user (more efficient to create empty array to fill than
    # a length 0 array to append to)
    for uid, items in grouped:
        # Calculate the length and the start and end indices for each split for each user
        lens, idx = determine_split_lens(len(items), split=split)
        train_lengths[uid], val_lengths[uid], test_lengths[uid] = lens
        train_idx[uid], val_idx[uid], test_idx[uid] = idx

    # Preinitialise appropriately sized but empty Df's
    train_size = int(sum(train_lengths.values()))
    val_size = int(sum(val_lengths.values()))
    test_size = int(sum(test_lengths.values()))

    # Assumes there are 3 features
    train_np = np.empty((train_size, 3), dtype=np.int32)
    val_np = np.empty((val_size, 3), dtype=np.int32)
    test_np = np.empty((test_size, 3), dtype=np.int32)

    train_seen = 0
    val_seen = 0
    test_seen = 0

    # Iterate over interactions of each user, make previously calculated slices and append to train, validation
    # and test arrays as appropriate.
    for uid, items in grouped:

        train_start, train_end = train_idx[uid]
        val_start, val_end = val_idx[uid]
        test_start, test_end = test_idx[uid]

        train_len = train_lengths[uid]
        val_len = val_lengths[uid]
        test_len = test_lengths[uid]

        u_train_slice = items.iloc[train_start:train_end]
        u_val_slice = items.iloc[val_start:val_end]
        u_test_slice = items.iloc[test_start:test_end]

        train_np[train_seen:train_seen+train_len] = u_train_slice
        val_np[val_seen:val_seen+val_len] = u_val_slice
        test_np[test_seen:test_seen+test_len] = u_test_slice

        train_seen += train_len
        val_seen += val_len
        test_seen += test_len

    train_df = pd.DataFrame(train_np, columns=col_names)
    val_df = pd.DataFrame(val_np, columns=col_names)
    test_df = pd.DataFrame(test_np, columns=col_names)

    return train_df, val_df, test_df

def calculate_num_seqs_for_user_big_hop(num_interactions, hop_size, seq_len):

    """
    Calculate the number of sequences that each user will have. It can be imagined
    that this predicts how many different window positions it will be possible to have
    while moving the window by 'hop_size' steps and the window size is seq_len.
    If the last window reaches outside the sequence the shorter subsequence is still
    counted as another sequence.

    Parameters
    ----------
    num_interactions : int
        Total number of interactions by a user.
    hop_size : int
        Number of steps by which the sliding window is moved each time.
    seq_len : int
        Number of interactions per sequence including 1 extra interaction for label. So to produce
        sequence of 20 features and 20 labels there is a need of seq_len==21.
    Returns
    -------
    int
        Number of individual sequences that should be reserved for a given user.
    """

    if seq_len < hop_size:
        print("SEQ_LEN < HOP_SIZE CURRENTLY UNSUPPORTED")
        sys.exit()

    # Length of the last sequence
    tail = num_interactions % hop_size

    # Number of total sequences (float)
    num_seqs = num_interactions / hop_size

    # Round down if last seq will not have 1 interaction for features and one for the label otherwise round up
    num_seqs = np.floor(num_seqs) if tail < 2 else np.ceil(num_seqs)

    # Needed as hs:1, ni:5 should result in ns:4 not 5
    if hop_size == 1:
        num_seqs -= 1

    return int(num_seqs)

def pad_subseq(seq, predictions_per_seq):

    """
    Adds all zeroes as last interactions if the sequence is shorter than 'predictions_per_seq' and returns
    the padded sequence. Supports padding of both X values which are 2 dimensional (time x features) and 1 dimensional
    labels (time).

    Parameters
    ----------
    seq : np.array
        Either 1 or 2 dimemsional array where the first dimension is time.
    predictions_per_seq : int
        The number of interactions that the output will have after padding is applied.

    Returns
    -------
    np.array
        np.int32 array of the same rank extended up to length 'predictions_per_seq' along the first dimension with
        0 values.
    """

    dim = len(seq.shape)
    padding_size = predictions_per_seq - len(seq)

    if dim == 2:
        return np.pad(seq, ((0, padding_size), (0, 0)), "constant", constant_values=0)
    elif dim == 1:
        return np.pad(seq, ((0, padding_size)), "constant", constant_values=0)
    else:
        sys.exit()

def generate_big_hop_numpy_files(df,
                                 X_file_path=None,
                                 Y_file_path=None,
                                 lens_file_path=None,
                                 seq_len=21,
                                 hop_size=20,
                                 features=["uid", "iid"],
                                 save=True):

    """
    Given a pd.DataFrame belonging to a subset (train, validation, test) converts it into a new
    pd.DataFrame where each entry along the first dimension is a sequence of subsequent
    seq_len MINUS ONE interactions performed by the same user. Each time the sequence is
    completed the sliding window is moved into the future by hop_size steps. The interactions
    are thus assumed to be sorted in time with earlier actions happening first. When save is True
    all file paths have to be provided in order to save numpy arrays at those locations. If False no
    saving takes place and arrays are only returned.
    If a user does not have a sufficient number of interactions for their last batch the sequence
    is still padded up to length seq_len-1 in order to produce a valid matrix. The lengths of unpadded
    sequences which are important for masking (metrics and loss calculation) are documented in seq_lens.npy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of all interactions of all users sorted by ascending time. Usually DataFtame of
        subset of interactions e.g. only all train interactions.
    X_file_path : str or Path
        Location where to write X.npy (uid, iid, optional - time).
    Y_file_path : str or Path
        Location where to write y.npy (iid).
    lens_file_path : str or Path
        Location where to write seq_lens.npy
    seq_len : int
        Number of interactions per sequence including 1 extra interaction for label. So to produce
        sequence of 20 features and 20 labels there is a need of seq_len==21.
    hop_size : int
        Number of steps by which the sliding window is moved each time.
    features : list
        Columns to use. Assumes that at least 2 are called "uid" and "iid"
    save : bool
        If True X_file_path, Y_file_path and lens_file_path have to be specied and numpy arrays
        are saved in the specified locations.

    Returns
    -------
    epoch_X : np.int32
        Numpy array generally with values taken from input's columns {"uid", "iid", "delta_t"}
        or whatever specified by 'features'. The second dimension is of size seq_len - 1.
    epoch_Y : np.int32
        Numpy array with second dimension of size seq_len - 1 containing item labels for each
        item that is going to happen at each timestep.
    epoch_lens : np.int32
        Numpy array with containing the number of interactions that happened in each sequence
        (as opposed to the total padded length of each sequence that is the same in each
        sample in the epoch).

    """


    # Number of predictions made for 1 subsequence
    predictions_per_seq = seq_len - 1

    grouped_per_user = df.groupby("uid")

    num_interactions_per_user = grouped_per_user.size()

    # Calculate number of subsequences for each user
    num_subseqs_per_user = {u: calculate_num_seqs_for_user_big_hop(count,
                                                                   hop_size,
                                                                   seq_len)
                            for u, count in num_interactions_per_user.items()}

    # Number of subseqs in epoch
    total_seqs = int(sum(num_subseqs_per_user.values()))

    # Empty placeholders
    epoch_X = np.empty((total_seqs, predictions_per_seq, len(features)), dtype=np.int32)
    epoch_Y = np.empty((total_seqs, predictions_per_seq), dtype=np.int32)

    # Unpadded lenghts
    epoch_lens = np.empty((total_seqs), dtype=np.int32)

    # Current row in epoch
    pos_in_epoch = 0

    # Iterate over users
    for uid, num_interactions in num_subseqs_per_user.items():

        if uid % 100 == 0:
            print("Processing user number: {}".format(uid))

        # Extract user sequence
        user_seq = grouped_per_user.get_group(uid)

        # Iterate over the number of subseqs by the user
        for uid_iter in np.arange(0, num_interactions):

            # Find the starting index for the subseq
            user_start = uid_iter * hop_size
            user_end = user_start + seq_len
            if len(user_seq) < user_end:
                user_end = len(user_seq)
            seq = user_seq[user_start:user_end]

            seq_X = seq[:-1][features].values
            seq_Y = seq[1:]["iid"].values

            if len(seq_X) == 0 or len(seq_Y) == 0:
                print('ok')

            epoch_lens[pos_in_epoch] = predictions_per_seq

            # Unpadded subsequence length
            actual_num_predictions = len(seq_X)

            # If sequence needs padding
            if actual_num_predictions != predictions_per_seq:

                # Amend unpadded sequence length
                epoch_lens[pos_in_epoch] = len(seq_X)
                # Apply padding with the last element in the sequence
                seq_X = pad_subseq(seq_X, predictions_per_seq)
                seq_Y = pad_subseq(seq_Y, predictions_per_seq)

            # Appropriately shaped numpy array
            seq_X = seq_X.reshape((1, predictions_per_seq, len(features)))
            seq_Y = seq_Y.reshape((1, predictions_per_seq))

            epoch_X[pos_in_epoch] = seq_X
            epoch_Y[pos_in_epoch] = seq_Y

            pos_in_epoch += 1
    if save:
        assert (X_file_path is not None and Y_file_path is not None and lens_file_path is not None)
        np.save(X_file_path, epoch_X, allow_pickle=False)
        np.save(Y_file_path, epoch_Y, allow_pickle=False)
        np.save(lens_file_path, epoch_lens, allow_pickle=False)
    return epoch_X, epoch_Y, epoch_lens

def big_hop_generator(df,
                      path,
                      train_or_test,
                      dataset_name,
                      seq_len=21,
                      batch_size=1000,
                      hop_size = 20,
                      features=["uid", "iid"],
                      min_seq_len=None,
                      num_samples=None,
                      randomize=False):
    """
    Generator with ability to save numpy arrays for x, y, lens
    Designed for hop size 20 and seq_len 21 (20 features + 20 predictions)
    Outputs bs x timesteps (padded) x 2 for both X and Y


    Parameters
    ----------
    df
    path
    train_or_test
    dataset_name
    seq_len
    batch_size
    hop_size
    features
    min_seq_len
    num_samples
    randomize

    Returns
    -------

    """

    assert seq_len >= 2

    if train_or_test == "Mid_Epoch_Test":
        train_or_test = "Test"

    X_file_name = generate_numpy_filename(dataset_name, train_or_test, df, seq_len, hop_size, "X")
    X_file_path = path + X_file_name

    Y_file_name = generate_numpy_filename(dataset_name, train_or_test, df, seq_len, hop_size, "Y")
    Y_file_path = path + Y_file_name

    lens_file_name = generate_numpy_filename(dataset_name, train_or_test, df, seq_len, hop_size, "lens")
    lens_file_path = path + lens_file_name

    files_exist = os.path.isfile(X_file_path) and \
                  os.path.isfile(Y_file_path) and \
                  os.path.isfile(lens_file_path)

    t = time()
    if files_exist:
        epoch_X = np.load(X_file_path, allow_pickle=False)
        epoch_Y = np.load(Y_file_path, allow_pickle=False)
        epoch_lens = np.load(lens_file_path, allow_pickle=False)
        print("{} arrays loaded. Elapsed {} seconds".format(train_or_test, round(time() - t, 2)))

    else:

        epoch_X, epoch_Y, epoch_lens = generate_big_hop_numpy_files(df=df,
                                                                    X_file_path=X_file_path,
                                                                    Y_file_path=Y_file_path,
                                                                    lens_file_path=lens_file_path,
                                                                    seq_len=seq_len,
                                                                    hop_size=hop_size,
                                                                    features=features)
        print("{} arrays generated and saved. Elapsed {} seconds".format(train_or_test, round(time() - t, 2)))

    num_seqs = len(epoch_X)
    seq_order = np.fromiter(range(num_seqs), dtype=np.int32)

    if randomize:
        np.random.shuffle(seq_order)

    no_batches = int(np.ceil(num_seqs / batch_size))
    batch_start = 0
    for batch_id in range(no_batches):
        # with timeit_context("Generation"):
        batch_end = batch_start + batch_size
        if batch_id == no_batches - 1:
            batch_end = num_seqs
        seq_ids = seq_order[batch_start:batch_end]
        batch_X = epoch_X[seq_ids]
        batch_Y = epoch_Y[seq_ids]
        batch_lens = epoch_lens[seq_ids]
        batch_start += batch_size
        user_ids = batch_X[:,0,0]
        yield batch_id, batch_X, batch_Y, batch_lens, user_ids

def generate_numpy_filename(dataset_name, features, train_or_test, df, seq_len, hop_size, type):
    return "/" + dataset_name + "_" + \
           "_".join(features) + "_" + \
           str(len(df)) + "_" + \
           train_or_test + "_" + \
           str(seq_len) + "_" + \
           str(hop_size) + "_" +\
           type + ".npy"

if __name__ == "__main__":
    input_filename = "/Users/nknyazev/Documents/Delft/Thesis/temporal/data/processed/lastfm_10_pc_per_user_incr.csv"

    df = pd.read_csv(input_filename, header=None, names=["uid", "iid", "time"])
    tr, val, ts = train_val_test_split_train_overlapping(df)
    g = big_hop_generator(df=ts,
                          path="/Users/nknyazev/Documents/Delft/Thesis/temporal/data/numpy",
                          train_or_test="Test",
                          dataset_name="lastfm_10_pc",
                          randomize=False
                          )
    for a,b,c in g:
        print("ok")
        break
    