from __future__ import division

import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer

import model.estimator.estimator_utils as estimator_utils
import model.estimator.recommender_model as recommender_model
import model.estimator.estimator_dataset as estimator_dataset

from model.evaluation.sequence_level_evaluation import recall_at_k, mrr_at_k
from model.evaluation.user_level_evaluation import recall_at_k as u_recall_at_k, mrr_at_k as u_mrr_at_k

try:
    import itertools.izip as zip
except ImportError:
    pass

from model.utils.backwards_compatibility import merge_dicts

def estimator_model_fn(features, labels, mode, params):

    """
    Parameters
    ----------
    features : list
        Each entry is a dict sent to one of the towers. Keys are {uid, iid, delta_t, seq_lens, user_ids}.
        Vals are tf.float32/tf.int32 tensors with first dimension of size batch_size_for_one_tower.
    labels : list
        Each entry is a tensor sent to one of the towers. The tf.float32 tensor is of the shape
        batch_size_for_one_tower x timesteps.
    mode : tf.estimator.ModeKeys object
        Passed by Estimator - either TRAIN, EVAL or PREDICT
    params : tf.contrib.training.HParams object
        Contains all parameters for the run - extracted from json by init_basic_argument_parser
    Returns
    -------
    tf.estimator.EstimatorSpec
        Object containing the built model
    """

    # Hacky fix for model_fn accepting lists as features whereas serving_input_receiver_fn requires a dict
    # Assumes predictions are served with only one tower
    if type(features) != list:
        features = [features]
        labels = [labels]

    # Flag whether weights are provided as a part of the inputs
    use_weights = "weights" in features[0].keys()

    # tower_features and labels are lists of dicts. Each item in the list goes to one tower,
    # each entry in a dict is a pair in {uid, iid, delta_t, seq_lens, user_ids} and {y} and its batch
    tower_features = features
    tower_labels = labels
    num_gpus = params.num_gpu if mode != tf.estimator.ModeKeys.PREDICT else 0

    # When not 1 GPU then always all results combined on CPU, if 1 GPU then combined on device according to param
    variable_strategy = "CPU" if (params.variable_strategy_CPU or mode == tf.estimator.ModeKeys.PREDICT) else "GPU"

    # Outputs of all towers
    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    # Devices on which towers are built are either CPU if no GPUs or GPU if any available
    if num_gpus == 0:
        num_devices = 1
        device_type = "cpu"
    else:
        num_devices = num_gpus
        device_type = "gpu"

    # Build towers
    for i in range(num_devices):
        worker_device = "/{}:{}".format(device_type, i)

        # Strategy of instantiating variables on appropriate devices
        if variable_strategy == "CPU":
            device_setter = estimator_utils.local_device_setter(
                worker_device=worker_device
            )
        elif variable_strategy == 'GPU':
            device_setter = estimator_utils.local_device_setter(
                ps_device_type='gpu',
                worker_device=worker_device,
                ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    num_gpus, tf.contrib.training.byte_size_load_fn))

        # Reuse variables between towers - only init once on the first tower
        with tf.variable_scope("model", reuse=bool(i != 0)):
            with tf.name_scope("tower_%d" % i) as name_scope:
                with tf.device(device_setter):

                    # No labels available for PREDICT
                    tower_labs_or_None = tower_labels[i] if tower_labels else None

                    # Parameters for regularisation
                    regularization = {"user_reg_weight": params.user_reg_weight,
                                      "user_related_weights": params.user_related_weights}

                    # Dict of outputs - always tower_predictions, gradvars and loss during training
                    tower_outputs = _tower_fn(features=tower_features[i],
                                              labels=tower_labs_or_None,
                                              params=params,
                                              num_towers=num_devices,
                                              variable_strategy=variable_strategy,
                                              regularization=regularization,
                                              mode=mode)

                    if mode == tf.estimator.ModeKeys.TRAIN:
                        tower_gradvars.append(tower_outputs["gradvars"])
                        tower_losses.append(tower_outputs["tower_loss"])
                    if mode == tf.estimator.ModeKeys.EVAL:
                        tower_losses.append(tower_outputs["tower_loss"])
                        tower_preds.append(tower_outputs["tower_predictions"])
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        tower_preds.append(tower_outputs["tower_predictions"])

    # Combine the outputs on the master node
    consolidation_device = "/gpu:0" if variable_strategy == "GPU" else "/cpu:0"
    with tf.device(consolidation_device):

        if mode != tf.estimator.ModeKeys.TRAIN:
            preds = {k:tf.concat([x[k] for x in tower_preds], axis=0) for k in tower_preds[0].keys()}

        # Combine non-feature inputs from all towers
        with tf.name_scope("merge_tower_inputs"):
            stacked_seq_lens = tf.concat([t["seq_lens"] for t in tower_features], axis=0)
            stacked_batch_user_ids = tf.concat([t["uid"][:, 0] for t in tower_features], axis=0)
            stacked_weights = None
            if use_weights:
                stacked_weights = tf.concat([t["weights"] for t in tower_features], axis=0)

        if mode == tf.estimator.ModeKeys.PREDICT:

            # If only interested in the last prediction (e.g. real recommendation)
            if params.last_prediction_only:
                # For each sequence slice the last real timestep
                # preds = {k: tf.gather_nd(v, stacked_seq_lens-1) for k, v in preds.items()}
                batch_size = tf.shape(stacked_seq_lens)[0]
                slices = tf.concat([tf.expand_dims(tf.range(batch_size), 1), tf.expand_dims(stacked_seq_lens, 1)-1], axis=1)
                preds = {k: tf.gather_nd(v, slices) for k, v in preds.items()}

            # If want recommendations to be traceable back to specific users
            if params.prediction_include_uid:
                preds = merge_dicts(preds, {"user_ids": stacked_batch_user_ids})

            return tf.estimator.EstimatorSpec(mode,
                                              predictions=preds,
                                              export_outputs=None) #TODO Specify my own outputs

        # Counts of individual user interactions per tower -
        # used to offset effects of differing sequence lens on things like metrics
        with tf.name_scope("total_interactions_count"):
            # If using weights: sequence mask's 0 and weight non-1 values have to be accounted for
            if use_weights:
                sequence_mask =  tf.sequence_mask(stacked_seq_lens, params.timesteps, dtype=tf.float32,
                                                  name="total_interactions_seq_mask")
                total_interactions_op = tf.reduce_sum(tf.multiply(sequence_mask, stacked_weights),
                                                      name="total_interactions_op_weights")
            else:
                total_interactions_op = tf.reduce_sum(stacked_seq_lens, name="total_interactions_op_no_weights")


        # Combine all labels from all towers
        with tf.name_scope("merge_tower_labels"):
            stacked_labels = tf.concat(labels, axis=0)

        # Calculate total batch loss
        with tf.name_scope("merge_tower_losses"):
            loss = reduce_tower_losses(tower_losses, total_interactions_op)

    if mode == tf.estimator.ModeKeys.TRAIN:

        # Calculate total gradients to apply (scaled by number of interactions in each batch)
        with tf.name_scope('average_gradients'):
            gradvars = average_gradients(tower_gradvars, total_interactions_op)

        with tf.device(consolidation_device):
            # Apply gradients
            with tf.name_scope("apply_gradients"):
                optimizer = LazyAdamOptimizer(params.learning_rate)

                # TODO Check if need params.sync
                train_op = optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())
            metrics = None

    else:
        with tf.device(consolidation_device):
            # Create a dict of metric_name: (metric_var, metric_update_op)
            with tf.name_scope("build_metrics"):
                metrics = build_metrics(labels=stacked_labels,
                                        predictions=preds["top_k"],
                                        seq_lens=stacked_seq_lens,
                                        batch_user_ids=stacked_batch_user_ids,
                                        params=params,
                                        input_top_k=True,
                                        weights=stacked_weights
                                        )
            train_op = None
            # Due to memory constraints loss is not recorded during evaluation, though it needs to be set to a tensor
            if params.zero_loss:
                with tf.name_scope("zero_loss"):
                    loss = tf.constant(0)

    with tf.device(consolidation_device):
        # Count processing speed
        batch_size = params.train_batch_size if mode == tf.estimator.ModeKeys.TRAIN else params.validation_batch_size
        training_hooks = [estimator_utils.ExamplesPerSecondHook(batch_size, every_n_steps=10)]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_chief_hooks=training_hooks
        )


def reduce_tower_losses(tower_losses, total_num_interactions):

    """
    Calculates total average loss based on each tower's loss scaled by the number interactions that took place
    in that tower (may be different due to different seq_lens)

    Parameters
    ----------
    tower_losses : list
        Each entry (scalar) is a sum (not mean!) of all losses for every unmasked timestep in the tower.
    total_num_interactions : tf.int32
        Num of all interactions that took place in the batch
    Returns
    -------
    tf.float32
        Total batch loss
    """
    return tf.divide(tf.reduce_sum(tower_losses), tf.cast(total_num_interactions, tf.float32), name="reduce_tower_losses")


def average_gradients(tower_grads, num_interactions):

    """
    Iterates over gradient and variable pairs, sums the gradients (different seq_lens are accounted for as
    the gradients in each tower are calculated based on the sum of losses rather than mean) and scales the gradients
    by the total number of interactions.

    Parameters
    ----------
    tower_grads : list
        Each entry belongs to one tower. These entries are also lists of tuples of gradient and variable name.
    num_interactions : tf.int32
        Total number of interactions made by users in the batch.
    Returns
    -------
    average_grads : list
        Iterable of variable averaged scaled gradients and names.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        var_type = grad_and_vars[0][0]._tf_api_names[0]
        with tf.device(v.device):

            # IndexedSlices should be just combined into a concatenation of values/indices
            # As then it does not get brutally transformed into a dense array
            if var_type == "IndexedSlices":

                # Unnormalised values
                concat_values = tf.concat([x[0].values for x in grad_and_vars], axis=0)
                # Divide by total num unmasked interactions in batch
                concat_values = tf.divide(concat_values, tf.cast(num_interactions, tf.float32))
                # Indices
                concat_indices = tf.concat([x[0].indices for x in grad_and_vars], axis=0)
                # Combine into one variable
                grad = tf.IndexedSlices(
                    values=concat_values,
                    indices=concat_indices,
                    dense_shape=v.shape
                )
            else:
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_sum(grad, 0, name="reduce_gradvars")

                # Scale gradient by number of interactions within all towers
                grad = tf.divide(grad, tf.cast(num_interactions, tf.float32))

        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def _tower_fn(features, labels, params, num_towers, variable_strategy, regularization, mode):

    """
    Build the model for each tower and return pointers to either logits or also gradvars and losses for train.

    Parameters
    ----------
    features : dict
        Keys are {uid, iid, seq_lens, user_ids, delta_t (optional)}.
        Values are tf.float32 or tf.int32 tensors with first dimension of tower_batch_size.
    labels : tf.int32
        A tensor of shape tower_batch_size x timesteps.
    params : tf.contrib.training.HParam
        Object containing all run and dataset specific parameters.
    num_towers : int
        Number of towers (together with the current one).
    variable_strategy : str
        'CPU' if cell variables should be placed on the CPU or 'GPU' if cell variables should be on the GPU.
    regularization : dict
        Expecting two k:v pairs: 'user_reg_weight' : float determining the magnitude of regularisation (lambda),
        'user_related_weights' : list containing names of matrices on which L2 regularisation is applied upon,
        these matrices should be in trainable variables.
    mode : tf.estimator.ModeKeys object
        Passed by Estimator - either TRAIN, EVAL or PREDICT

    Returns
    -------
    dict
        Either only one key 'tower_predictions' containing logits or additional keys 'gradvars' containing
        a list of gradient and variable name tuples, 'tower_loss' containing tf.float32 value of the summed
        loss over all predictions of the tower's batch.
    """

    # Additional bool that is true if training passed to the model - model is decoupled from estimator
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # Additional bool that is true if current model built for training AND softmax loss is calculated via sampling
    apply_sampling = is_training and params.num_sampled_classes != 0 and params.num_sampled_classes < params.num_items

    #
    if apply_sampling:
        sampling_params = {"sampling": {"num_sampled": params.num_sampled_classes, "labels": labels}}
    else:
        sampling_params = {"sampling": None}

    # Size of the batch assigned to this tower - inferred dynamically
    batch_size = tf.shape(features["uid"])[0]

    # Initialise the model for a given tower - not built yet
    model = recommender_model.TimeDeltaModel(params, is_training, batch_size, variable_strategy)

    # Build the model - behaves differently when training and evaluating (Dropout)
    with tf.name_scope("forward_pass"):
        # User_ids are not actually required but are still passed to make code shorter
        forward_pass_features = merge_dicts(features, sampling_params)

        # batch_size x timesteps x num_items
        forward_pass_outputs = model.forward_pass(**forward_pass_features)

    if not apply_sampling:
        logits = forward_pass_outputs

    if mode != tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("predictions"):

            # If evaluating the model return k (usually 20) predictions to calculate @k metrics
            if mode == tf.estimator.ModeKeys.EVAL:
                k = params.k
            # If actual recommendation requires a different number of items then return that many items
            # If not specified defaults to params.k
            elif mode == tf.estimator.ModeKeys.PREDICT:
                k = params.k_at_prediction if params.k_at_prediction is not None else params.k

            # Get top k recommendations
            _, top_k_indices = tf.nn.top_k(logits, k)
            tower_pred = {
                "top_k": top_k_indices
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return {"tower_predictions": tower_pred}

    if not apply_sampling:
        # Calculate loss
        with tf.name_scope("cross_entropy"):
            # batch_size x timesteps
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                            labels=labels,
                                                                            name="cross_entropy")
        # Tensorflow calculates loss for all steps including the padded ones
        with tf.name_scope("tower_mask"):
            mask = tf.sequence_mask(features["seq_lens"], params.timesteps, dtype=tf.float32, name="tower_mask")
            if "weights" in features.keys():
                mask = tf.multiply(features["weights"], mask, name="tower_weights_mask")
        # Zero out the loss for padded timesteps
        with tf.name_scope("sum_tower_loss"):
            tower_loss = tf.reduce_sum(tf.multiply(cross_entropy, mask), name="tower_loss")

    else:
        tower_loss = forward_pass_outputs

    if regularization["user_reg_weight"] > 0:
        if params.regularize_all:
            user_related_weights = tf.trainable_variables()
        else:
            user_related_weights = [x for x in tf.trainable_variables() if x._shared_name.split("/")[-1] in regularization["user_related_weights"]]
        print("Regularising {} matrices.".format(len(user_related_weights)))
        regularization_loss = tf.reduce_sum(regularization["user_reg_weight"] * tf.stack(
            [tf.nn.l2_loss(x) for x in user_related_weights]
        ), name="unscaled_regularization_loss")
        scaled_regularization_loss = tf.divide(regularization_loss, tf.cast(num_towers, tf.float32), "regularization_loss")

        tf.summary.scalar("scaled_regularization_loss", scaled_regularization_loss)
        tf.summary.scalar("tower_loss_wo_regularization", tower_loss)

        tower_loss += scaled_regularization_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return {"tower_loss": tower_loss,
                "tower_predictions": tower_pred}

    # Calculate the gradients for all the trainable gradients
    with tf.name_scope("calculate_gradients"):
        model_params = tf.trainable_variables()
        tower_grad = tf.gradients(tower_loss, model_params)
        # Pairs of gradient and variable name
        gradvars = zip(tower_grad, model_params)

    return {"tower_loss": tower_loss,
            "gradvars": gradvars}


def input_fn(data_dir, subset, num_shards, batch_size, X_cols_to_use, input_data_format="npy", shuffle=None,
             additional_arrays=[], delta_t_mean=None, delta_t_std=None):

    """

    Estimator function that reads the data from disk, splits it into sub-batches in case
    of multiple towers and returns pointers to the final lists containing each tower's share
    of data. Assumes that data_dict contains three directories, normally 'train', 'evaluation'
    and 'test'. Subset should also share its value with one of these three directories' names.
    Each of those directories is expected to contain 4 files: X.npy (with usually columns:
    uid, iid, timestep - that one is optional), y.npy containing 1 column (labels), seq_lens.npy
    containing the lengths of unpadded training sequences (values between 1 and 20) and
    user_ids.npy containing the id of the user to whom each sequence belongs to.
    Additionally assumes that the batch can be split equally between towers with no tower
    having more data than the others.

    Parameters
    ----------
    data_dir : str or Path
        Location of absolute path where data set's subset directories are located.
    subset : str {'train', 'validation', 'test'}
        Subset of the data to use from the data set.
    num_shards : int
        Number of towers to use, minimum 1.
    batch_size : int
        Number of sequences in a batch.
    X_cols_to_use : dict
        k:v pairs where k is the index of the column and v is the name of the feature that is
        assigned to the column.
    shuffle : int
        If not None the tf.data.Dataset object also has .shuffle(shuffle) applied to it - the param
        determines how many first sequences are taken from the dataset from which we randomly
        select however many we need. See documentation for tf.data.Dataset for more info.
    input_data_format : {"npy", "tfrecords"}
        Type of data to process - datasets currently saved as numpy arrays or tfrecords.
    additional_arrays : iterable of str
        Names of additional arrays to be used in training/validation. Currently only supporting "weights".
    delta_t_mean : float or None
        If provided the `delta_t` feature of the data (if used) will have `delta_t_mean` subtracted from every entry.
    delta_t_std : float or None
        If provided the `delta_t` feature of the data (if used) will be divided by `delta_t_mean` in every entry.
    Returns
    -------
    features : list
        List of length num_shards where each entry is a dict of tensors with first dimension of batch_size/num_shards.
        The keys for those tensors are {'uid', 'iid', 'delta_t' - optional, 'seq_lens', 'user_ids'}.
    labels : list
        List of length num_shards where each entry is a a tensor of shape batch_size/num_shards.
    """

    with tf.name_scope("input_fn"):

        # If using any additional supported arrays such as weights add them to the model
        files_to_use = ["X", "y", "seq_lens"] + additional_arrays

        # Instantiate DataSet so it points to the location of the data and knows
        # which columns to return
        dataset = estimator_dataset.RecommendationDataSet(data_dir=data_dir,
                                                          subset=subset,
                                                          X_cols_to_use=X_cols_to_use,
                                                          input_data_format=input_data_format,
                                                          file_types=files_to_use,
                                                          shuffle_buffer_size=shuffle,
                                                          delta_t_mean=delta_t_mean,
                                                          delta_t_std=delta_t_std)

        # Return a full batch in the form of a dict with the following keys:
        # features {'uid', 'iid', 'delta_t' - optional, 'seq_lens', 'user_ids'} and tensor of labels ('y').
        # Each of those keys is associated with a tensor with its first dimension of size batch_size.
        with tf.name_scope("make_batch"):
            features, labels = dataset.make_batch(batch_size)

        with tf.name_scope("return_batch"):

            # Ensure num_shards is a positive integer
            assert (num_shards > 0 and isinstance(num_shards, int))

            # Model_fn still assumes that there are N towers - thus always placing data inside an iterable.
            if num_shards == 1:
                return [features], [labels]
                # When there are more than 1 tower split data into sub-batches in a circle round fashion
            with tf.name_scope("split_batch_into_towers"):
                assert batch_size % num_shards == 0
                feature_shards = {k: tf.split(v, num_shards) for k,v in features.items()}
                feature_shards = [dict(zip(feature_shards.keys(), x)) for x in list(zip(*feature_shards.values()))]
                label_shards = tf.split(labels, num_shards)
                return feature_shards, label_shards


def create_estimator(run_config, hparams, warm_start_from=None):

    """
    Instantiate a new tf.estimator.Estimator object with its model_fn being the model_fn defined in this module.
    The run_config and hparams are also passeed to the new Estimator.

    Parameters
    ----------
    run_config : tf.estimator.RunConfig
        Object specifying runtime parameters e.g. model and check pointing directory, check pointing frequency.
        See documentation for tf.estimator.RunConfig for more information.
    hparams : tf.contrib.training.HParams
        Object containing all specifics about the model e.g. num_hidden and batch_size.
    warm_start_from : str
        If specified, checkpoint path from which the model is restarted.
    Returns
    -------
    estimator : tf.estimator.Estimator
        Object specifying how the model is built and using which parameters.
    """

    estimator = tf.estimator.Estimator(model_fn=estimator_model_fn,
                                       params=hparams,
                                       config=run_config,
                                       warm_start_from=warm_start_from
                                       )
    return estimator


def build_metrics(labels, predictions, seq_lens, batch_user_ids, params, input_top_k=False, weights=None):

    """

    Build recall, mrr, user based recall and user based mrr. Return them in the form of a dict as expected
    by Estimator. Note that all tensors used as input are from the whole batch - not split into towers.

    Parameters
    ----------
    labels : tf.int32 tensor
        Tensor of shape batch_size x timesteps containing the indices of the true items.
    logits : tf.float32 tensor
        Tensor of shape batch_size x timesteps x num_items containing the probability-like scores for each
        of the items at each timestep.
    seq_lens : tf.int32 tensor
        Tensor of shape batch_size containing the lengths of unmasked sequences.
    batch_user_ids : tf.int32 tensor
        Tensor of shape batch_size containing the indices user to whom any given sequence belongs to - used to
        calculate user based metrics.
    params : tf.contrib.training.HParams
         Object containing all specifics about the model e.g. num_hidden and batch_size.
    Returns
    -------
    metrics : dict
        Keys are the names of the metrics and values are tuples of the metric variable and its update.
    """

    # Parameter k:v pairs passed both to sequence based and user based metrics
    metrics_params = {"labels": labels,
                      "predictions": predictions,
                      "seq_lens": seq_lens,
                      "timesteps": params.timesteps,
                      "k": params.k,
                      "input_top_k": input_top_k,
                      "weights": weights}

    # Parameter k:v pairs passed only to user based metrics
    u_metrics_params = {"user_ids": batch_user_ids,
                        "num_users": params.num_users}

    # Instantiate sequence based metrics - returns the metrics variable and metrics op
    recall, recall_op = recall_at_k(**merge_dicts(metrics_params, {"name": "recall"}))
    mrr, mrr_op = mrr_at_k(**merge_dicts(metrics_params, {"name": "mrr"}))

    # Instantiate user based metrics - returns the metrics variable, unfiltered metrics array and a tuple
    # of metrics and metrics array update ops. For explanation of unfiltered metrics array check the documentation
    # of user based metrics functions.
    u_recall, uf_recall_arr, u_recall_updates = u_recall_at_k(**merge_dicts(metrics_params,
                                                                            u_metrics_params,
                                                                            {"name":"u_recall"}))
    u_mrr, uf_mrr_arr, u_mrr_updates = u_mrr_at_k(**merge_dicts(metrics_params, u_metrics_params, {"name": "u_mrr"}))

    # Return only the updates associated with the metrics, not the arrays.
    u_recall_update = u_recall_updates[0]
    u_mrr_update = u_mrr_updates[0]

    metrics = {"recall": (recall, recall_op),
               "mrr": (mrr, mrr_op),
               "u_recall": (u_recall, u_recall_update),
               "u_mrr": (u_mrr, u_mrr_update)}

    return metrics

def serving_input_fn(hyperparameters):

    executing_on_sagemaker = type(hyperparameters) == dict

    if executing_on_sagemaker:
        hyperparameters = tf.contrib.training.HParams(**hyperparameters)

    use_dt = len(hyperparameters.X_cols_to_use) == 3
    scale_dt = hyperparameters.delta_t_mean is not None or hyperparameters.delta_t_std is not None

    def dt_serving_input_fn():

        receiver_tensors = {"X": tf.placeholder(tf.int32, [None, None, 3]),
                            "seq_lens": tf.placeholder(tf.int32, [None])}

        slice_X = lambda col: tf.squeeze(tf.slice(receiver_tensors["X"], [0,0,col], (-1, -1, 1)), [2])

        features = {"uid": slice_X(0),
                    "iid": slice_X(1),
                    "seq_lens": receiver_tensors["seq_lens"]}

        if use_dt:
            features["delta_t"] = slice_X(2)

            # If `delta_t` was used during training and was rescaled apply the same rescaling during prediction.
            if scale_dt:
                features["delta_t"] = tf.cast(features["delta_t"], tf.float64)
                if hyperparameters.delta_t_mean is not None:
                    features["delta_t"] -= hyperparameters.delta_t_mean
                if hyperparameters.delta_t_std is not None:
                    features["delta_t"] /= hyperparameters.delta_t_std

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    # Hacky fix - tensorflow wants the method object while sagemaker wants the output of the method
    if executing_on_sagemaker:
        return dt_serving_input_fn()
    return dt_serving_input_fn