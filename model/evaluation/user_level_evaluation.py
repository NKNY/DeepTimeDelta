from __future__ import division
import tensorflow as tf
import numpy as np
import sys

def batch_top_indices(predictions, k):

    """
    Return k top indices for each sequence in the batch for every timestep (highest logits)

    Parameters
    ----------
    predictions
    k

    Returns
    -------

    """

    return np.flip(predictions.argsort()[:,:,-k:],2)


class SelectedTimestepsEvaluator(object):

    def __init__(self, timesteps_to_include, num_users):

        """
        Class to perform user level evaluation ONLY for timesteps indicated upon initialisation. These timesteps
        should be given as a numpy array or other iterable that can be converted to a list. These should be sorted in ascending order.
        The interactions upon which the evaluation metrics are calculated should also come in a sorted order.

        Parameters
        ----------
        timesteps_to_include : iterable
            Iterable that can be converted into a flat list containing indices of interactions for which the evaluation
            metrics are calculated.
        """

        self._timesteps_to_include = list(timesteps_to_include)
        self._num_denom = {}
        self._current_timestep = {}
        self._included_timesteps_seen = {}
        self._num_users = num_users
        self._found_all_timesteps = {}

    def update_num_denom(self, metric, num, denom):
        if metric not in self._num_denom:
            self._num_denom[metric] = (np.zeros([self._num_users]), np.zeros([self._num_users]))
        self._num_denom[metric] = (self._num_denom[metric][0]+num, self._num_denom[metric][1]+denom)

    def update_recall(self, top_indices, labels, seq_lens, user_ids, weights=None):

        """
        Initialises the metric if previously uninitialised. Iterates over all timesteps in the batch
        and increments the numerator and denominator if one of the steps specified during init is
        encountered - as the numerator and denominator are stored as a tuple of numpy arrays of len
        num_users, the increment happens only for the users to whom the sequence belongs.
        Weights is an optional argument which multiplies the numerator and the denominator of the interaction
        by the weight of that particular timestep.

        Parameters
        ----------
        top_indices
        labels
        seq_lens
        timesteps
        user_ids

        Returns
        -------

        """

        if "u_recall" not in self._current_timestep:
            self._current_timestep["u_recall"] = 0
            self._included_timesteps_seen["u_recall"] = 0
            self._found_all_timesteps["u_recall"] = False

        batch_num = np.zeros([self._num_users])
        batch_denom = np.zeros([self._num_users])

        for seq in range(len(top_indices)):
            user_id = user_ids[seq]
            for t in range(seq_lens[seq]):
                if self._found_all_timesteps["u_recall"]:
                    break
                if self._current_timestep["u_recall"] == self._timesteps_to_include[self._included_timesteps_seen["u_recall"]]:
                    self._included_timesteps_seen["u_recall"] += 1
                    if labels[seq][t] in top_indices[seq][t]:
                        numerator_update = weights[seq, t] if weights is not None else 1
                        batch_num[user_id] += numerator_update
                    denominator_update = weights[seq, t] if weights is not None else 1
                    batch_denom[user_id] += denominator_update
                self._current_timestep["u_recall"] += 1
                # If all timesteps have been found break the evaluation
                if len(self._timesteps_to_include) == self._included_timesteps_seen["u_recall"]:
                    self._found_all_timesteps["u_recall"] = True
            if self._found_all_timesteps["u_recall"]:
                break


        self.update_num_denom("u_recall", batch_num, batch_denom)


    def update_mrr(self, top_indices, labels, seq_lens, user_ids, weights=None):

        """
        Initialises the metric if previously uninitialised. Iterates over all timesteps in the batch
        and increments the numerator and denominator if one of the steps specified during init is
        encountered - as the numerator and denominator are stored as a tuple of numpy arrays of len
        num_users, the increment happens only for the users to whom the sequence belongs.
        Weights is an optional argument which multiplies the numerator and the denominator of the interaction
        by the weight of that particular timestep.

        Parameters
        ----------
        top_indices
        labels
        seq_lens
        timesteps
        user_ids

        Returns
        -------

        """

        if "u_mrr" not in self._current_timestep:
            self._current_timestep["u_mrr"] = 0
            self._included_timesteps_seen["u_mrr"] = 0
            self._found_all_timesteps["u_mrr"] = False

        batch_num = np.zeros([self._num_users])
        batch_denom = np.zeros([self._num_users])

        for seq in range(len(top_indices)):
            user_id = user_ids[seq]
            for t in range(seq_lens[seq]):
                if self._found_all_timesteps["u_mrr"]:
                    break
                if self._current_timestep["u_mrr"] == self._timesteps_to_include[self._included_timesteps_seen["u_mrr"]]:
                    self._included_timesteps_seen["u_mrr"] += 1
                    if labels[seq][t] in top_indices[seq][t]:
                        numerator_update = 1. / (1. + np.where(top_indices[seq][t] == labels[seq][t])[0][0])
                        numerator_update = numerator_update * weights[seq, t] if weights is not None else numerator_update
                        batch_num[user_id] += numerator_update
                    denominator_update = weights[seq, t] if weights is not None else 1
                    batch_denom[user_id] += denominator_update
                self._current_timestep["u_mrr"] += 1
                if len(self._timesteps_to_include) == self._included_timesteps_seen["u_mrr"]:
                    self._found_all_timesteps["u_mrr"] = True

            if self._found_all_timesteps["u_mrr"]:
                break

        self.update_num_denom("u_mrr", batch_num, batch_denom)

    def calculate_metric(self, metric):
        try:
            non_zero_users = self._num_denom[metric][1] != 0
            return np.mean(np.divide(self._num_denom[metric][0][non_zero_users], self._num_denom[metric][1][non_zero_users]))
        except KeyError as e:
            print("Metric {} not supported OR not yet initialised. Initialisation is via self.update_mrr".format(metric))
            sys.exit(1)
        except ZeroDivisionError as e2:
            print("Denominator is zero. Division by zero not possible.")
            sys.exit(1)





def calculate_batch_recall_at_k(top_indices, labels, seq_lens, user_ids, num_users, timesteps, num_denom=None):

    """
    If current_num, current_denom not specified then returns the batch's numerator and denominator for recall@k metric
    for every user separately. Missing users are not filtered out.
    If the above are specified then the counts are increased by the current batch values.
    Parameters
    ----------
    top_indices
    labels
    seq_lens
    user_ids
    num_users
    timesteps
    num_denom

    Returns
    -------

    """



    batch_nums = np.zeros([len(top_indices), timesteps])

    if num_denom is None:
        num = np.array([0.]*num_users)
        denom = np.array([0.]*num_users)
    else:
        num, denom = num_denom

    for seq in range(len(top_indices)):
        for t in range(seq_lens[seq]):
            if labels[seq][t] in top_indices[seq][t]:
                batch_nums[seq][t] = 1

    for seq in range(len(top_indices)):
        user = user_ids[seq]
        num[user] += np.sum(batch_nums[seq])
        denom[user] += seq_lens[seq]

    return num, denom

def calculate_batch_mrr_at_k(top_indices, labels, seq_lens, user_ids, num_users, timesteps, num_denom=None):

    """
    If current_num, current_denom not specified then returns the batch's numerator and denominator for mrr@k metric
    for every user separately. Missing users are not filtered out.
    If the above are specified then the counts are increased by the current batch values.
    Parameters
    ----------
    top_indices
    labels
    seq_lens
    user_ids
    num_users
    timesteps
    num_denom

    Returns
    -------

    """

    batch_nums = np.zeros([len(top_indices), timesteps])

    if num_denom is None:
        num = np.array([0.]*num_users)
        denom = np.array([0.]*num_users)
    else:
        num, denom = num_denom

    for seq in range(len(top_indices)):
        for t in range(seq_lens[seq]):
            if labels[seq][t] in top_indices[seq][t]:
                batch_nums[seq][t] = 1./(1.+np.where(top_indices[seq][t]==labels[seq][t])[0][0])

    for seq in range(len(top_indices)):
        user = user_ids[seq]
        num[user] += np.sum(batch_nums[seq])
        denom[user] += seq_lens[seq]

    return num, denom

# Helper methods
def _at_k_name(name, k=None):
  if k is not None:
    name = '%s_at_%d' % (name, k)
  else:
    name = '%s_at_k' % (name)
  return name

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

  return tf.Variable(
      lambda: tf.zeros(shape, dtype),
      trainable=False,
      collections=[
          tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      name=name)

# Recall internal ops
def _compare_equal(pred, lab):
    return tf.equal(pred, lab)

def _lab_in_preds(preds, lab):
    return tf.reduce_any(_compare_equal(preds, lab))

def _lab_in_preds_per_timestep(preds, lab):
    return tf.map_fn(lambda x: _lab_in_preds(x[0], x[1]), (preds, lab), dtype=tf.bool)

def _find_correct_bool_per_timestep(predictions_idx, labels):
    return tf.map_fn(lambda x: _lab_in_preds_per_timestep(x[0], x[1]),
                     [predictions_idx, labels], dtype=tf.bool)

def _find_count_correct_per_seq(predictions_idx, labels, mask):
    correct_per_timestep = tf.cast(_find_correct_bool_per_timestep(predictions_idx, labels), tf.float64)
    masked = tf.multiply(correct_per_timestep, mask)
    return tf.reduce_sum(masked, axis=1)

def _streaming_tp_at_k(predictions_idx, labels, mask, k, user_ids, num_users, name=None):

    with tf.name_scope(name, _at_k_name('true_positives', k),
                       (predictions_idx, labels)) as scope:
        target_in_top_k = tf.reduce_any(tf.equal(tf.tile(tf.expand_dims(labels, 2), [1, 1, k]), predictions_idx), 2)

        masked = tf.multiply(tf.cast(target_in_top_k, tf.float64), mask)

        count_correct_per_seq = tf.reduce_sum(masked, axis=1)

        var = metric_variable([num_users], tf.float64, name=scope)
        return var, tf.scatter_add(var, user_ids, count_correct_per_seq)


def _streaming_num_predictions(mask, user_ids, num_users, k, name=None) :

    with tf.name_scope(name, _at_k_name("total_prediction_count", k)) as scope:

        var = metric_variable([num_users], tf.float64, name=scope)

        return var, tf.scatter_add(var, user_ids, tf.cast(tf.reduce_sum(mask, axis=1), tf.float64))

def remove_empty_users(tp, total):
    where_positive = tf.greater(total, 0.99)
    return tf.boolean_mask(tp, where_positive), tf.boolean_mask(total, where_positive)



def recall_at_k(labels, predictions, seq_lens, timesteps, k, user_ids, num_users, name,
                input_top_k=False,
                metrics_collections=None,
                updates_collections=None,
                weights=None):

    with tf.name_scope(name, _at_k_name("recall", k),
                       (predictions, labels)) as scope:

        if not input_top_k:
            _, predictions_idx = tf.nn.top_k(predictions, k)
        else:
            predictions_idx = predictions

        top_k_idx = tf.to_int64(predictions_idx)
        labels = tf.to_int64(labels)

        mask = tf.sequence_mask(seq_lens, timesteps, dtype=tf.float64, name="u_recall_mask")
        if weights is not None:
            weights = tf.cast(weights, tf.float64, name="weights_float64")
            mask = tf.multiply(mask, weights, name="weighted_u_recall_mask")

        r_tp, r_tp_update = _streaming_tp_at_k(predictions_idx=top_k_idx,
                                               labels=labels,
                                               mask=mask,
                                               k=k,
                                               user_ids=user_ids,
                                               num_users=num_users)

        r_predictions, r_predictions_update = _streaming_num_predictions(mask=mask,
                                                                         user_ids=user_ids,
                                                                         num_users=num_users,
                                                                         k=k)
        # Avoid nan's for division
        r_tp_1, r_predictions_1 = remove_empty_users(r_tp, r_predictions)
        r_tp_update_1, r_predictions_update_1 = remove_empty_users(r_tp_update, r_predictions_update)

        metric_array = tf.div(r_tp_1, r_predictions_1, name=scope)
        update_array = tf.div(r_tp_update_1, r_predictions_update_1, name="update")

        # Also output array with recall (or nan) for each user
        uf_metric_array = tf.div(r_tp, r_predictions, name=scope + "_unfiltered")
        uf_update_array = tf.div(r_tp_update, r_predictions_update, name="update_unfiltered")

        # Division to output one average value of recall for users with predictions
        metric = tf.reduce_mean(metric_array)
        update = tf.reduce_mean(update_array)

        # WILL NOT RETURN FILTERED ARRAYS
        if metrics_collections:
            tf.add_to_collection(metrics_collections + "_num", r_tp)
            tf.add_to_collection(metrics_collections + "_denom", r_predictions)

        if updates_collections:
            tf.add_to_collection(updates_collections + "_num", r_tp_update)
            tf.add_to_collection(updates_collections + "_denom", r_predictions_update)

        return metric, uf_metric_array, [update, uf_update_array]


def _find_idx(preds, target):
    return tf.where(tf.equal(preds, target))

def _lab_in_preds_per_timestep_per_batch(preds, targets):
    return tf.map_fn(lambda x: _lab_in_preds_per_timestep(*x), (preds, targets), dtype=tf.bool)

def _mrr_target_in_preds(preds, val):
    return 1/(tf.squeeze(_find_idx(preds, val)) + 1)

def _mrr_target_not_in_preds():
    return tf.cast(0.0, tf.float64)

def _calculate_mrr_for_seq(preds, targets, lab_in_preds):
    return tf.map_fn(lambda x: tf.cond(x[2],
                                lambda: _mrr_target_in_preds(x[0], x[1]),
                                       _mrr_target_not_in_preds),
             [preds, targets, lab_in_preds],
             dtype=tf.float64)



def _streaming_mrr_sum_at_k(predictions_idx, labels, mask, k, user_ids, num_users, name=None):
    with tf.name_scope(name, _at_k_name('mrr_sum', k),
                       (predictions_idx, labels)) as scope:

        pred_is_target = tf.equal(tf.tile(tf.expand_dims(labels, 2), [1, 1, k]), predictions_idx)
        target_idx_in_pred = tf.argmax(tf.cast(pred_is_target, tf.int32), 2, name="target_idx_in_pred")
        target_in_pred = tf.cast(tf.reduce_any(pred_is_target, 2, name="target_in_pred"), tf.int64)
        inv_mrr = target_idx_in_pred + target_in_pred
        mrr_per_seq = tf.minimum(tf.cast(target_in_pred, tf.float64), 1 / inv_mrr, name="remove_inf")

        masked = tf.multiply(mrr_per_seq, mask, name="masking")

        mrr = tf.reduce_sum(masked, axis=1)

        var = metric_variable([num_users], tf.float64, name=scope)
        return var, tf.scatter_add(var, user_ids, mrr)



def mrr_at_k(labels, predictions, seq_lens, timesteps, k, user_ids, num_users, name,
             input_top_k=False,
             metrics_collections=None, updates_collections=None,
             weights=None):

    with tf.name_scope(name, _at_k_name("mrr", k),
                       (predictions, labels)) as scope:
        if not input_top_k:
            _, predictions_idx = tf.nn.top_k(predictions, k)
        else:
            predictions_idx = predictions

        top_k_idx = tf.to_int64(predictions_idx)
        labels = tf.to_int64(labels)

        mask = tf.sequence_mask(seq_lens, timesteps, dtype=tf.float64, name="u_mrr_mask")
        if weights is not None:
            weights = tf.cast(weights, tf.float64, name="weights_float64")
            mask = tf.multiply(mask, weights, name="weighted_u_mrr_mask")

        mrr_tp, mrr_tp_update = _streaming_mrr_sum_at_k(predictions_idx=top_k_idx,
                                                        labels=labels,
                                                        mask=mask,
                                                        k=k,
                                                        user_ids=user_ids,
                                                        num_users=num_users)

        mrr_predictions, mrr_predictions_update = _streaming_num_predictions(mask=mask,
                                                                             user_ids=user_ids,
                                                                             num_users=num_users,
                                                                             k=k)

        # Avoid nan's for division
        mrr_tp_1, mrr_predictions_1 = remove_empty_users(mrr_tp, mrr_predictions)
        mrr_tp_update_1, mrr_predictions_update_1 = remove_empty_users(mrr_tp_update, mrr_predictions_update)

        metric_array = tf.div(mrr_tp_1, mrr_predictions_1, name=scope)
        update_array = tf.div(mrr_tp_update_1, mrr_predictions_update_1, name="update")

        # Also output array with recall (or nan) for each user
        uf_metric_array = tf.div(mrr_tp, mrr_predictions, name=scope + "_unfiltered")
        uf_update_array = tf.div(mrr_tp_update, mrr_predictions_update, name="update_unfiltered")


        # Division to output one average value of recall for users with predictions
        metric = tf.reduce_mean(metric_array)
        update = tf.reduce_mean(update_array)

        # WILL NOT RETURN FILTERED ARRAYS
        if metrics_collections:
            tf.add_to_collection(metrics_collections + "_num", mrr_tp)
            tf.add_to_collection(metrics_collections + "_denom", mrr_predictions)

        if updates_collections:
            tf.add_to_collection(updates_collections + "_num", mrr_tp_update)
            tf.add_to_collection(updates_collections + "_denom", mrr_predictions_update)

        return metric, uf_metric_array, [update, uf_update_array]

def average_user_metric(unfiltered):
    unfiltered = unfiltered[~np.isnan(unfiltered)]
    return np.mean(unfiltered)


def recall_test():
    # Test recall
    np.random.seed(0)

    # Dummy model params
    timesteps = 5
    k = 2
    num_users = 3
    num_items = 3

    # Dummy arrays
    predictions = [np.array([[[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
                             [[3, 2, 1], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]])]

    predictions.append(np.array([[[3, 1, 2], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
                                 [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]))

    labels = [np.zeros((2, 5)), np.ones((2, 5)) * 2]
    user_ids = [np.array([0, 1]), np.array([0, 1])]
    seq_lens = [np.array([5, 5]), np.array([5, 5])]

    _labels = tf.placeholder(tf.int64, [None, timesteps])
    _predictions = tf.placeholder(tf.float32, [None, timesteps, num_items])
    _seq_lens = tf.placeholder(tf.int32, [None])
    _user_ids = tf.placeholder(tf.int32, [None])

    fun = mrr_at_k

    with tf.Session() as sess:
        u_recall, u_recall_array, u_recall_update = fun(labels=_labels,
                                                        predictions=_predictions,
                                                        seq_lens=_seq_lens,
                                                        timesteps=timesteps,
                                                        k=k,
                                                        user_ids=_user_ids,
                                                        num_users=num_users,
                                                        name="test")

        u_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test")
        u_recall_vars_initializer = tf.variables_initializer(var_list=u_recall_vars)
        u_recall_vars_initializer.run()

        for i in range(2):
            fetches = [u_recall_update]
            feed_dict = {_labels: labels[i],
                         _predictions: predictions[i],
                         _seq_lens: seq_lens[i],
                         _user_ids: user_ids[i]}
            u = sess.run(fetches, feed_dict)
            pass

        r = sess.run(u_recall)
        r2 = sess.run(u_recall_array)
        print(r)
        print(r2)

if __name__ == "__main__":

    recall_test()