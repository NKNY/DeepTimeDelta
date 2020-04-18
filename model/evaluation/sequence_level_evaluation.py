from __future__ import division

from collections import defaultdict

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

    def __init__(self, timesteps_to_include, keep_individual_interactions=False):

        """
        Class to perform sequence level evaluation ONLY for timesteps indicated upon initialisation. These timesteps
        should be given as a numpy array or other iterable that can be converted to a list. These should be sorted in ascending order.
        The interactions upon which the evaluation metrics are calculated should also come in sorted order.

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
        self._found_all_timesteps = {}
        self._keep_individual_interactions = keep_individual_interactions
        if self._keep_individual_interactions:
            self._individual_included_interactions = defaultdict(list)

    def update_num_denom(self, metric, num, denom):
        if metric not in self._num_denom:
            self._num_denom[metric] = (0, 0)
        self._num_denom[metric] = (self._num_denom[metric][0]+num, self._num_denom[metric][1]+denom)

    def update_recall(self, top_indices, labels, seq_lens, weights=None):

        """
        Initialises the metric if previously uninitialised. Iterates over all timesteps in the batch
        and increments the numerator and denominator if one of the steps specified during init is
        encountered. No masking is required as the approach iterates only over the number of steps
        specified for each sequence in seq_lens. Weights is an optional argument which multiplies the
        numerattor and the denominator of the interaction by the weight of that particular timestep.

        Parameters
        ----------
        top_indices
        labels
        seq_lens
        timesteps
        weights : np.float32
            bs x timesteps

        Returns
        -------

        """

        if "recall" not in self._current_timestep:
            self._current_timestep["recall"] = 0
            self._included_timesteps_seen["recall"] = 0
            self._found_all_timesteps["recall"] = False

        batch_num = 0
        batch_denom = 0

        for seq in range(len(top_indices)):
            for t in range(seq_lens[seq]):
                # print("{}, {}".format(seq, t))
                if self._found_all_timesteps["recall"]:
                    break
                if self._current_timestep["recall"] == self._timesteps_to_include[self._included_timesteps_seen["recall"]]:
                    self._included_timesteps_seen["recall"] += 1
                    if labels[seq][t] in top_indices[seq][t]:
                        numerator_update = weights[seq, t] if weights is not None else 1
                        batch_num += numerator_update
                    else:
                        numerator_update = 0
                    denominator_update = weights[seq, t] if weights is not None else 1
                    if self._keep_individual_interactions and denominator_update > 0:
                        self._individual_included_interactions["recall"].append(numerator_update/denominator_update)
                    batch_denom += denominator_update
                self._current_timestep["recall"] += 1

                if len(self._timesteps_to_include) == self._included_timesteps_seen["recall"]:
                    self._found_all_timesteps["recall"] = True
            if self._found_all_timesteps["recall"]:
                break



        self.update_num_denom("recall", batch_num, batch_denom)


    def update_mrr(self, top_indices, labels, seq_lens, weights=None):

        """
        Initialises the metric if previously uninitialised. Iterates over all timesteps in the batch
        and increments the numerator and denominator if one of the steps specified during init is
        encountered. No masking is required as the approach iterates only over the number of steps
        specified for each sequence in seq_lens. Weights is an optional argument which multiplies the
        numerattor and the denominator of the interaction by the weight of that particular timestep.

        Parameters
        ----------
        top_indices
        labels
        seq_lens
        timesteps

        Returns
        -------

        """

        if "mrr" not in self._current_timestep:
            self._current_timestep["mrr"] = 0
            self._included_timesteps_seen["mrr"] = 0
            self._found_all_timesteps["mrr"] = False

        batch_num = 0
        batch_denom = 0

        for seq in range(len(top_indices)):
            for t in range(seq_lens[seq]):
                if self._found_all_timesteps["mrr"]:
                    break
                if self._current_timestep["mrr"] == self._timesteps_to_include[self._included_timesteps_seen["mrr"]]:
                    self._included_timesteps_seen["mrr"] += 1
                    if labels[seq][t] in top_indices[seq][t]:
                        numerator_update = 1. / (1. + np.where(top_indices[seq][t] == labels[seq][t])[0][0])
                        numerator_update = numerator_update * weights[seq,t] if weights is not None else numerator_update
                        batch_num += numerator_update
                    else:
                        numerator_update = 0
                    denominator_update = weights[seq, t] if weights is not None else 1

                    if self._keep_individual_interactions and denominator_update > 0:
                        self._individual_included_interactions["mrr"].append(numerator_update/denominator_update)

                    batch_denom += denominator_update

                self._current_timestep["mrr"] += 1

                # If all timesteps have been found break the evaluation
                if len(self._timesteps_to_include) == self._included_timesteps_seen["mrr"]:
                    self._found_all_timesteps["mrr"] = True

            if self._found_all_timesteps["mrr"]:
                break


        self.update_num_denom("mrr", batch_num, batch_denom)

    def calculate_metric(self, metric):
        try:
            return np.divide(*self._num_denom[metric])
        except KeyError as e:
            print("Metric {} not supported OR not yet initialised. Initialisation is via self.update_mrr".format(metric))
            sys.exit(1)
        except ZeroDivisionError as e2:
            print("Denominator is zero. Division by zero not possible.")
            sys.exit(1)



def calculate_batch_recall_at_k(top_indices, labels, seq_lens, timesteps, num_denom = (0., 0.)):

    """
    If current_num, current_denom not specified then returns the batch's numerator and denominator for recall@k metric.
    If the above are specified then the counts are increased by the current batch values.
    Parameters
    ----------
    top_indices
    labels
    seq_lens
    timesteps
    num_denom

    Returns
    -------

    """

    num, denom = num_denom
    batch_nums = np.zeros([len(top_indices), timesteps])

    for seq in range(len(top_indices)):
        for t in range(seq_lens[seq]):
            if labels[seq][t] in top_indices[seq][t]:
                batch_nums[seq][t] = 1

    num += np.sum(batch_nums)
    denom += np.sum(seq_lens)

    return num, denom

def calculate_batch_mrr_at_k(top_indices, labels, seq_lens, timesteps, num_denom = (0., 0.)):

    """
    If current_num, current_denom not specified then returns the batch's numerator and denominator for mrr@k metric.
    If the above are specified then the counts are increased by the current batch values.
    Parameters
    ----------
    top_indices
    labels
    seq_lens
    timesteps
    num_denom

    Returns
    -------

    """
    num, denom = num_denom
    batch_nums = np.zeros([len(top_indices), timesteps])

    for seq in range(len(top_indices)):
        for t in range(seq_lens[seq]):
            if labels[seq][t] in top_indices[seq][t]:
                batch_nums[seq][t] = 1./(1.+np.where(top_indices[seq][t]==labels[seq][t])[0][0])

    num += np.sum(batch_nums)
    denom += np.sum(seq_lens)
    return num, denom


# def calculate_batch_recall_at_k2(batch_logits, targets, k, count_ties = False):
#
#     batch_size = len(batch_logits)
#
#     recall = 0
#
#     # For every sequence in batch
#     for seq_id in range(batch_size):
#
#         # Predictions for each item
#         logits = batch_logits[seq_id, :]
#         # Index of the target
#         target = targets[seq_id]
#         # Update recall based on that sequence
#         recall += calculate_sequence_recall_at_k(logits, target, k, count_ties)
#
#     return recall / batch_size

# def calculate_sequence_recall_at_k2(logits, item_id, k, count_ties):
#
#     # Get the value of true item
#     item_value = logits[item_id]
#
#     # Count how many items exceeded that value
#     items_above_item_value = 0
#
#     # Iterate over all items or until we have more items exceeding
#     # item_value than k
#     for i in range(len(logits)):
#
#         # Skip the item itself
#         if i == item_id:
#             continue
#
#         # If score of an item is higher (or equal if we count ties)
#         # to the score of the true item increment count
#         if logits[i] > item_value or (count_ties and logits[i]):
#             items_above_item_value += 1
#
#         # If k items already have a higher score than the true item
#         # break out of the loop
#         if items_above_item_value >= k:
#             return False
#
#     return True

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


def _streaming_mrr_sum_at_k(predictions_idx,
                            labels,
                            mask,
                            k=None,
                            name=None):

    """
    Calculates the sum of individual MRR@k terms

    Parameters
    ----------
    predictions: tensor, batch_size x items, logits (or probabilities) for every item in TOP K
    labels: tensor, batch_size, indices of the true subsequent items
    k: int, number of top results to consider
    name: string or None, optional

    Returns
    -------
    tf.float64, sum of MRR@20 terms without division by Q
    Operation, increments the MRR@20
    """

    with tf.name_scope(name, _at_k_name('mrr_unnormalized', k),
                       (predictions_idx, labels, mask)) as scope:

        pred_is_target = tf.equal(tf.tile(tf.expand_dims(labels, 2), [1, 1, k]), predictions_idx)
        target_idx_in_pred = tf.argmax(tf.cast(pred_is_target, tf.int32), 2, name="target_idx_in_pred")
        target_in_pred = tf.cast(tf.reduce_any(pred_is_target, 2, name="target_in_pred"), tf.int64)
        inv_mrr = target_idx_in_pred + target_in_pred
        mrr_per_seq = tf.minimum(tf.cast(target_in_pred, tf.float64), 1 / inv_mrr, name="remove_inf")

        mrr_per_seq = tf.multiply(mrr_per_seq, mask, name="masking")

        # Combine MRR from all sequences in the batch
        mrr_per_batch = tf.reduce_sum(mrr_per_seq)

        # Return unnormalised MRR variable and the operation for incrementing it
        var = metric_variable([], tf.float64, name=scope)
        return var, tf.assign_add(var, mrr_per_batch, name="update")

def _streaming_num_predictions(mask,
                               k=None,
                               name=None):

    """
    Parameters
    ----------
    labels: tensor, batch_size, indices of the true subsequent items
    k: int, number of top results to consider
    name: string or None, optional

    Returns
    -------
    tf.float64, number of sequences seen so far in the batch
    Operation, increments the above
    """

    with tf.name_scope(name, _at_k_name('total_prediction_count', k), [mask]) as scope:

        # Find number of sequences
        total = tf.cast(tf.reduce_sum(mask), tf.float64)

        # Return unnormalised number of seqs variable and the operation for incrementing it
        var = metric_variable([], tf.float64, name=scope)
        return var, tf.assign_add(var, total, name="update")

def mrr_at_k(predictions,
             labels,
             seq_lens,
             timesteps,
             k,
             name=None,
             input_top_k=False,
             metrics_collections=None,
             updates_collections=None,
            weights=None):

    """
    Tensorflow-like evaluation procedure for MRR@k. While the target item is within
    top k results the value for a given sequence in 1/k, otherwise 0. This value is
    summed up over all sequences in all batches and is in the end divided by the
    total number of sequences in the epoch.

    Parameters
    ----------
    predictions: tensor, batch_size x items, logits (or probabilities) for every item in the batch
    labels: tensor, batch_size, indices of the true subsequent items
    k: int, number of top results to consider
    name: string or None, optional

    Returns
    -------
    metric: tf.float64, MRR@20 over the whole epoch
    update: Operation that increments mrr@k_over_seen_batches and the total number of seqs
            seen in the epoch so far

    """
    with tf.name_scope(name, _at_k_name("mrr", k),
                       (predictions, labels)) as scope:

        # Get top k indices for each sequence
        if not input_top_k:
            _, predictions_idx = tf.nn.top_k(predictions, k)
        else:
            predictions_idx = predictions

        top_k_idx = tf.to_int64(predictions_idx)
        labels = tf.to_int64(labels)

        mask = tf.sequence_mask(seq_lens, timesteps, dtype=tf.float64, name="mrr_mask")
        if weights is not None:
            weights = tf.cast(weights, tf.float64, name="weights_float64")
            mask = tf.multiply(mask, weights, name="weighted_mrr_mask")

        # Calculate the unnormalised term of mrr
        mrr_unnormalised, mrr_unnormalised_update = _streaming_mrr_sum_at_k(predictions_idx=top_k_idx,
                                                                            labels=labels,
                                                                            mask=mask,
                                                                            k=k)
        # Calculate the normalisation constant (Q on wikipedia)
        mrr_total_seqs, mrr_total_seqs_update = _streaming_num_predictions(mask=mask,
                                                                          k=k)

        # Calculate the resulting value
        metric = tf.div(mrr_unnormalised, mrr_total_seqs, name=scope)
        # Calculate the update
        update = tf.div(mrr_unnormalised_update, mrr_total_seqs_update, name="update")

        if metrics_collections:
            tf.add_to_collection(metrics_collections + "_num", mrr_unnormalised)
            tf.add_to_collection(metrics_collections + "_denom", mrr_total_seqs)
        if updates_collections:
            tf.add_to_collection(updates_collections + "_num", mrr_unnormalised_update)
            tf.add_to_collection(updates_collections + "_denom", mrr_total_seqs_update)

        return metric, update



def _streaming_tp_at_k(predictions_idx, labels, mask, k, name=None):

    with tf.name_scope(name, _at_k_name('true_positives', k),
                       (predictions_idx, labels, mask)) as scope:

        target_in_top_k = tf.reduce_any(tf.equal(tf.tile(tf.expand_dims(labels, 2), [1, 1, k]), predictions_idx), 2)

        target_in_top_k = tf.multiply(tf.cast(target_in_top_k, tf.float64), mask)

        tp_per_batch = tf.reduce_sum(target_in_top_k)

        var = metric_variable([], tf.float64, name=scope)
        return var, tf.assign_add(var, tp_per_batch, name="update")


def recall_at_k(labels, predictions, seq_lens, timesteps, k, name,
                input_top_k=False,
                metrics_collections=None, updates_collections=None,
                weights=None):

    """
    Implementation of recall
    Used for predictions of rank 2
    Requires lengths of unpadded sequences to ignore padded timesteps in keeping track
    of total predictions made

    Parameters
    ----------
    labels
    predictions
    seq_lens
    timesteps
    k
    name

    Returns
    -------

    """

    with tf.name_scope(name, _at_k_name("recall", k),
                       (predictions, labels)) as scope:
        # Get top k indices for each sequence

        if not input_top_k:
            _, predictions_idx = tf.nn.top_k(predictions, k)
        else:
            predictions_idx = predictions

        top_k_idx = tf.to_int32(predictions_idx)
        labels = tf.to_int32(labels)

        mask = tf.sequence_mask(seq_lens, timesteps, dtype=tf.float64, name="recall_mask")
        if weights is not None:
            weights = tf.cast(weights, tf.float64, name="weights_float64")
            mask = tf.multiply(mask, weights, name="weighted_recall_mask")

        # Calculate the unnormalised term of mrr
        recall_tp, recall_tp_update = _streaming_tp_at_k(predictions_idx=top_k_idx,
                                                         labels=labels,
                                                         mask=mask,
                                                         k=k)

        # Calculate the normalisation constant (Q on wikipedia)
        recall_total_seqs, recall_total_seqs_update = _streaming_num_predictions(mask=mask,
                                                                                 k=k)

        # Calculate the resulting value
        metric = tf.div(recall_tp, recall_total_seqs, name=scope)
        # Calculate the update
        update = tf.div(recall_tp_update, recall_total_seqs_update, name="update")

        if metrics_collections:
            tf.add_to_collection(metrics_collections + "_num", recall_tp)
            tf.add_to_collection(metrics_collections + "_denom", recall_total_seqs)
        if updates_collections:
            tf.add_to_collection(updates_collections + "_num", recall_tp_update)
            tf.add_to_collection(updates_collections + "_denom", recall_total_seqs_update)

        return metric, update

