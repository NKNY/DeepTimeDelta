import tensorflow as tf
from copy import deepcopy

import model.estimator.estimator_model as estimator_model

# TODO Add support for multiple metrics and their functions to filter on
def best_checkpoint(model_dir, eval_dir, metric):

    """

    Returns best checkpoint in model directory using metrics from evaluation directory as well as metrics themselves.
    The association between the two is done based on the step number which is used as keys when extracting metrics and
    is normally included at the end of the checkpoint filename e.g '/model_dir/model.ckpt-1234'.
    If the best result does not have an associated checkpoint (e.g. best result was achieved with a checkpoint that
    hadn't been kept due to RunConfig's keep_checkpoint_max) attempts to return the next best config.

    Parameters
    ----------
    model_dir : str or Path
        Indicates the location of checkpoints.
    eval_dir : str or Path
        Indicates the location of metric containing file (events.out.tfevents)
    metric : str
        Name of the metric for which the best checkpoint is returned. Generally is but not necessarily the metric
        used for early stopping.

    Returns
    -------
    str
        Path of the checkpoint with the best metric
    metrics : dict
        Keys are metric names and values are the associated results for the returned checkpoint.

    Raises
    -------
    ValueError if none of the metrics in eval_dir have an associated checkpoint in model_dir.
    """

    model_dir = str(model_dir)
    eval_dir = str(eval_dir)

    # {step:metrics_dict}
    metrics = tf.contrib.estimator.read_eval_metrics(eval_dir)
    # Sorted iterable((step,metrics_dict))
    step_metrics = [(step, metrics[step]) for step in sorted(metrics,
                                                            key=lambda x: (metrics[x][metric]),
                                                            reverse=True)]
    # iterable of checkpoint_path strings
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir=model_dir).all_model_checkpoint_paths
    # step:checkpoint_path
    step_checkpoint = {int(c.split("-")[-1]):c for c in checkpoints}

    for step, metrics in step_metrics:
        if step in step_checkpoint:
            return step_checkpoint[step], metrics
    raise ValueError("None of the checkpoints in model_dir are among the checkpoints for which metrics data is available.")


def evaluate_multiple_checkpoints(model_dir, eval_dir, num_checkpoints, metric,
                                  input_fn, run_config, hparams, num_steps_in_eval):

    """

    Orders all checkpoints in `model_dir` by their performance on `metric` recorded in `eval_dir`
    from the best to worst, iterates over `num_checkpoints` of them and returns the performance
    on the dataset specified in `input_fn`. The run and model parameters are specified in `run_config`
    and `hparams` respectively. The dataset returned by `input_fn` is evaluated for `num_steps_in_eval`.
    If there are fewer than `num_checkpoints` checkpoints in `model_dir` all the `model_dir`
    checkpoints are evaluated. Only checkpoints for which there is metric information are counted
    towards the number of checkpoints evaluated.

    Parameters
    ----------
    model_dir : str or Path
        Indicates the location of checkpoints.
    eval_dir : str or Path
        Indicates the location of metric containing file (events.out.tfevents)
    metric : str
        Name of the metric for which the best checkpoint is returned. Generally is but not necessarily the metric
        used for early stopping.
    num_checkpoints : int
        Number of best checkpoints to retrieve
    input_fn : method
        Usually a lambda function specifying the actual input function with its parameters that is fed to the estimator.
    run_config : tf.estimator.RunConfig
        Object specifying runtime parameters e.g. model and check pointing directory, check pointing frequency.
        See documentation for tf.estimator.RunConfig for more information.
    hparams : tf.contrib.training.HParams
        Object containing all specifics about the model e.g. num_hidden and batch_size.
    num_steps_in_eval : int
        Length of one evaluation run.

    Returns
    -------
    None
    """

    model_dir = str(model_dir)
    eval_dir = str(eval_dir)

    # {step:metrics_dict}
    metrics = tf.contrib.estimator.read_eval_metrics(eval_dir)
    # Sorted iterable((step,metrics_dict))
    step_metrics = [(step, metrics[step]) for step in sorted(metrics,
                                                             key=lambda x: (metrics[x][metric]),
                                                             reverse=True)]
    # iterable of checkpoint_path strings
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir=model_dir).all_model_checkpoint_paths
    # step:checkpoint_path
    step_checkpoint = {int(c.split("-")[-1]): c for c in checkpoints}

    checkpoints_visited = 0

    for step, metrics in step_metrics:
        if step in step_checkpoint:
            checkpoint_path = step_checkpoint[step]
            print("Running checkpoint {}\nIts performance on evaluation set was {}".format(step, metrics))

            # Remove model_dir from previous run_config as that causes evaluation to ignore warm_start_from
            new_config = deepcopy(run_config)
            setattr(new_config, "_model_dir", None)

            estimator = estimator_model.create_estimator(new_config, hparams, warm_start_from=checkpoint_path)

            results = estimator.evaluate(input_fn=input_fn, steps=num_steps_in_eval)
            print("Results: {}".format(results))

            checkpoints_visited += 1

            if checkpoints_visited == num_checkpoints:
                print("Evaluated required number of checkpoints ({}). Finishing multiple checkpoint evaluation.".format(
                    num_checkpoints
                ))
                return

    print("Exhausted all available checkpoints. Finishing multiple checkpoint evaluation.")
    return


def custom_checkpoint_compare_fn(default_key = "loss"):
    def custom_checkpoint_compare_fn_wrapped(best_eval_result, current_eval_result):
        """Compares two evaluation results and returns true if the current result is HIGHER
        than the previous best IF the metric is not loss. If metric is loss then returns
        True if current result is LOWER than previous lowest.
        Both evaluation results should have the values for MetricKeys.LOSS, which are
        used for comparison.
        Args:
          best_eval_result: best eval metrics.
          current_eval_result: current eval metrics.
        Returns:
          True if the loss of current_eval_result is smaller; otherwise, False.
        Raises:
          ValueError: If input eval result is None or no loss is available.
        """
        if not best_eval_result or default_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or default_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')
        if default_key == "loss":
            return best_eval_result[default_key] > current_eval_result[default_key]
        else:
            return best_eval_result[default_key] < current_eval_result[default_key]

    return custom_checkpoint_compare_fn_wrapped