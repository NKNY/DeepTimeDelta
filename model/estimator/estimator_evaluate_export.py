from __future__ import division

import os
import sys
import argparse

import numpy as np
import tensorflow as tf

import pickle

# Add model folder as the root of the project
PROJECT_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_MODEL_ROOT)

from model.evaluation.sequence_level_evaluation import SelectedTimestepsEvaluator
from model.evaluation.user_level_evaluation import SelectedTimestepsEvaluator as UserSelectedTimestepsEvaluator
from model.estimator.estimator_utils import remove_device_placement_from_uncompressed_model

def subset_paths(subset_data_dir):
    return {
        "X": os.path.join(subset_data_dir, "X.npy"),
        "y": os.path.join(subset_data_dir, "y.npy"),
        "seq_lens": os.path.join(subset_data_dir, "seq_lens.npy")
    }


def subset_arrays(subset_paths_dict):
    subset_arrays_dict = {}
    for k, v in subset_paths_dict.items():
        array = np.load(v)
        subset_arrays_dict[k] = array
    return subset_arrays_dict

def load_subset_data(data_path, subset_name, timesteps):

    """

    Loads the arrays for the specified subset located inside data_path. Loads 1-dimensional
    numpy array of timesteps only for which the evaluation is performed. If timesteps path is
    not specified (None) then timesteps becomes a numpy array specifying all timesteps via
    seq_lens.

    Parameters
    ----------
    data_path : str
        Location of the data
    subset_name : str
        Name of the subset. Should be {"train", "validation", "test"}
    timesteps : str
        Location of the timesteps file

    Returns
    -------
    selected_subset_arrays : dict
        Dict with keys X, y, seq_lens, containing validation/testing data
    timesteps : np.array 1-dimensional
        Not timesteps as in 20. Timesteps are the unique indicies of user-item interactions for which we are wanting to
        calculate the metrics.
    """

    selected_subset_paths = subset_paths(os.path.join(data_path, subset_name))
    selected_subset_arrays = subset_arrays(selected_subset_paths)

    load_selected_timesteps = lambda x: np.load(x)

    if timesteps is not None:
        selected_subset_timesteps = load_selected_timesteps(timesteps)
    else:
        selected_subset_timesteps = np.array(range(int(np.sum(selected_subset_arrays["seq_lens"]))))

    return selected_subset_arrays, selected_subset_timesteps

def pad_arrays(arrays, batch_size):

    """
    Pads the array with the last entry so the last batch would be same length as all others. Should not
    affect the evaluation as evaluation is only done on steps specified explicitly by user/sum(seq_lens).

    Parameters
    ----------
    arrays : dict
        Dict with keys X, y, seq_lens containing validation/testing data
    batch_size : int
        Size of the batch
    Returns
    -------
        Dict with keys X, y, seq_lens containing validation/testing data
    """

    rows_needed = batch_size - len(arrays["y"]) % batch_size
    return {k: np.pad(v, [(0,rows_needed)] + [(0,0)]*(len(v.shape)-1), "edge") for k,v in arrays.items()}

def evaluate(predictor, arrays, batch_size, timesteps, endpoint_type="python", keep_individual_interactions=False):

    """
    Iterate over the data using the predictor, make predictions each batch, calculate metrics specified inside
    Evaluators and return the evaluator objects with incremented counts.

    Parameters
    ----------
    predictor : tf.estimator.predictor
        Tensorflow predictor loaded from the exported model
    arrays : dict
        Dict with keys X, y, seq_lens containing validation/testing data
    batch_size : int
        Size of the batch during training - required as with Estimators it's not possible to have a variable batch size
        and thus the training batch size has to be provided.
    timesteps : np.array 1-dimensional
        Not timesteps as in 20. Timesteps are the unique indicies of user-item interactions for which we are wanting to
        calculate the metrics.
    endpoint_type : str
        Flag for whether the predictor was built using tensorflow serving or python based serving. If it's
        tensorflow serving then the inputs have to undergo an additional preprocessing inside this function.
        Note that for `tensorflow` value the only valid batch size is 1.
    keep_individual_interactions : bool
        When True SelectedTimestepsEvaluator also keeps an array in which the scores for each individual valid
        interaction are kept. Useful when combined with pickling of SelectedTimestepsEvaluator to later inspect
        results for individual interactions.
    Returns
    -------
    SelectedTimestepsEvaluator, UserSelectedTimestepsEvaluator
    """

    get_uids = lambda x: x["X"][:,0,0]

    # Objects containing numerator and denominator
    evaluator = SelectedTimestepsEvaluator(timesteps, keep_individual_interactions=keep_individual_interactions)
    u_evaluator = UserSelectedTimestepsEvaluator(timesteps, np.max(get_uids(arrays)) + 1)

    num_sequences = len(arrays["y"])

    # Iterate over each batch
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = start_idx + batch_size
        # Slice the indicies belonging to the batch
        batch = {k:v[start_idx:end_idx] for k,v in arrays.items()}
        # Split y from the rest of the data (now called inputs)
        y = batch["y"]
        inputs = {k:v for k,v in batch.items() if k != "y"}

        # If dealing with tensorflow serving model which expects json-serializable inputs process them
        if endpoint_type == "tensorflow":
            formatted_inputs = process_tensorflow_serving_inputs(inputs)
        else:
            formatted_inputs = inputs

        # Calculate predictions which are top_k items of shape batch_size * timesteps (20) * k
        # Does not contain scores
        prediction = predict(predictor, formatted_inputs)
        # Calculate metrics for the batch and increment the counts
        evaluator.update_recall(prediction, y, inputs["seq_lens"])
        evaluator.update_mrr(prediction, y, inputs["seq_lens"])
        u_evaluator.update_recall(prediction, y, inputs["seq_lens"], get_uids(inputs))
        u_evaluator.update_mrr(prediction, y, inputs["seq_lens"], get_uids(inputs))

    return evaluator, u_evaluator

def process_tensorflow_serving_inputs(inputs):
    return {k:v.squeeze().tolist() for k, v in inputs.items()}


def predict(predictor, inputs):

    """
    Feeds the input data to the predictor object and returns the predictions. Different predictor objects
    (working via tf, tf serving endpoint and python based endpoint) require different ways of passing
    data into them and format output differently.

    Parameters
    ----------
    predictor : tf.estimator.predictor | sagemaker.tensorflow.serving.Predictor | sagemaker.tensorflow.model.TensorFlowPredictor
    inputs : dict
        Keys defined by the serving_input_fn used for exporting the model. Usually "X", "y", "seq_lens". Associated np.ndarrays
        are bs x timesteps (20) x 3 for X, bs x timesteps for y and bs for seq_lens FOR tf.estimator.predictor
        and sagemaker.tensorflow.model.TensorFlowPredictor. In contrast sagemaker.tensorflow.serving.Predictor
        only accepts inputs 1 per batch and thus expects X: timesteps x 3, y: timesteps,  seq_lens: scalar.
        All values are of type np.int32.

    Returns
    -------
        (bs x) timesteps x 20 (num_predictions) with item indices as values.
    """
    predictor_type = type(predictor).__name__

    # Standard tensorflow predictor
    if predictor_type in ["tf.estimator.predictor", "SavedModelPredictor"]:
        return predictor(inputs)["top_k"]

    # Python based endpoint
    elif predictor_type == "sagemaker.tensorflow.model.TensorFlowPredictor":
        prediction = predictor.predict(inputs)
        top_k = prediction["outputs"]["top_k"]
        output_shape = [y["size"] for y in top_k["tensor_shape"]["dim"]]
        output_val = np.array(top_k["int_val"]).reshape(*output_shape)
        return output_val

    # Tensorflow serving based endpoint
    elif predictor_type in ["sagemaker.tensorflow.serving.Predictor", "Predictor"]:
        prediction = predictor.predict(inputs)
        return np.array(prediction["predictions"])
    else:
        print("Predict method failed. Supplied predictor type {} not supported.".format(predictor_type))

def main(**kwargs):

    data_path = kwargs["data_path"]

    metrics = ["mrr", "recall"]
    u_metrics = ["u_mrr", "u_recall"]

    pickle_non_user_evaluators = kwargs["pickle_non_user_evaluators"]

    # Config allowing soft device placement
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    export_path = kwargs["export_path"]
    if kwargs["reset_devices"]:
        new_path = os.path.join(os.path.dirname(export_path), "0000000001")
        try:
            remove_device_placement_from_uncompressed_model(export_path, new_path)
        except AssertionError:
            pass
        export_path = new_path


    # Load exported model
    predictor = tf.contrib.predictor.from_saved_model(export_path, config=config)

    # Load X, y, seq_lens; read steps from previously separated numpy file on which eval is performed
    # for both validation and test set (separate files). If no timestep file is provided (is None) then
    # the evaluated timesteps are the sum of lengths specified in seq_lens.npy
    validation_arrays, validation_timesteps = load_subset_data(data_path, "validation", kwargs["validation_timesteps"])
    test_arrays, test_timesteps = load_subset_data(data_path, "test", kwargs["test_timesteps"])

    batch_size = kwargs["batch_size"]

    # Add padding as batches have to be of the same size as training but data may not be divisible by the batch size
    # The padded steps are not used for calculating the metrics as they would be outside of the maximum value that is
    # eiter specified in the timesteps file or calculated as the sum of lengths specified in seq_lens.npy
    if len(validation_arrays["y"]) % batch_size != 0:
        validation_arrays = pad_arrays(validation_arrays, batch_size)

    if len(test_arrays["y"]) % batch_size != 0:
        test_arrays = pad_arrays(test_arrays, batch_size)

    # Performs the evaluation for all batches and returns the object holding all numerators and denominators
    validation_evaluator, validation_u_evaluator = evaluate(predictor,
                                                            validation_arrays,
                                                            batch_size,
                                                            validation_timesteps,
                                                            keep_individual_interactions=pickle_non_user_evaluators)
    test_evaluator, test_u_evaluator = evaluate(predictor, test_arrays, batch_size, test_timesteps,
                                                keep_individual_interactions=pickle_non_user_evaluators)

    # Merge non-user and user metrics
    validation_metrics = {
        **{k: validation_evaluator.calculate_metric(k) for k in metrics},
        **{k: validation_u_evaluator.calculate_metric(k) for k in u_metrics}
    }

    test_metrics = {
        **{k: test_evaluator.calculate_metric(k) for k in metrics},
        **{k: test_u_evaluator.calculate_metric(k) for k in u_metrics}
    }

    print("Validation metrics")
    for k,v in validation_metrics.items():
        print("{}: {}".format(k, v))

    print("Test metrics")
    for k,v in test_metrics.items():
        print("{}: {}".format(k, v))

    # If wanting to pickle the user evaluators
    if kwargs["pickle_folder"] is not None:
        try:
            os.makedirs(kwargs["pickle_folder"])
        except FileExistsError:
            print("Path {} already exists. Please provide a new path.".format(kwargs["pickle_folder"]))
            raise

        # Link objects to subsets and metrics
        pickle_path_to_evaluator = {
            "validation": {
                "evaluator": validation_evaluator,
                "u_evaluator": validation_u_evaluator
            },
            "test": {
                "evaluator": test_evaluator,
                "u_evaluator": test_u_evaluator
            }
        }

        # Export every object
        for subset, evaluators in pickle_path_to_evaluator.items():
            for evaluator_name, evaluator in evaluators.items():
                if not evaluator_name.startswith("u_") and not kwargs["pickle_non_user_evaluators"]: continue
                pickle_path = os.path.join(kwargs["pickle_folder"], "{}_{}".format(subset, evaluator_name))
                with open(pickle_path, "wb") as output_file:
                    pickle.dump(evaluator, output_file)
                    print("Dumped {}_{} to {}".format(subset, evaluator_name, pickle_path))




def none_or_str(value):
    if value == 'None':
        return None
    return value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--validation_timesteps", type=none_or_str, default=None)
    parser.add_argument("--test_timesteps", type=none_or_str, default=None)
    parser.add_argument("--reset_devices", type=bool, default=None)
    parser.add_argument("--pickle_folder", type=str, default=None)
    parser.add_argument("--pickle_non_user_evaluators", type=bool, default=False)
    args=parser.parse_args()

    main(**vars(args))