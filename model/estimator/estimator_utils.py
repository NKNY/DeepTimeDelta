from __future__ import division

import collections
import six
from six import iteritems

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

from tensorflow.python.saved_model import tag_constants

from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

import json
import argparse
import os

import model.estimator.estimator_dataset as estimator_dataset

class RunConfig(tf.contrib.learn.RunConfig):
  def uid(self, whitelist=None):
    """Generates a 'Unique Identifier' based on all internal fields.
    Caller should use the uid string to check `RunConfig` instance integrity
    in one session use, but should not rely on the implementation details, which
    is subject to change.
    Args:
      whitelist: A list of the string names of the properties uid should not
        include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
        includes most properties user allowes to change.
    Returns:
      A uid string.
    """
    if whitelist is None:
      whitelist = run_config._DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    for k in whitelist:
      state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
         sorted(ordered_state['_cluster_spec'].as_dict().items(),
                key=lambda t: t[0])
      )
    return ', '.join(
        '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.
    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook.
      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)

def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser


class MetadataHook(SessionRunHook):
    def __init__ (self,
                  save_steps=None,
                  save_secs=None,
                  report_tensor_allocation_upon_oom=False,
                  output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._report_tensor_allocation_upon_oom = report_tensor_allocation_upon_oom
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        self._writer = tf.summary.FileWriter (self._output_dir, tf.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                              report_tensor_allocations_upon_oom=self._report_tensor_allocation_upon_oom)
            if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()


def init_basic_argument_parser(default_json_file):

    """
    Builds an ArgumentParser object that processes user provided arguments or uses default values.
    Both which arguments to use and the defaults are specified in the json file. Additional processing
    also takes place based on the values specified e.g. based on dataset provided as one parameter
    the output will also contain the size of the dataset. User provides parameters using the following
    formula: --param1_name param1_value --param2_name param2_value

    Parameters
    ----------
    default_json_file : str or Path
        Path of the json file containing all specifications for the user provided parameters and the default values.

    Returns : argparse.Namespace
        Object with its properties reflecting the user provided/default values.

    -------

    """

    with open(default_json_file, "r") as f:
        config = json.load(f)

    # Set up user input parsing rules
    parser = argparse.ArgumentParser()
    for k, v in iteritems(config):
        required = v["required"] if "required" in v else False
        default = v["default"] if "default" in v else None
        help = v["help"] if "help" in v else ""
        nargs = v["nargs"] if "nargs" in v else "?"
        # Note: Booleans are not natively processed by this module, need custom logic to transform str into bool
        arg_type = eval(v["type"] if "type" in v else "str")
        arg_is_bool = arg_type == bool
        type = str2bool if arg_is_bool else arg_type

        parser.add_argument("--%s" % k,
                            type=type,
                            nargs=nargs,
                            required=required,
                            default=default,
                            help=help
                            )
    # Process user input based on provided rules
    args = parser.parse_args()

    # Add parameters specific to the dataset, execution location and model id to a list of dicts
    additional_modifications = []
    additional_modifications.append(get_dataset_params(args.dataset))
    additional_modifications.append(get_exec_loc_params(args.exec_loc))
    additional_modifications.append({"X_cols_to_use": cols_to_use_for_model(args.model_id)})

    # Flatten list of dict to a dict with k:v being the same as in the output object
    additional_modifications = {k:v for d in additional_modifications for k,v in iteritems(d)}

    # Apply all the additional modifications to the output object
    modify_parameters(args, additional_modifications)

    return args

def str2bool(v):

    """
    Custom logic to process booleans by ArgumentParser

    Parameters
    ----------
    v : str
        User provided string that logically should be equivalent to a bool.
    Returns
    -------
    bool
        In case of successful parsing returns the parsed value.
    Raises
    -------
    ArgumentTypeError
        When the string could not be reduced to a boolean.
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_exec_loc_params(exec_loc):

    """
    Return the locations of all datasets and where all logs are written if a supported location is provided.

    Parameters
    ----------
    exec_loc : str
        Either 'cluster' or 'local'

    Returns
    -------
    dict
        Keys data_dir_path and log_dir_path with absolute paths to the locations of general data and log directories

    Raises
    -------
    LookupError
        When user provides a string that is not supported.
    """

    if exec_loc == "cluster":
        return {"data_dir_path": "/home/nfs/nknyazev/thesis/data/numpy",
                # "log_dir_path": "/home/nfs/nknyazev/thesis/logs" # User space - limited space
                "log_dir_path": "/tudelft.net/staff-bulk/ewi/insy/MMC/nknyazev/logs" # Bulk storage
                # "log_dir_path": "/dev/shm/nknyazev/" # On the machine executing task - limited space
                }
    elif exec_loc == "local":
        return {
            #"data_dir_path": "/Users/nknyazev/Documents/Delft/Thesis/temporal/data/numpy",
            "data_dir_path": "/Users/nknyazev/Documents/Delft/Thesis/temporal/data/processed/final",
                "log_dir_path": "/Users/nknyazev/Documents/Delft/Thesis/temporal/logs/local"
                }
    elif exec_loc == "sagemaker":
        return {
            # Sagemaker Estimator's fit method should be called as follows:
            # estimator.fit(inputs={"data_root": path_to_all_datasets})
            "data_dir_path": os.environ.get("SM_CHANNEL_SAGEMAKER_DATA_ROOT"),
            "log_dir_path": None
        }
    else:
        raise LookupError("Unsupported execution location: {}".format(exec_loc))

def get_dataset_params(dataset):

    '''
    Processes the dataset and returns params pertaining to that specific dataset
    e.g. filename, num of users and items

    Parameters
    ----------
    flag : string
        Name of the dataset being processed

    Returns
    -------
    list
        (*(string, value)), pairs of attribute and value

    '''
    flag = dataset

    if flag == "lastfm_10_pc":
        dataset_obj = estimator_dataset.EstimatorDatasetLastfm10pcFinal()
    if flag == "ml-10m":
        dataset_obj = estimator_dataset.EstimatorDatasetML10M()
    if "rtl" in flag:
        dataset_obj = estimator_dataset.EstimatorDatasetRTL()

    print("Dataset object: {}".format(dataset_obj))

    # Get names and values of all the relevant parameters
    params = dict([(a, getattr(dataset_obj, a)) for a in dir(dataset_obj) if not a.startswith('__')])
    return params

def modify_parameters(object, param_dict):

    """
    Modifies all the attributes of the object based on a {k:v} dict, where k is the attribute and v is the assigned value
    Parameters
    ----------
    object : object
        Object whose properties are overwritten.
    param_dict - dict
        k:v of attribute and the new value

    Returns
    object : object
        Modified object
    -------

    """

    for k, v in iteritems(param_dict):
        setattr(object, k, v)
    return object

def cols_to_use_for_model(model_id):

    """
    Return the features col_number:feature_name from the numpy array to use.
    Parameters
    ----------
    model_id : int
        Number of the model from range(0, 8)

    Returns
    -------
    dict
        col_num:feature_name which to use in the numpy arrays and how to name them.

    Raises
    -------
    LookupError
        Model id is not in the supported range.
    """

    if model_id == 0:
        return {0: "uid", 1: "iid"}
    elif 0 < model_id <= 7:
        return {0: "uid", 1: "iid", 2: "delta_t"}
    else:
        raise LookupError("Model id not within supported range.")


def make_early_stopping_hook(estimator,
                             metric_name,
                             max_train_steps_without_improvement,
                             min_steps,
                             run_every_secs=None,
                             run_every_steps=None):

    """

    Returns either a stop_if_no_decrease_hook or stop_if_no_increase_hook whichever appropriate for the provided metric.
    This hook stops training if the metrics hasn't improved since the previous call to the hook.

    Parameters
    ----------
    estimator : tf.estimator.Estimator
        Estimator for which the hook is going to be implemented.
    metric : str
        Name of the metric which is listened to.
    max_train_steps_without_improvement : int
        Number of steps which need to have passed without improvement  since last improvement of the specified metric
        to trigger early stopping.
    min_steps : int
        Number of first steps for which early stopping is ignored even if there is no metric improvement.
    run_every_secs : int or None
        Frequency of calls to the hook in seconds. Either this or run_every_steps has to be specified. Only executed
        if a new checkpoint is available.
    run_every_steps : int or None
        Frequency of calls to the hook in steps. Either this or run_every_steps has to be specified. Only executed
        if a new checkpoint is available.
    Returns
    -------
    tf.contrib.estimator.stop_if_no_increase_hook or tf.contrib.estimator.stop_if_no_decrease_hook
        Early stopping hook with specified parameters.
    Raises
    -------
    ValueError
        Raised in case the provided metric isn't recognised (hard-coded inside this method).

    """

    stop_if_no_decrease_metrics = ["loss"]
    stop_if_no_increase_metrics = ["mrr", "recall", "u_mrr", "u_recall"]

    if metric_name in stop_if_no_decrease_metrics:
        hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name=metric_name,
            # Number of TRAINING steps, finds the first eval batch it encounters by going back in steps, so no difference between small values like 1-10
            max_steps_without_decrease=max_train_steps_without_improvement,
            min_steps=min_steps,
            run_every_secs=run_every_secs,
            # Number of TRAINING steps, waits until the next eval, so no difference between small values
            run_every_steps=run_every_steps # TODO Might need to fix this - currently runs every time there is a new checkpoint - might want to decouple
        )
    elif metric_name in stop_if_no_increase_metrics:
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator=estimator,
            metric_name=metric_name,
            # Number of TRAINING steps, finds the first eval batch it encounters by going back in steps, so no difference between small values like 1-10
            max_steps_without_increase=max_train_steps_without_improvement,
            min_steps=min_steps,
            run_every_secs=run_every_secs,
            # Number of TRAINING steps, waits until the next eval, so no difference between small values
            run_every_steps=run_every_steps # TODO Might need to fix this - currently runs every time there is a new checkpoint - might want to decouple
        )
    else:
        supported_metrics_str = ", ".join([str(x) for x in stop_if_no_decrease_metrics+stop_if_no_increase_metrics])
        raise ValueError("Unknown metrics provided to make_early_stopping_hook. Currently supported metrics: " + supported_metrics_str)
    return hook

def remove_device_placement_from_uncompressed_model(path_to_model, output_path):

    """
    Loads the model defined by MetaGraphDef (saved_model.pb + variables folder), clears devices and then saves
    to the output path. Currently supports only one signature def map, which is DEFAULT_SERVING_SIGNATURE_DEF_KEY.

    Parameters
    ----------
    path_to_model : str
        Absolute path to the folder, which contains saved_model.pb and variables.
    output_path : str
        Absolute path to the folder, which will contain new saved_model.pb and variables.

    Returns
    -------

    """

    builder = tf.saved_model.Builder(output_path)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            meta_graph = tf.saved_model.loader.load(sess, [tag_constants.SERVING], path_to_model, clear_devices=True)
            key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            sig = meta_graph.signature_def[key]
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map={key: sig})
    builder.save()