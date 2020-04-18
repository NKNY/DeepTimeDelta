# Helper functions used for printing, logging, saving

import os
import sys
import time

import subprocess

try:
    import GPUtil
except ModuleNotFoundError:
    subprocess.call(["pip", "install", "GPUtil"])
    import GPUtil

import psutil
from threading import Thread
from sys import argv

# Tensorboard logger cannot be used on sagemaker (which requires both py2 and py3)
# while other envs run py3 and have the module. Ignore logging on Sagemaker
if int(sys.version[0]) >= 3:
    try:
        from tensorboard_logger import Logger
    except ModuleNotFoundError as e:
        pass

def print_run_info(config):

    # Print current run's start time
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
    # Print run's settings
    print("Dataset: {}".format(config.dataset))
    print("Model id: {}".format(config.model_id))
    print("Embedding Size: {}".format(config.num_units))
    print("Std Delta_t: {}".format(config.standardise_d_t))
    print("Time embedding depth: {}".format(config.time_embedding_depth))
    print("Time embedding fun: {}".format(config.time_non_lin_fun))
    print("Trainable zero state: {}".format(config.trainable_zero_state))
    print("Number GPUs: {}".format(config.num_gpu))
    print("Epochs: {}".format(config.num_epochs))
    print("Train Batch Size: {}".format(config.train_batch_size))
    print("Validation Batch Size: {}".format(config.validation_batch_size))
    print("Keep Probs - Item: {},  User: {}, Time: {}".format(config.item_keep_prob, config.user_keep_prob, config.time_keep_prob if config.time_keep_prob else "No time"))
    print("User regularisation parameter: {}".format(config.user_reg_weight))
    print("Regularised weights: {}".format(config.user_related_weights))
    print("Random seed: {}".format(config.tf_random_seed))


def generate_log_folder_name(destination, *args):
    for arg in args:
        destination = os.path.join(destination, str(arg))
    return os.path.join(destination, time.strftime("%H%M", time.localtime()))


def retrieve_model_dir(log_dir_path, model_path, *args):
    if not model_path:
        model_dir = try_create_folder(generate_log_folder_name(log_dir_path, *args))
        print("Starting new run. Folder: {}".format(model_dir))
        return model_dir
    model_dir = os.path.join(log_dir_path, model_path)
    print("Loading previous run. Folder: {}".format(model_dir))
    return model_dir


def try_create_folder(destination):

    original_destination = destination
    directory_made = False
    attempts_made = 1
    while not directory_made:
        try:
            os.makedirs(destination)
            directory_made = True
            print("Log folder name: {}".format(destination))
        except FileExistsError as e:
            destination = original_destination + "_" + str(attempts_made)
            attempts_made += 1

    return destination

def start_cpu_gpu_profiling_from_argv():

    """
    Process argv. If argv contains keys --cpu_profiling_freq or --gpu_profiling_freq
    extracts the frequency from the parameters and starts the monitoring in separate threads
    that print the usage levels of all cpus and gpu's respectively.
    Parameters
    ----------

    Returns
    -------
    cpu_monitor : CPUMonitor or None
        A process running in a separate thread printing the CPU usage every n seconds as specified
        by the value following the --cpu_profiling_freq key in argv. None if n is not specified or None

    gpu_monitor : GPUMonitor or None
        A process running in a separate thread printing the GPU usage every n seconds as specified
        by the value following the --gpu_profiling_freq key in argv. None if n is not specified or None
    """

    args = argv[1:]

    class GPUMonitor(Thread):
        # Author in https://github.com/anderskm/gputil
        def __init__(self, delay, initial_delay=0):
            super(GPUMonitor, self).__init__()
            self.stopped = False
            self.delay = delay  # Time between calls to GPUtil
            self.initial_delay = initial_delay
            self.start()

        def run(self):
            if self.initial_delay is not None:
                time.sleep(self.initial_delay)
            while not self.stopped:
                GPUtil.showUtilization()
                time.sleep(self.delay)

        def stop(self):
            self.stopped = True

    class CPUMonitor(Thread):
        # Extended from https://github.com/anderskm/gputil
        def __init__(self, delay, initial_delay):
            super(CPUMonitor, self).__init__()
            self.stopped = False
            self.delay = delay
            self.initial_delay = initial_delay
            self.start()

        def run(self):
            if self.initial_delay is not None:
                time.sleep(self.initial_delay)
            while not self.stopped:
                print("CPU usage: {}".format(psutil.cpu_percent(percpu=True)))
                time.sleep(self.delay)

        def stop(self):
            self.stopped = True

    # Given a key, finds the key and retrieves the associated value (float).
    # If the key has no associated value then returns None
    def extract_profiling_value(key):
        value_index = args.index(key) + 1
        if value_index == len(args) or \
                args[value_index].startswith("--") or \
                args[value_index] is "None":
            value = None
        else:
            value = float(args[value_index])
        return value


    # Determine whether cpu or gpu profiling needed
    profile_cpu = "--cpu_profiling_freq" in args
    profile_gpu = "--gpu_profiling_freq" in args
    delay_cpu_profiling = "--cpu_profiling_delay" in args
    delay_gpu_profiling = "--gpu_profiling_delay" in args

    # Default outputs
    cpu_profiling_freq, gpu_profiling_freq = None, None
    cpu_profiling_delay, gpu_profiling_delay = None, None
    cpu_monitor, gpu_monitor = None, None

    # If cpu profiling needed
    if profile_cpu:
        # Frequency of profiling (float) or None
        cpu_profiling_freq = extract_profiling_value("--cpu_profiling_freq")
    # If cpu profiling needed
    if profile_gpu:
        # Frequency of profiling (float) or None
        gpu_profiling_freq = extract_profiling_value("--gpu_profiling_freq")

    if cpu_profiling_freq is not None:
        if delay_cpu_profiling:
            cpu_profiling_delay = extract_profiling_value("--cpu_profiling_delay")
        # Start a separate thread monitoring CPU usage with specified freq (or None)
        cpu_monitor = CPUMonitor(cpu_profiling_freq, cpu_profiling_delay)
    if gpu_profiling_freq is not None:
        if delay_gpu_profiling:
            gpu_profiling_delay = extract_profiling_value("--gpu_profiling_delay")
        # Start a separate thread monitoring GPU usage with specified freq (or None)
        gpu_monitor = GPUMonitor(gpu_profiling_freq, gpu_profiling_delay)

    return cpu_monitor, gpu_monitor

def end_cpu_gpu_profiling(cpu_monitor, gpu_monitor):

    """
    Companion function for `start_cpu_gpu_profiling_from_argv`. Stops the monitors if those
    were previously running. Otherwise does nothing.

    Parameters
    ----------
    cpu_monitor : CPUMonitor or None
        A process running in a separate thread printing the CPU usage every n seconds as specified
        by the value following the --cpu_profiling_freq key in argv. None if n is not specified or None

    gpu_monitor : GPUMonitor or None
        A process running in a separate thread printing the GPU usage every n seconds as specified
        by the value following the --gpu_profiling_freq key in argv. None if n is not specified or None


    Returns
    -------

    """

    if cpu_monitor is not None:
        cpu_monitor.stop()
    if gpu_monitor is not None:
        gpu_monitor.stop()




