from __future__ import division

import os

# Prevent sagemaker timing out
os.environ["S3_CONNECT_TIMEOUT_MSEC"] = "60000"
os.environ["S3_REQUEST_TIMEOUT_MSEC"] = "60000"

import sys
from copy import deepcopy
import subprocess
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# Add model folder as the root of the project

PROJECT_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_MODEL_ROOT)

import model.estimator.estimator_model as estimator_model
import model.estimator.estimator_utils as estimator_utils
import model.estimator.estimator_checkpointing as estimator_checkpointing
import model.estimator.estimator_sagemaker as estimator_sagemaker

from model.utils.helper import print_run_info, retrieve_model_dir, start_cpu_gpu_profiling_from_argv, end_cpu_gpu_profiling

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    # Path to file specifying all runtime and model parameters and how to process user command line input.
    config_file_path = os.path.join(PROJECT_MODEL_ROOT, "configs/default.json")

    # Argparse namespace combining json defaults and user command line inputs
    args = estimator_utils.init_basic_argument_parser(config_file_path)
    # Transfer all k:v pairs from the Argparse namespace to HParams
    hparams = tf.contrib.training.HParams(**vars(args))
    # Print stats about the current run
    print_run_info(args)

    # Calculate the number of steps needed to complete one epoch for each of the subsets
    steps_in_epoch_train = np.ceil(args.num_samples["train"] / args.train_batch_size)
    steps_in_epoch_val = np.ceil(args.num_samples["validation"] / args.validation_batch_size)

    # Number of training steps to perform during train_and_evaluate
    total_train_steps = int(steps_in_epoch_train * args.num_epochs)
    # Minimum number of steps during which no early stopping can occur
    train_steps_without_stopping = steps_in_epoch_train * args.train_epochs_without_stopping
    # Number of steps during which no metric improvement happened that is needed to initiate early stopping
    max_train_steps_without_improvement = int(steps_in_epoch_train * args.max_train_epochs_without_improvement)
    # Number of evaluation steps that are performed during each of the calls to evaluation during train_and_evaluate
    eval_steps_during_train = int(steps_in_epoch_val * args.eval_pc_during_train)
    # Number of steps during which evaluation is not performed
    train_steps_without_evaluation = int(steps_in_epoch_train * args.delay_evaluation_epochs)


    throttle_secs = args.save_checkpoints_secs
    save_checkpoints_steps = None
    # Only one of secs and steps for checkpointing frequency is allowed to be saved
    assert (args.save_checkpoints_secs is not None) ^ (args.checkpoint_freq_epochs is not None)
    if args.checkpoint_freq_epochs is not None:
        save_checkpoints_steps = np.ceil(steps_in_epoch_train * args.checkpoint_freq_epochs) # TODO Ensure this is never zero
        throttle_secs = 1

    # Number of towers
    num_shards = args.num_gpu if args.num_gpu > 0 else 1

    # Path object pointing to the location where the checkpoints and results are saved
    # If model path is provided then load a previously instantiated model and train/evaluate
    # using the previous values.

    folder_naming_vars = []
    for x in args.folder_naming_vars:
        folder_naming_vars.append(eval(x)) # For some reason list comprehension doesn't work


    execution_date = time.strftime("%Y%b%d", time.localtime()) if args.execution_date is None else args.execution_date

    # Sagemaker provides model_dir or when running elsewhere creates new model_dir or loads previous run via model_path
    if hparams.model_dir is None:
        model_dir = retrieve_model_dir(args.log_dir_path, args.model_path, execution_date, *folder_naming_vars)
        hparams.set_hparam("model_dir", model_dir)
        setattr(args, "model_dir", model_dir)

    # Path pointing to the location of the current data set (e.g. .../numpy/lastfm_10_pc)
    data_dir = os.path.join(
        args.data_dir_path if args.data_dir_path else "",
        "" if args.exec_loc == "sagemaker" else args.dataset,
        "tfrecords" if args.input_data_format == "tfrecords" else "",
        "sharded" if args.exec_loc == "sagemaker" else ""
    )

    # Tensorflow device allocation settings
    config_proto = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement,
                                    log_device_placement=args.log_device_placement)
    config_proto.gpu_options.allow_growth = True

    # Object specifying current run settings e.g. logging frequency and num of check points saved.
    run_config = tf.estimator.RunConfig(
        tf_random_seed=args.tf_random_seed,
        model_dir=args.model_dir,
        session_config=config_proto,
        save_summary_steps=20,
        save_checkpoints_steps=save_checkpoints_steps if not args.overwrite else 1,
        save_checkpoints_secs=args.save_checkpoints_secs,
        keep_checkpoint_max=args.keep_checkpoint_max,
        log_step_count_steps=100,
    )

    # Instantiate an Estimator object with the model_fn from this module.
    estimator = estimator_model.create_estimator(run_config, hparams)

    # The degree of shuffling - int. Check tf.Data.dataset.shuffle() for additional documentation.
    shuffle_train = int(args.num_samples["train"]*args.shuffle_train) if args.shuffle_train else 1
    shuffle_val = int(args.num_samples["val"]*args.shuffle_test) if args.shuffle_test else 1

    additional_arrays = ["weights"] if args.use_weights else []

    # https://cloud.google.com/blog/products/gcp/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine
    with tf.name_scope("TrainSpec_and_hook"):
        with tf.name_scope("Early_stop_hook"):
            try:
                os.makedirs(estimator.eval_dir())
            except FileExistsError:
                pass

            training_hooks = []

            early_stopping_hook = estimator_utils.make_early_stopping_hook(
                estimator=estimator,
                metric_name=args.key_metrics[0],
                max_train_steps_without_improvement=max_train_steps_without_improvement,
                min_steps=train_steps_without_stopping,
                run_every_secs=None,
                run_every_steps=1
            )
            if args.early_stopping:
                training_hooks.append(early_stopping_hook)

            # from https://stackoverflow.com/questions/45719176/how-to-display-runtime-statistics-in-tensorboard-using-estimator-api-in-a-distri
            if args.metadata_hook_saving_frequency:
                runtime_stats_hook = estimator_utils.MetadataHook(save_secs=args.metadata_hook_saving_frequency, output_dir=str(args.model_dir))
                training_hooks.append(runtime_stats_hook)

            if args.profiler_hook:
                profiler_hook = tf.train.ProfilerHook(
                    save_steps=10, save_secs=None, output_dir=str(os.path.join(args.model_dir, "timelines")), show_memory=True
                )
                training_hooks.append(profiler_hook)

            # Debugging
            if args.tensorboard_debug_address:
                debug_hook = tf_debug.TensorBoardDebugHook(args.tensorboard_debug_address)
                training_hooks.append(debug_hook)
            if args.debug:
                debug_hook = tf_debug.LocalCLIDebugHook()
                training_hooks.append(debug_hook)
            if args.debug:
                debug_hook = tf_debug.DumpingDebugHook(args.debug_dump_path)
                training_hooks.append(debug_hook)


        with tf.name_scope("TrainSpec"):
            train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: estimator_model.input_fn(
                    data_dir=data_dir,
                    subset="train",
                    num_shards=num_shards,
                    batch_size=args.train_batch_size,
                    X_cols_to_use=args.X_cols_to_use,
                    input_data_format=args.input_data_format,
                    shuffle=shuffle_train,
                    additional_arrays=additional_arrays,
                    delta_t_mean=args.delta_t_mean,
                    delta_t_std=args.delta_t_std
                ),
                max_steps=total_train_steps if not args.overwrite else 10,
                hooks=training_hooks)

    with tf.name_scope("EvalSpec_and_exporter"):
        with tf.name_scope("Exporter"):
            # TODO Define function to process the input e.g. seq for the whole user - this function used to simulate real data
            exporters = []
            for key_metric in args.key_metrics:
                exporters.append(tf.estimator.BestExporter(
                    name=key_metric,
                    serving_input_receiver_fn=estimator_model.serving_input_fn(args),
                    compare_fn=estimator_checkpointing.custom_checkpoint_compare_fn(default_key=key_metric),
                    exports_to_keep=1,
                    as_text=False
                ))

        with tf.name_scope("EvalSpec"):
            eval_spec = tf.estimator.EvalSpec(
                input_fn = lambda: estimator_model.input_fn(
                    data_dir=data_dir,
                    subset="validation",
                    num_shards=num_shards,
                    batch_size=args.validation_batch_size,
                    X_cols_to_use=args.X_cols_to_use,
                    input_data_format=args.input_data_format,
                    shuffle=shuffle_val,
                    additional_arrays=additional_arrays,
                    delta_t_mean=args.delta_t_mean,
                    delta_t_std=args.delta_t_std
                ),
                exporters=exporters if args.use_exporter else None, #TODO
                steps=eval_steps_during_train if not args.overwrite else 1,
                throttle_secs=throttle_secs,
                start_delay_secs=args.start_delay_secs
            )

    if train_steps_without_evaluation > 0:
        print("Starting preliminary training for {} steps during which no evaluation is performed.".format(train_steps_without_evaluation))
        estimator.train(
            input_fn=lambda: estimator_model.input_fn(
                data_dir=data_dir,
                subset="train",
                num_shards=num_shards,
                batch_size=args.train_batch_size,
                X_cols_to_use=args.X_cols_to_use,
                input_data_format=args.input_data_format,
                shuffle=shuffle_train,
                additional_arrays=additional_arrays,
                delta_t_mean=args.delta_t_mean,
                delta_t_std=args.delta_t_std
            ),
            max_steps=train_steps_without_evaluation if not args.overwrite else 10,
            hooks=training_hooks
        )
        # Export the model for the offchance that the metrics for validation don't improve after the first run
        # when I believe no export is performed
        export_dir = os.path.join(args.model_dir, "export", args.key_metrics[0])
        estimator.export_savedmodel(export_dir, estimator_model.serving_input_fn(args), strip_default_attrs=True)

    print("Starting Train and Evaluate for {} training steps with Evaluation every {} second(s) or {} steps for {} evaluation steps.".format(
        total_train_steps, throttle_secs, save_checkpoints_steps, eval_steps_during_train))

    with tf.name_scope("Train_and_Evaluate"):
        tf.estimator.train_and_evaluate(
            estimator=estimator,
            train_spec=train_spec,
            eval_spec=eval_spec
        )
    if args.exec_loc == "sagemaker":
        updated_model_path = estimator_sagemaker.sagemaker_postprocessing(args)
        predictor_param_names =  ["predictor_s3_input_path", "predictor_s3_output_path", "predictor_batch_size"]
        predictor_params = [getattr(args, x) for x in predictor_param_names]
        if np.all([x is not None for x in predictor_params]):
            estimator_sagemaker.predict_s3_numpy(
                saved_model_path=updated_model_path,
                input_s3_path=args.predictor_s3_input_path,
                output_s3_path=args.predictor_s3_output_path,
                batch_size=args.predictor_batch_size
            )
    else:

        # Evaluate trained model
        steps_in_epoch_test = np.ceil(args.num_samples["test"] / args.validation_batch_size)
        shuffle_test = args.num_samples["train"] if args.shuffle_test else 1

        with tf.name_scope("Evaluate_trained_model"):

            train_input_fn = lambda: estimator_model.input_fn(
                data_dir=data_dir,
                subset="train",
                num_shards=num_shards,  #Switch to one and adjust bs/num_gpu for single device
                batch_size=args.train_batch_size,  #TODO Does that work for serving
                X_cols_to_use=args.X_cols_to_use,
                input_data_format=args.input_data_format,
                shuffle=shuffle_train,
                additional_arrays=additional_arrays,
                delta_t_mean=args.delta_t_mean,
                delta_t_std=args.delta_t_std
            )

            test_input_fn = lambda: estimator_model.input_fn(
                data_dir=data_dir,
                subset="test",
                num_shards=num_shards,
                batch_size=args.validation_batch_size,
                X_cols_to_use=args.X_cols_to_use,
                input_data_format=args.input_data_format,
                shuffle=shuffle_test,
                additional_arrays=additional_arrays,
                delta_t_mean=args.delta_t_mean,
                delta_t_std=args.delta_t_std
            )

            if not args.final_eval_multiple_models:

                # Find best checkpoint and its associated metrics
                best_checkpoint_path, best_checkpoint_metrics = estimator_checkpointing.best_checkpoint(model_dir=args.model_dir,
                                                                                                        eval_dir=estimator.eval_dir(),
                                                                                                        metric=args.key_metrics[0])
                print("Best checkpoint: {}".format(best_checkpoint_path))
                print("Best metrics: {}".format(best_checkpoint_metrics))

                # Remove model_dir from previous run_config as that causes evaluation to ignore warm_start_from
                eval_run_config = deepcopy(run_config)
                setattr(eval_run_config, "_model_dir", None)

                # New estimator restarted with best result for user-specified metric
                estimator = estimator_model.create_estimator(eval_run_config, hparams, warm_start_from=best_checkpoint_path)

                train_results = estimator.evaluate(input_fn=train_input_fn, steps=steps_in_epoch_train)
                print("Final evaluation on train subset: {}".format(train_results))

                test_results = estimator.evaluate(input_fn=test_input_fn, steps=steps_in_epoch_test)
                print("Final evaluation on test subset: {}".format(test_results))

            else:
                estimator_checkpointing.evaluate_multiple_checkpoints(model_dir=args.model_dir,
                                                                      eval_dir=estimator.eval_dir(),
                                                                      num_checkpoints=args.keep_checkpoint_max,
                                                                      metric=args.key_metrics[0],
                                                                      input_fn=test_input_fn,
                                                                      run_config=run_config,
                                                                      hparams=hparams,
                                                                      num_steps_in_eval=steps_in_epoch_test if not args.overwrite else 1)

        if args.clear_checkpoints:

            rm_graph_command = "for f in $(find {} -name 'graph.pbtxt'); do rm $f; done".format(str(model_dir))
            rm_checkpoints_command = "for f in $(find {} -name 'model.ckpt-*'); do rm $f; done".format(str(model_dir))

            process = subprocess.run(rm_graph_command, shell=True, check=True)
            process = subprocess.run(rm_checkpoints_command, shell=True, check=True)

            print("Cleared model_dir: {}".format(str(model_dir)))

if __name__ == "__main__":
    cpu_monitor, gpu_monitor = start_cpu_gpu_profiling_from_argv()
    main()
    end_cpu_gpu_profiling(cpu_monitor, gpu_monitor)