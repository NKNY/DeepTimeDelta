import json
import os
import tempfile
import shutil

from io import BytesIO

import boto3
import numpy as np
import tensorflow as tf

from model.estimator.estimator_utils import remove_device_placement_from_uncompressed_model

def sagemaker_postprocessing(args):

    """
    Execute all code required after a successful training of a sagemaker training script

    """

    OUTPUT_MODEL_ID = "0" * 9 + "1"

    export_local_path = move_export(args)
    # Replicate the model structure where the export is located e.g. /opt/ml/model/export/Servo/{version}
    # and create a sibling folder {OUTPUT_MODEL_ID} alongside {version}
    export_local_path_output = os.path.join(os.path.dirname(export_local_path), OUTPUT_MODEL_ID)
    # Remove harcoded device placement
    remove_device_placement_from_uncompressed_model(export_local_path, export_local_path_output)
    # Remove the model with hardcoded device placement
    remove_dir(export_local_path)
    # Return the local path where the model without hardcoded placement exists
    return export_local_path_output

def move_export(args):

    """
    Sagemaker requires you to move the final model folder to SM_MODEL_DIR, following which sagemaker
    tars it and saves it to s3 location indicated using output_path.

    Parameters
    ----------
    args : tf.contrib.training.HParams
        Tensorflow object containing all training parameters, including the key metric which is used in
        the export path

    Returns
    -------

    """

    # TODO Handle cases where no export had been produced (e.g. no improvements in the metric)
    # S3 location containing model artifacts
    s3_model_dir = args.model_dir
    # Folder where sagemaker wants all the model artifacts to be
    local_model_dir = python_endpoint_path()
    # Name of the folder inside s3:// ... /model_dir/export
    key_metric = args.key_metrics[0]
    # Combine into one path
    export_parent = os.path.join(s3_model_dir, "export", key_metric)
    # Pull saved model from S3 to local model_dir
    # subprocess.call(["aws", "s3", export_parent, local_model_dir, "--recursive"])
    download_s3_folder(export_parent, local_model_dir)
    # TODO Verify something has actually been downloaded. No download that doesn't error still gets to the print statement.
    print("Download successful")

    # Find the version number of the model
    export_id = find_last_child_folder(local_model_dir)
    # Combine into one path
    export_local_path = os.path.join(local_model_dir, export_id)
    # Return the local path of the model
    return export_local_path

def python_endpoint_path():

    """

    As described in https://github.com/aws/sagemaker-python-sdk/issues/599 TensorFlowModel 'Python' endpoints need the
    'export/Servo/{version}/saved_model.pb' structure, whereas 'Tensorflow' endpoints for Model can have
    structure of '{any_path}/{version}/saved_model.pb'. This method generates the TensorFlowModel-adhering folder.

    Returns
    -------
    str
        Local path to which the model can be saved

    """

    return os.path.join(os.environ.get("SM_MODEL_DIR"), "export", "Servo")

def remove_dir(dir):
    shutil.rmtree(dir)

def find_first_child_folder(dir):
    return sorted(next(os.walk(dir))[1])[0]

def find_last_child_folder(dir):
    return sorted(next(os.walk(dir))[1])[-1]

def download_s3_folder(s3_path, local_path):

    """
        Given the name of the bucket, s3 key and a local path
        downloads all files which contain the s3 key to the local path

        Parameters
        ----------
        bucket : str
            Name of the bucket where input data is stored.
        s3_path : str
            Path relatively to the bucket whose all "children" are downloaded locally.
        local_path : str
            Absolute or relative path where all the downloaded files are stored.

        Returns
        ----------
        None

        """

    s3 = boto3.resource('s3')
    bucket = s3.Bucket("ci-data-apps")
    prefix = os.path.join(*(s3_path.split(os.path.sep)[3:]))
    # Removes first slash
    get_tail = lambda x: x[len(prefix)+1:]
    objs = bucket.objects.filter(Prefix=prefix)
    s3_paths = [x.key for x in objs if "." in x.key]
    for s3_path in s3_paths:
        # Removes first slash
        tail = get_tail(s3_path)
        path, filename = os.path.split(tail)
        try:
            os.makedirs(os.path.join(local_path, path))
        except FileExistsError:
            pass
        save_path = os.path.join(local_path, tail)
        bucket.download_file(s3_path, save_path)
        print("Downloaded {} to {}".format(s3_path, save_path))

def cleanup_filter(obj):

    """
    Returns true if object needs to be deleted.

    Parameters
    ----------
    obj : S3 object
    Returns
    -------
    bool
        True if object has an attribute which sets it up for deletion.
    """
    key = obj.key

    conditions = [
        ".ckpt" in key,
        "/export/" in key
    ]

    return any(conditions)

def cleanup_s3_folder(s3_path, filter_fun=cleanup_filter):

    """
    Given an s3 location removes all items that are highlighted by the filter_fun.

    Parameters
    ----------
    s3_path : str
        S3 location whose children need to be deleted.
    filter_fun : method
        Function that when given an object from S3 bucket returns true if that object should be deleted.
    Returns
    -------

    """

    s3 = boto3.resource('s3')
    bucket = s3.Bucket("ci-data-apps")

    # Remove s3://bucket_name/ from the rest of the path
    prefix = os.path.join(*(s3_path.split(os.path.sep)[3:]))
    # Get all objects from the bucket whose key contains the folder being cleared
    objs = bucket.objects.filter(Prefix=prefix)
    # Get a list of objects that need to be removed based on filter_fun
    obj_to_remove = {'Objects': [{"Key": o.key} for o in objs if filter_fun(o)]}
    # Delete objects.
    s3.meta.client.delete_objects(Bucket="MyBucket", Delete=obj_to_remove)


def load_npy(resource, bucket, key):

    """
    Loads a numpy file from S3.

    Parameters
    ----------
    resource : boto3.resource
        S3 resource made by calling boto3.resource("s3")
    bucket : str
        Bucket containing the data
    key : str
        Location inside the bucket containing the data

    Returns
    -------
    np.ndarray
    """

    obj = resource.Object(bucket, key)
    with BytesIO(obj.get()["Body"].read()) as f:
        return np.load(f)


def load_sharded_npy(resource, bucket, folder_key, concatenate=True):

    """
    Given an s3 key downloads all objects inside it and tries to load them as numpy arrays. If `concatenate` combines
    all entries along the 0th axis (doesn't increase the rank), otherwise returns a list of the arrays.

    Parameters
    ----------
    resource : boto3.resource
        S3 resource made by calling boto3.resource("s3")
    bucket : str
        Bucket containing the data
    folder_key : str
        Location inside the bucket containing 1 or more files.
    concatenate : bool
        Whether to concatenate the outputs along the 0th dimension.
    Returns
    -------
    np.ndarray | list
    If `concatenate` is True returns a concatenation along the first dimension of the arrays, otherwise returns
    a list of individual numpy arrays.

    """

    shards = [
        load_npy(resource, bucket, key) for key in sorted([
            s.key for s in resource.Bucket(bucket).objects.filter(Prefix=folder_key)
        ],
            key=lambda x: int(x.split("/")[-1].split(".")[0]))
    ]
    if concatenate:
        return np.concatenate(shards)
    return shards


def object_is_ndarray(obj):

    """
    Returns True if object is a numpy array, else False.

    Parameters
    ----------
    obj : any
        Object whose type needs to be queries

    Returns
    -------
    bool
        True if object is instance of numpy array else False
    """

    return isinstance(obj, np.ndarray)

def write_dict(d, s3_output_path):

    """
    Write dictionary object to s3. If any item in the dict is a numpy array it is converted to a list to make it
    json-serialisable.

    Parameters
    ----------
    d : dict
        Dictionary to be written to s3
    s3_output_path : str
        S3 path to which directly write the dictionary

    Returns
    -------
    None

    """

    d = {k: v.tolist() if object_is_ndarray(v) else v for k, v in d.items()}
    temp = tempfile.NamedTemporaryFile("w")
    json.dump(d, temp)
    temp.seek(0)
    temp_file_name = temp.name

    output_bucket, output_key = s3_path_to_bucket_key(s3_output_path)
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(temp_file_name, output_bucket, output_key)

def s3_path_to_bucket_key(s3_path):

    """

    Takes path of form s3://bucket/k/e/y and returns bucket and k/e/y

    Parameters
    ----------
    s3_path : str
        S3 path

    Returns
    -------
    (str, str)
        Bucket and key from the string
    """

    return [x[::-1] for x in os.path.split(os.path.split(s3_path[::-1])[0])][::-1]

def predict_s3_numpy(saved_model_path, input_s3_path, output_s3_path, batch_size):

    """

    Downloads the numpy-format input data, combines it into one big array (be careful with memory),
    loads a predictor object from an exported model, feeds batches of input data to get outputs.
    Writes the outputs as jsons to s3.

    Parameters
    ----------
    saved_model_path : str
        Local model path containing graph.pb(txt) and variables
    input_s3_path : str
        S3 path whose children are folders containing numpy arrays (e.g. input_s3_path/X/X_0.npy)
    output_s3_path : str
        S3 path where to write the output serialized as jsons (output_s3_path/0.json)
    batch_size : int
        Num sequences to be processed at once (except last batch, which may be smaller)

    Returns
    -------
        None
    """

    # Object to interact with S3
    s3 = boto3.resource("s3")
    # Extract the bucket and folder key for the input
    input_bucket, input_key = s3_path_to_bucket_key(input_s3_path)
    # Hardcoded names of arrays that the model expects as input
    array_names = ["X", "seq_lens"]
    arrays = {}
    # For each of the input arrays download all of its components under the appropriate S3 path
    for a_n in array_names:
        array_key = os.path.join(input_key, a_n)
        arrays[a_n] = load_sharded_npy(resource=s3, bucket=input_bucket, folder_key=array_key)

    # Total number of input sequences
    num_inputs = arrays["seq_lens"].shape[0]

    # Load the predictor from the trained model
    predictor = tf.contrib.predictor.from_saved_model(saved_model_path)

    # Sample batches, extract predictions, make them json serialisable, upload to S3
    for batch_start in range(0, num_inputs, batch_size):
        batch_end = np.min((batch_start + batch_size, num_inputs))
        batch = {k:v[batch_start:batch_end] for k,v in arrays.items()}
        predictions = predictor(batch)
        output_s3_file_path = os.path.join(output_s3_path, "{}.json".format(batch_start))
        write_dict(predictions, output_s3_file_path)