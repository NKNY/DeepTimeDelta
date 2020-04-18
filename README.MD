## DeepTimeDelta

Data preprocessing, model training and evaluation code released alongside Master's Thesis of Norman Knyazev "Modelling 
Time Delta in User-Item Interactions Using Deep Recommender Systems", available 
[here](https://repository.tudelft.nl/islandora/object/uuid%3A34ffc4b0-6c61-4649-a628-9d1f7b9fa846?collection=education). 

`preprocessing` contains Python notebooks for preprocessing of two publically available datasets used for evaluation.

`model` features the `Tensorflow` [Estimator API](https://www.tensorflow.org/guide/estimator) implementation of our model, along with the code used to run one instance of 
training and evaluation, along with the default parameters and custom metric implementations. It also includes methods
allowing to use a trained exported model for evaluation on additional data.

`evaluation` contains code for statistical analysis of Research Questions 1, 2 and 3 as well as code for user
partitioning for Research Question 3.

---
### Requirements (tested on Python 3.6.5):

* `gputil`
* `IPython`
* `jupyterlab`
* `numpy`
* `pandas` 
* `tensorflow/tensorflow-gpu==1.13.2`


---

### Usage

* Single model training/evaluation run: ```python3 model/estimator/estimator_main.py --param_name param_value ...``` 
* Evaluation of a trained exported model: ```python3 model/estimator/estimator_evaluate_export.py --param_name 
param_value ...```
---
### Project structure
#### `preprocessing`
* `lastfm.ipynb` - generate training, validation and test data in `.npy` format for _LastFM 1K user_ dataset (with sampling).
* ` ml-10m.ipynb` - generate training, validation and test data in `.npy` format for _MovieLens 10M_ dataset.
* `npy2tfrecord.ipynb` - optional code to convert the above `.npy` files to sharded `.tfrecords` files (more optimised for `tensorflow`).

#### `model`
* ##### `cells`
    * `time_delta_cell.py` - contains `TimeDeltaCell` class, defining how to combine user, item, previous hidden 
    state and time delta embeddings based on the provided `model_id` parameter. `model_id == 0` describes the 
        baseline model (Donkers et al.).
* ##### `configs`
    * `default.json` - default runtime parameters. If no value provided to a parameter without a default value then
    the key in the resulting parameter store object maps to `None`.
* ##### `estimator`
    * `estimator_checkpointing.py` - methods used to compare multiple saved `.ckpt` files against each other in
    terms of performance.
    * `estimator_dataset.py` - methods for data loading, shuffling and consumption.
    * `estimator_evaluate_export.py` - methods forming a pipeline for evaluating an exported model on batches of 
    new data. Relies on objects from `model.evaluation.(sequence|user)_level_evaluation` to keep track of
    overall performance and performance on each user. Those objects can be pickled and saved to a path if specified.
    * `estimator_main.py` - main pipeline for training/evaluation of a model with the supplied (hyper)parameters.
    Supports execution on a local instance as well as [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
    * `estimator_model.py`- methods regulating data flow related to model training e.g. loss, regularisation, 
    gradient calculation and propagation, processing shards of data and combining the results in case of multiple 
    GPUs.
    * `estimator_sagemaker.py` - helper methods for using the model on Amazon SageMaker.
    * `estimator_utils.py` - helper methods for logging, parameter processing, early stopping etc.
    * `recommender_model.py` - defines data flow from being given a batch of input tensors (`uid`s, `iid`s etc.) to 
    softmax e.g. embeddings lookup, rnn, softmax. Note that embeddings are combined by the rules specified in
    `model.cells.time_delta_cell` 
* ##### `evaluation`
    * `sequence_level_evaluation.py` - Vectorised implementations of Recall@20, MRR@20 and `SelectedTimestepsEvaluator` 
    object allowing to evaluate models on all or a subset of interactions of a supplied set of data.
    * `user_level_evaluation.py` - UserRecall@20, UserMRR@20 and `SelectedTimestepsEvaluator` 
    object allowing to evaluate models on all or a subset of interactions of a supplied set of data.
* ##### `utils`
    * `backwards compatibility.py` - methods allowing execution using Python 2.X.
    * `datasplit.py` - methods used for partitioning the data as described in Thesis Sections 4.2 and 4.3.
    * `helper.py` - additional logging and folder/path manipulation methods.
#### `evaluation`
* ##### `RQ1`
    * `generate_stat_test_data` - processing the results data of Research Question 1, used for statistical evaluation in R.
    * `ANOVA.R` - data standardisation, statistical evaluation via ANOVAs and Dunnett's tests.
* ##### `RQ2`
    * `generate_stat_test_data` - processing the results data of Research Question 2, used for statistical evaluation in R.
    * `ANOVA.R` - data standardisation, statistical evaluation via ANOVAs and Dunnett's tests.
* ##### `RQ3`
    * `general` - ordered pipeline containing steps used for gathering the results for statistical evaluation of 
    Research Question 3 used for both methods of user partitioning. Missing numerical values indicate that notebooks
    for those steps are in `dt` and `mainstreamness` folders
    * `dt` - user partitioning on the basis of their mean time gap.
    * `mainstreamness` - user partitioning on the basis of their mainstreamness.
    * `example.json` - example structure of results data used for statistical evaluation of Research Question 1 and 2. 
    `all` refers to the performance on the whole subset of data for Research Questions 1, whereas `low`, `medium`, `high` 
    refer to the interaction groups from Research Question 2.
    
---
### Device placement during training/evaluation run
* `num_gpu == 0` will constrain training to CPU.
* `num_gpu == 1` will assign operations to GPU where applicable.
* `num_gpu > 1` will split the batch samples equally among all GPUs, each to process and return the scores and/or
gradients. Gradients are combined on the CPU and then applied to the parameters.
* If user embedding matrices too big for a single GPU memory they can be placed on a specific device (e.g. CPU)
via parameter `user_embedding_device` of the training script.
