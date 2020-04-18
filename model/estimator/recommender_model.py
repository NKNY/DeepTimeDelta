import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer
import numpy as np

from model.cells.time_delta_cell import TimeDeltaCell


class TimeDeltaModel():

    def __init__(self, config, is_training, tower_batch_size, variable_strategy):

        """
        Time Delta Model abstraction. Upon initialisation all parameters are transferred and forward_pass can then
        be used to calculate logits for every timestep for every sequence in the batch.

        Parameters
        ----------
        config : tf.contrib.training.HParams
            Object containing all specifics about the model e.g. num_hidden and keep probabilities.
        is_training : bool
            True if is training, in which case DropOut is applied
        tower_batch_size : int
            Size of the batch assigned to this tower
        variable_strategy : str
            CPU' if cell variables should be placed on the CPU or 'GPU' if cell variables should be on the GPU.
        """

        self._timesteps = config.timesteps
        self._num_units = config.num_units
        # Here batch size refers to the full batch divided between towers
        self._batch_size = tower_batch_size

        self._num_users = config.num_users
        self._num_items = config.num_items

        self._item_keep_prob = config.item_keep_prob
        self._user_keep_prob = config.user_keep_prob
        self._time_keep_prob = config.time_keep_prob

        self._weight_initializer = tf.initializers.random_uniform(-config.initializer_range, config.initializer_range)
        self._bias_initializer = tf.initializers.random_uniform(-config.initializer_range, config.initializer_range)

        self._config = config
        self._cell = eval(config.cell_type)
        self._cell_type = config.model_id
        self._time_embedding_depth = config.time_embedding_depth
        self._time_non_lin_fun = eval(config.time_non_lin_fun)
        self._trainable_zero_state = config.trainable_zero_state

        self._variable_strategy = variable_strategy
        self._is_training = is_training
        self._use_dynamic_rnn = config.use_dynamic_rnn
        self._place_cell_vars_on_cpu = config.place_cell_vars_on_cpu
        self._user_embedding_device = config.user_embedding_device

    def forward_pass(self, uid, iid, seq_lens, sampling=None, delta_t=None, weights=None):

        """
        Perform a forward pass on the data. Namely:
            * Get item, user (and time) embeddings for each timestep for each seq in batch.
            * If training apply dropout to each of the features independently with specified keep probability.
            * Perform the RNN calculation for every timestep (all sequences at the same time)
              returning a collection of hidden states for each timestep - one per sequence.
            * Apply a final Wx+b on the each of the timesteps leading to logit scores for all items for
              each timestep for each sequence in batch.
        Parameters
        ----------
        uid : tf.int32
            Tensor of shape batch_size x timesteps where each value specifies which user performed the interaction.
        iid : tf.int32
            Tensor of shape batch_size x timesteps where each value specifies which item was consumed at that point.
        seq_lens : tf.int32
            Tensor of shape batch_size where each value specifies what is the length of the un-padded sequence.
            This is important for masking some sequences where no action is taken anymore but it had to be padded
            with fake consumption events in order to have a tensor.
        sampling : dict
            If provided forward pass returns LOSS instead of LOGITS. In this case `sampling` should contain
            two entries: `labels` - a tensor of shape batch_size x timesteps; and `num_sampled` - tf.int32 specifying
            how many negative labels should be sampled to calculate the loss.
        delta_t : tf.int32
            Tensor of shape batch_size x timesteps where each value specifies the time that it took a given user
            from his previous interaction to perform the current interaction.
        Returns
        -------
        logits : tf.float32
            Tensor of shape batch_size x timesteps x num_items where each value is a score assigned to an item
            for that timestep in that sequence.
        OR
        reduced_loss : tf.float32
            A scalar indicating the sampled softmax loss (accounts for masking). Should only be calculated during
            training - during validation loss should be calculated on all items.

        """

        # batch_size, timesteps - determined by the shape of the actual input data. Useful when training and prediction
        # input_fn's feed in data of different length/batch_size.
        dynamic_batch_size_timesteps = tf.shape(uid)

        # Instantiate the cell
        gru_cell = self._cell(self._num_units,
                              self._cell_type,
                              kernel_initializer=self._weight_initializer,
                              bias_initializer=self._bias_initializer,
                              estimator_variable_strategy=self._variable_strategy)

        with tf.name_scope("Zero_state"):
            # Set initial states for each sequence in a batch to 0
            if not self._trainable_zero_state:
                initial_state = gru_cell.zero_state(self._batch_size, dtype=tf.float32)
            # Trainable zero state
            else:
                initial_state = self.initialize_zero_state(dtype=tf.float32)

        with tf.name_scope("Embedding_layer"):
            item_embeddings = tf.get_variable("item_embeddings",
                                              shape=(self._num_items, self._num_units),
                                              dtype=tf.float32,
                                              initializer=self._weight_initializer)
            if self._user_embedding_device is not None:
                with tf.device(self._user_embedding_device):
                    user_embeddings = tf.get_variable("user_embeddings",
                                                      shape=(self._num_users, self._num_units),
                                                      dtype=tf.float32,
                                                      initializer=self._weight_initializer)
            else:
                user_embeddings = tf.get_variable("user_embeddings",
                                                  shape=(self._num_users, self._num_units),
                                                  dtype=tf.float32,
                                                  initializer=self._weight_initializer)

            selected_item_embeddings = tf.nn.embedding_lookup(item_embeddings,
                                                              iid,
                                                              name="item_embedding_lookup")
            selected_user_embeddings = tf.nn.embedding_lookup(user_embeddings,
                                                              uid,
                                                              name="user_embedding_lookup")

            if delta_t is not None:
                # Flatten the time delta to batch_size*timesteps 1-dimensional vector
                time_gap_input_flat = tf.reshape(tf.cast(delta_t, tf.float32), [-1, 1])
                # batch_size*timesteps x num_units
                time_gap_embeddings_flat = self.time_embedding(time_gap_input_flat, self._time_embedding_depth)
                # batch_size x timesteps x num_units
                time_gap_embeddings = tf.reshape(tensor=time_gap_embeddings_flat,
                                                 shape=[dynamic_batch_size_timesteps[0], dynamic_batch_size_timesteps[1], self._num_units],
                                                 name="time_gap_embeddings"
                                                 )
            with tf.name_scope("Dropout"):
                if self._is_training:
                    if self._item_keep_prob < 1:
                        selected_item_embeddings = tf.nn.dropout(x=selected_item_embeddings,
                                                                 keep_prob=self._item_keep_prob)
                    if self._user_keep_prob < 1:
                        selected_user_embeddings = tf.nn.dropout(x=selected_user_embeddings,
                                                                 keep_prob=self._user_keep_prob)
                    if delta_t is not None and self._time_keep_prob < 1:
                        time_gap_embeddings = tf.nn.dropout(x=time_gap_embeddings,
                                                            keep_prob=self._time_keep_prob)
            # Concatenate along the embedding axis
            embeddings_list = [selected_item_embeddings, selected_user_embeddings]

            if delta_t is not None:
                embeddings_list.append(time_gap_embeddings)

            # Combine embeddings into one tensor of shape batch_size x timesteps x embedding_size
            embeddings_concat = tf.concat(embeddings_list, 2)

        if self._use_dynamic_rnn:
            with tf.name_scope("RNN"):
                # batch_size x timesteps x num_hidden
                outputs, state = tf.nn.dynamic_rnn(cell=gru_cell,
                                                   inputs=embeddings_concat,
                                                   sequence_length=seq_lens,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32)
            with tf.name_scope("Output_layer_Dynamic_RNN"):
                # batch_size * timesteps x num_hidden
                outputs = tf.reshape(outputs, [dynamic_batch_size_timesteps[0]*dynamic_batch_size_timesteps[1], self._num_units])
        else:
            with tf.name_scope("RNN"):
                # timesteps x batch_size x num_hidden
                outputs, state = tf.nn.static_rnn(cell=gru_cell,
                                                  inputs=tf.unstack(embeddings_concat,
                                                                    self._timesteps,
                                                                    axis=1),
                                                  initial_state=initial_state,
                                                  sequence_length=seq_lens,
                                                  dtype=tf.float32)

            with tf.name_scope("Output_layer_Static_RNN"):
                # batch_size * timesteps x num_hidden
                outputs = tf.reshape(tf.transpose(outputs, perm=[1,0,2]), [-1, self._num_units])

        with tf.name_scope("Output_layer"):

            output_w = tf.get_variable("output_w", shape=[self._num_units, self._num_items],
                                       dtype=tf.float32, initializer=self._weight_initializer)

            output_b = tf.get_variable("output_b", shape=[self._num_items],
                                       dtype=tf.float32, initializer=self._bias_initializer)

            # Full softmax during evaluation/prediction
            if sampling is None:
                print("No sampling softmax")
                # batch_size * timesteps x items
                matmul = tf.matmul(outputs, output_w, name="big_matmul")
                logits = tf.nn.bias_add(matmul, output_b, name="big_bias_add")

                # batch_size x timesteps x items
                logits = tf.reshape(logits, [dynamic_batch_size_timesteps[0], dynamic_batch_size_timesteps[1], self._num_items], name="logits")

                return logits

            # Sampled softmax during training
            else:
                print("Sampling softmax")
                # Binary mask specifying which entries to remove as they came from padding
                # Shape batch_size*timesteps (one-dimensional)
                mask = tf.reshape(tf.sequence_mask(seq_lens, self._timesteps, dtype=tf.bool), [-1], name="sampled_softmax_mask")

                # Remove padded timesteps (i.e. no loss is calculated for them)
                sliced_outputs = tf.boolean_mask(outputs, mask, name="sampled_softmax_masked_inputs")

                # batch_size*timesteps
                labels = tf.reshape(sampling["labels"], [-1],name="reshaped_labels")
                # batch_size*timesteps minus sum of all padding applied for the current batch
                slices_labels = tf.expand_dims(tf.boolean_mask(labels, mask, name="sampled_softmax_masked_labels"), 1)

                # Calculate the loss per sequence (batch_size*timesteps - padding)
                loss = tf.nn.sampled_softmax_loss(weights=tf.transpose(output_w),
                                                  biases=output_b,
                                                  labels=slices_labels,
                                                  inputs=sliced_outputs,
                                                  num_sampled=sampling["num_sampled"],
                                                  num_classes=self._num_items,
                                                  num_true=1,
                                                  name="sampled_softmax_loss")
                if weights is not None:
                    weights = tf.reshape(weights, [-1], name="padded_reshaped_weights_mask")
                    weights_mask = tf.boolean_mask(weights, mask, name="unpadded_weights_mask")
                    loss = tf.multiply(loss, weights_mask, name="weighted_loss")

                # Loss for the batch
                reduced_loss = tf.reduce_sum(loss, name="sampled_softmax_reduced_loss")

                return reduced_loss

    def time_embedding(self, time_gaps_flat, depth=1):

        """

        Takes a batch_size*timesteps one-dimensional tensor, applies transformations and returns the
        batch_size*timesteps x num_units sized two-dimensional tensor representing the embedding of
        each of the delta_t values provided.

        Time embedding works as follows:
            * Given depth=0 only a linear transformation of transforming a single value from a scalar to a
              vector of size num_hidden (Wt+b) is applied.
            * Given depth=1 the same linear transformation is applied in addition to a single non-linear
              transformation (this non-linear transformation is defined during initialisation of the model.
            * Given depth>1 additional rounds of nonlin(Wx+b) are applied where x is the result of the
              previous layer. All additional matrices after the first one are of shape num_hidden x num_hidden.

        Parameters
        ----------
        time_gaps_flat : tf.int32
            A flattened batch_size*timesteps one-dimensional tensor representing time gaps between timesteps.
        depth : int
            Expects an int >= 0 that defines the number of times non linear transformations applied to the time
            embedding (in addition to in between linear transformations).

        Returns
        -------
        time_gap_embeddings_flat : tf.float32
            Tensor of shape batch_size x timesteps x num_units where values along the last dimension represent
            the embedding of the time gap of that timestep in that sequence in the batch.
        """
        time_gap_embeddings_flat = time_gaps_flat

        time_gap_weights = []
        time_gap_biases = []

        # If depth == 0, first subscript is 0. If depth > 0, first subscript is 1. Weight 0/1 is the 0th entry
        # in the list, weight 2 is the 1st entry, weight 3 is the 2nd etc.
        first_layer = 1 if depth > 0 else 0
        time_gap_weights.append(tf.get_variable("time_w_{}".format(first_layer), shape=[1, self._num_units],
                                                 dtype=tf.float32, initializer=self._weight_initializer))
        time_gap_biases.append(tf.get_variable("time_b_{}".format(first_layer), shape=[self._num_units],
                                                dtype=tf.float32, initializer=self._bias_initializer))

        time_gap_embeddings_flat = tf.matmul(time_gap_embeddings_flat, time_gap_weights[0],
                                             name="time_gap_matmul_{}".format(first_layer))
        time_gap_embeddings_flat = tf.nn.bias_add(time_gap_embeddings_flat, time_gap_biases[0],
                                                  name="time_gap_bias_{}".format(first_layer))

        # When depth is 0 eq 1 will be /xi = sigm(... + W_1(W_0*t+b_0)+b_1 == W_x(t)+b_x)
        if depth > 0:
            time_gap_embeddings_flat = self._time_non_lin_fun(time_gap_embeddings_flat,
                                                              name="time_gap_nonlinear_1")

        for layer_array_idx in range(1, depth):

            layer_name = layer_array_idx + 1

            time_gap_weights.append(tf.get_variable("time_w_{}".format(layer_name), shape=[self._num_units, self._num_units],
                                                     dtype=tf.float32, initializer=self._weight_initializer))
            time_gap_biases.append(tf.get_variable("time_b_{}".format(layer_name), shape=[self._num_units],
                                                    dtype=tf.float32, initializer=self._bias_initializer))
            time_gap_embeddings_flat = tf.matmul(time_gap_embeddings_flat, time_gap_weights[layer_array_idx],
                                                 name="time_gap_matmul_{}".format(layer_name))
            time_gap_embeddings_flat = tf.nn.bias_add(time_gap_embeddings_flat, time_gap_biases[layer_array_idx],
                                                      name="time_gap_bias_{}".format(layer_name))
            time_gap_embeddings_flat = self._time_non_lin_fun(time_gap_embeddings_flat,
                                                              name="time_gap_nonlinear_{}".format(layer_name))
        return time_gap_embeddings_flat

    def initialize_zero_state(self, name="single_zero_state", dtype=tf.float32):

        """
        Creates a 1-dimensional vector of size num_units and then copies these values batch_size times.
        This allows to have the zero for each sequence in the batch to be the same.

        Parameters
        ----------
        name : str
            Name of the num_units-dimensional vector
        dtype : tf.Dtype
            The variable type of the tensor.
        Returns
        -------
        tiled : tf.Tensor
            Tensor of shape batch_size x num_units where each entry along the first dimension is a num_units-sized
            copy of all other values.
        """
        one_dim = tf.get_variable(name=name,
                                  shape=(self._num_units),
                                  initializer=self._weight_initializer,
                                  dtype=dtype,
                                  )
        expanded = tf.expand_dims(one_dim, 0, name="expand_zero_state")
        tiled = tf.tile(expanded, [self._batch_size, 1], name="tiled_zero_state")
        return tiled