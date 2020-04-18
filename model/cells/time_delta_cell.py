import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base as base_layer

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

eq1_models = [1, 4, 5, 7]
eq2_models = [2, 4, 6, 7]
eq3_models = [3, 5, 6, 7]

class TimeDeltaCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 model_id,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 num_gpus=None,
                 place_cell_vars_on_cpu=None,
                 estimator_variable_strategy=None,
                 name=None):
        super(TimeDeltaCell, self).__init__(_reuse=reuse, name=name)

        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._num_gpus = num_gpus
        self._place_cell_vars_on_cpu = place_cell_vars_on_cpu
        self._estimator_variable_strategy = estimator_variable_strategy
        self._modifying_eq1 = model_id in eq1_models
        self._modifying_eq2 = model_id in eq2_models
        self._modifying_eq3 = model_id in eq3_models
        self._model_id = model_id

    @property
    def state_size(self):
        return self._num_units

    @ property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):

        # Hacky support for both estimator and multi-gpu native tensorflow
        if self._estimator_variable_strategy == "CPU":
            device = "/cpu:0"
        elif self._estimator_variable_strategy == "GPU":
            device = "/device:GPU:0"
        # This is never reached with estimator setup 17.04.19 as estimator_variable_strategy is CPU or GPU always
        elif self._num_gpus > 1 or self._place_cell_vars_on_cpu:
            device = "/cpu:0"
        else:
            device = "/device:GPU:0"
        print("Cell built on device: {}".format(device))
        with tf.device(device):

            ### USER GATE WEIGHT
            self._user_gate_h_weight = self.add_variable(
                "user_gate_h_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._user_gate_i_weight = self.add_variable(
                "user_gate_i_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._user_gate_u_weight = self.add_variable(
                "user_gate_u_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._user_gate_h_bias = self.add_variable(
                "user_gate_h_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            self._user_gate_i_bias = self.add_variable(
                "user_gate_i_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            self._user_gate_u_bias = self.add_variable(
                "user_gate_u_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            if self._modifying_eq1:

                self._user_gate_t_weight = self.add_variable(
                    "user_gate_t_weight",
                    shape=[self._num_units, self._num_units],
                    initializer=self._kernel_initializer
                )

                self._user_gate_t_bias = self.add_variable(
                    "user_gate_t_bias",
                    shape=[self._num_units],
                    initializer=self._bias_initializer
                )

            ### HIDDEN STATE GATE
            hidden_state_height_multiplier = 4 if self._modifying_eq2 else 3

            self._h_gate_weight = self.add_variable(
                "h_gate_weight",
                shape=[hidden_state_height_multiplier * self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer
            )

            self._h_gate_bias = self.add_variable(
                "h_gate_bias",
                shape=[2 * self._num_units],
                initializer=self._bias_initializer
            )

            ### UPDATE VECTOR

            self._update_vector_h_weight = self.add_variable(
                "update_vector_h_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._update_vector_i_weight = self.add_variable(
                "update_vector_i_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._update_vector_u_weight = self.add_variable(
                "update_vector_u_weight",
                shape=[self._num_units, self._num_units],
                initializer=self._kernel_initializer
            )

            self._update_vector_h_bias = self.add_variable(
                "update_vector_h_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            self._update_vector_i_bias = self.add_variable(
                "update_vector_i_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            self._update_vector_u_bias = self.add_variable(
                "update_vector_u_bias",
                shape=[self._num_units],
                initializer=self._bias_initializer
            )

            if self._modifying_eq3:

                self._update_vector_t_weight = self.add_variable(
                    "update_vector_t_weight",
                    shape=[self._num_units, self._num_units],
                    initializer=self._kernel_initializer
                )

                self._update_vector_t_bias = self.add_variable(
                    "update_vector_t_bias",
                    shape=[self._num_units],
                    initializer=self._bias_initializer
                )

            self.built = True

    def call(self, inputs, state):
        # User gating

        # batch_size x num_units
        item_embedding = inputs[:, :self._num_units]
        user_embedding = inputs[:, self._num_units:2*self._num_units]
        time_embedding = inputs[:, 2*self._num_units:]

        # batch_size x num_units
        user_gate_h = math_ops.matmul(state, self._user_gate_h_weight, name="u_gate_h_matmul")
        user_gate_i = math_ops.matmul(item_embedding, self._user_gate_i_weight, name="u_gate_i_matmul")
        user_gate_u = math_ops.matmul(user_embedding, self._user_gate_u_weight, name="u_gate_u_matmul")

        # batch_size x num_units
        user_gate_h = nn_ops.bias_add(user_gate_h, self._user_gate_h_bias, name="u_gate_h_bias_add")
        user_gate_i = nn_ops.bias_add(user_gate_i, self._user_gate_i_bias, name="u_gate_i_bias_add")
        user_gate_u = nn_ops.bias_add(user_gate_u, self._user_gate_u_bias, name="u_gate_u_bias_add")

        to_sum_eq1 = [user_gate_h, user_gate_i, user_gate_u]

        if self._modifying_eq1:
            user_gate_t = math_ops.matmul(time_embedding, self._user_gate_t_weight, name="u_gate_t_matmul")
            user_gate_t = nn_ops.bias_add(user_gate_t, self._user_gate_t_bias, name="u_gate_t_bias_add")
            to_sum_eq1.append(user_gate_t)

        # batch_size x num_units
        xi = math_ops.sigmoid(math_ops.add_n(to_sum_eq1), name="xi")

        # batch_size x num_units
        one_minus_xi = math_ops.subtract(1., xi, name="one_minus_xi")

        # Hidden state gating
        gated_i = math_ops.multiply(one_minus_xi, item_embedding, name="gated_i")
        gated_u = math_ops.multiply(xi, user_embedding, name="gated_u")

        # batch_size x 3*num_units
        to_concat_eq2 = [state, gated_i, gated_u]

        if self._modifying_eq2:
            to_concat_eq2.append(time_embedding)

        concat = tf.concat(to_concat_eq2, 1, name="concat_for_eq_two")

        # batch_size x 2*num_units
        u_r = math_ops.mat_mul(concat, self._h_gate_weight, name="matmul_u_r")
        u_r = math_ops.sigmoid(nn_ops.bias_add(u_r, self._h_gate_bias), name="sigmoid_u_r")

        # batch_size x num_units
        u, r = array_ops.split(u_r, num_or_size_splits=2, axis=1, name="split_u_r")
        one_minus_u = math_ops.subtract(1., u, name="one_minus_u")

        # Update vector
        # batch_size x num_units
        gated_h = math_ops.multiply(r, state, name="gated_h")

        update_vector_h = math_ops.mat_mul(gated_h, self._update_vector_h_weight, name="update_vector_h_matmul")
        update_vector_i = math_ops.mat_mul(gated_i, self._update_vector_i_weight, name="update_vector_i_matmul")
        update_vector_u = math_ops.mat_mul(gated_u, self._update_vector_u_weight, name="update_vector_u_matmul")

        update_vector_h = nn_ops.bias_add(update_vector_h, self._update_vector_h_bias, name="update_vector_h_bias_add")
        update_vector_i = nn_ops.bias_add(update_vector_i, self._update_vector_i_bias, name="update_vector_i_bias_add")
        update_vector_u = nn_ops.bias_add(update_vector_u, self._update_vector_u_bias, name="update_vector_u_bias_add")

        to_sum_eq3 = [update_vector_h, update_vector_i, update_vector_u]

        if self._modifying_eq3:
            update_vector_t = math_ops.mat_mul(time_embedding, self._update_vector_t_weight, name="update_vector_t_matmul")
            update_vector_t = nn_ops.bias_add(update_vector_t, self._update_vector_t_bias, name="update_vector_t_bias_add")
            to_sum_eq3.append(update_vector_t)

        k = math_ops.tanh(math_ops.add_n(to_sum_eq3), name="k")

        # Update hidden state
        new_h = math_ops.add(math_ops.multiply(one_minus_u, state), math_ops.multiply(u, k), name="new_h")

        return new_h, new_h