import tensorflow as tf
from helper.decorators import variable_scope


class AdamOptimizerRememberLast(tf.train.AdamOptimizer):
    def __init__(self, *args, **kwargs):
        self._gradient_stored = {}
        super(AdamOptimizerRememberLast, self).__init__(*args, **kwargs)

    def _apply_dense(self, grad, var):
        #print(grad.get_shape().as_list())

        #print(var.name)
        #beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
        #update_beta1 = beta1_power.assign(
        #    beta1_power * self._beta1_t, use_locking=self._use_locking)
        #m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)

        previous = self._gradient_stored[var.name]
        no_grad = tf.constant(0., shape=grad.get_shape().as_list())
        condition = tf.less_equal(tf.abs(grad - previous), tf.constant(0.1))
        modified = tf.where(condition, grad, no_grad)
        apply = super()._apply_dense(modified, var)

        with tf.control_dependencies([apply]):
            return tf.assign(self._gradient_stored[var.name], grad)

    def _resource_apply_dense(self, grad, var):
        raise AttributeError()

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        #first_var = min(var_list, key=lambda x: x.name)
        #self._create_non_slot_variable(initial_value=self._beta1,
        #                               name="beta1_power",
        #                               colocate_with=first_var)

        for i, var in enumerate(var_list):
            initial = tf.constant(0., shape=var.shape)
            name = 'gradient_stored' + str(i)  # + var.name
            print(name)
            with tf.colocate_with(var):
                if var.name not in self._gradient_stored:
                    self._gradient_stored[var.name] = tf.get_variable(name, shape=var.shape, trainable=False)
            #self._create_non_slot_variable(initial, name, colocate_with=var)

        return super()._create_slots(var_list)

class FullyConnectedToLabel:
    def __init__(self, data):
        self.image = data.image_flat
        self.label = data.label

        self.prediction = self._prediction()
        self.optimize = self._optimize()
        self.error = self._error()

    @variable_scope(initializer=tf.contrib.layers.xavier_initializer())
    def _prediction(self):
        x = self.image
        x = tf.contrib.layers.fully_connected(x, 200)
        x = tf.contrib.layers.fully_connected(x, 200)
        x = tf.contrib.layers.fully_connected(x, 10, tf.nn.softmax)
        return x

    @variable_scope
    def _optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        self.cross_entropy = -tf.reduce_sum(self.label * logprob)
        self.optimizer = AdamOptimizerRememberLast(1e-2)
        #self.optimizer = tf.train.AdamOptimizer(1e-2)
        return self.optimizer.minimize(self.cross_entropy)

    @variable_scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
