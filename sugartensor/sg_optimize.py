from __future__ import absolute_import
import tensorflow as tf

import sugartensor.per_example_gradients
import sugartensor.dp_utils as utils


# noinspection PyAbstractClass
class AdaMaxOptimizer(tf.train.Optimizer):
    r"""Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

    excerpted from https://github.com/openai/iaf/blob/master/tf_utils/adamax.py

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdaMaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


# noinspection PyAbstractClass
class MaxPropOptimizer(tf.train.Optimizer):
    r"""Optimizer that implements the MaxProp algorithm by buriburisuri@gmail.com.
    """
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t

        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


class DPGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """Differentially private gradient descent optimizer.
  """

  def __init__(self, learning_rate, eps_delta, sanitizer,
               sigma=None, use_locking=False, name="DPGradientDescent",
               batches_per_lot=1):
    """Construct a differentially private gradient descent optimizer.

    The optimizer uses fixed privacy budget for each batch of training.

    Args:
      learning_rate: for GradientDescentOptimizer.
      eps_delta: EpsDelta pair for each epoch.
      sanitizer: for sanitizing the graident.
      sigma: noise sigma. If None, use eps_delta pair to compute sigma;
        otherwise use supplied sigma directly.
      use_locking: use locking.
      name: name for the object.
      batches_per_lot: Number of batches in a lot.
    """

    super(DPGradientDescentOptimizer, self).__init__(learning_rate,
                                                     use_locking, name)

    #Also, if needed, define the gradient accumulators
    self._batches_per_lot = batches_per_lot
    self._grad_accum_dict = {}
    if batches_per_lot > 1:
      self._batch_count = tf.Variable(1, dtype=tf.int32, trainable=False,
                                      name="batch_count")
    #   var_list = tf.trainable_variables()
    #   with tf.variable_scope("grad_acc_for"):
    #     for var in var_list:
    #       v_grad_accum = tf.Variable(tf.zeros_like(var),
    #                                  trainable=False,
    #                                  name=utils.GetTensorOpName(var))
    #       self._grad_accum_dict[var.name] = v_grad_accum

    self._eps_delta = eps_delta
    self._sanitizer = sanitizer
    self._sigma = sigma


  def compute_sanitized_gradients(self, loss, var_list=None,
                                add_noise=True):
    """Compute the sanitized gradients.
    Args:
      loss: the loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """

    self._assert_valid_dtypes([loss])

    xs = [tf.convert_to_tensor(x) for x in var_list]
    px_grads = per_example_gradients.PerExampleGradients(loss, xs)
    sanitized_grads = []

    print(_sanitizer)
    raise Exception('stop')
    for px_grad, v in zip(px_grads, var_list):
      tensor_name = utils.GetTensorOpName(v)
      sanitized_grad = self._sanitizer.sanitize(
          px_grad, self._eps_delta, sigma=self._sigma,
          tensor_name=tensor_name, add_noise=add_noise,
          num_examples=self._batches_per_lot * tf.slice(
              tf.shape(px_grad), [0], [1]))
      sanitized_grads.append(sanitized_grad)

    return sanitized_grads

  def minimize(self, loss, global_step=None, var_list=None,
               name=None):
    """Minimize using sanitized gradients.
    This gets a var_list which is the list of trainable variables.
    For each var in var_list, we defined a grad_accumulator variable
    during init. When batches_per_lot > 1, we accumulate the gradient
    update in those. At the end of each lot, we apply the update back to
    the variable. This has the effect that for each lot we compute
    gradients at the point at the beginning of the lot, and then apply one
    update at the end of the lot. In other words, semantically, we are doing
    SGD with one lot being the equivalent of one usual batch of size
    batch_size * batches_per_lot.
    This allows us to simulate larger batches than our memory size would permit.
    The lr and the num_steps are in the lot world.
    Args:
      loss: the loss tensor.
      global_step: the optional global step.
      var_list: the optional variables.
      name: the optional name.
    Returns:
      the operation that runs one step of DP gradient descent.
    """

    # First validate the var_list
    print('minimize called')
    if var_list is None:
      var_list = tf.trainable_variables()
    for var in var_list:
      if not isinstance(var, tf.Variable):
        raise TypeError("Argument is not a variable.Variable: %s" % var)

    # Modification: apply gradient once every batches_per_lot many steps.
    # This may lead to smaller error

    if self._batches_per_lot == 1:
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list)

      grads_and_vars = zip(sanitized_grads, var_list)
      self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])

      tf.sg_summary_gradient(v, g)
      apply_grads = self.apply_gradients(grads_and_vars,
                                         global_step=global_step, name=name)
      return apply_grads

    # Condition for deciding whether to accumulate the gradient
    # or actually apply it.
    # we use a private self_batch_count to keep track of number of batches.
    # global step will count number of lots processed.

    update_cond = tf.equal(tf.constant(0),
                           tf.mod(self._batch_count,
                                  tf.constant(self._batches_per_lot)))

    # Things to do for batches other than last of the lot.
    # Add non-noisy clipped grads to shadow variables.

    def non_last_in_lot_op(loss, var_list):
      """Ops to do for a typical batch.
      For a batch that is not the last one in the lot, we simply compute the
      sanitized gradients and apply them to the grad_acc variables.
      Args:
        loss: loss function tensor
        var_list: list of variables
      Returns:
        A tensorflow op to do the updates to the gradient accumulators
      """
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=False)

      update_ops_list = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        update_ops_list.append(grad_acc_v.assign_add(grad))
      update_ops_list.append(self._batch_count.assign_add(1))
      return tf.group(*update_ops_list)

    # Things to do for last batch of a lot.
    # Add noisy clipped grads to accumulator.
    # Apply accumulated grads to vars.

    def last_in_lot_op(loss, var_list, global_step):
      """Ops to do for last batch in a lot.
      For the last batch in the lot, we first add the sanitized gradients to
      the gradient acc variables, and then apply these
      values over to the original variables (via an apply gradient)
      Args:
        loss: loss function tensor
        var_list: list of variables
        global_step: optional global step to be passed to apply_gradients
      Returns:
        A tensorflow op to push updates from shadow vars to real vars.
      """

      # We add noise in the last lot. This is why we need this code snippet
      # that looks almost identical to the non_last_op case here.
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=True)

      normalized_grads = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        # To handle the lr difference per lot vs per batch, we divide the
        # update by number of batches per lot.
        normalized_grad = tf.div(grad_acc_v.assign_add(grad),
                                 tf.to_float(self._batches_per_lot))

        normalized_grads.append(normalized_grad)

      with tf.control_dependencies(normalized_grads):
        grads_and_vars = zip(normalized_grads, var_list)
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars if g is not None])
        apply_san_grads = self.apply_gradients(grads_and_vars,
                                               global_step=global_step,
                                               name="apply_grads")

      # Now reset the accumulators to zero
      resets_list = []
      with tf.control_dependencies([apply_san_grads]):
        for _, acc in self._grad_accum_dict.items():
          reset = tf.assign(acc, tf.zeros_like(acc))
          resets_list.append(reset)
      resets_list.append(self._batch_count.assign_add(1))

      last_step_update = tf.group(*([apply_san_grads] + resets_list))
      return last_step_update
    # pylint: disable=g-long-lambda
    update_op = tf.cond(update_cond,
                        lambda: last_in_lot_op(
                            loss, var_list,
                            global_step),
                        lambda: non_last_in_lot_op(
                            loss, var_list))
    return tf.group(update_op)
