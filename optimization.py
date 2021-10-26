import tensorflow as tf


class WarmUpThenLinDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule that first increases the learning rate linearly from 0 to initial_learning_rate
    and then decreases it linearly to zero. Note that there is a step when the schedule switches, as the
    linear decrease starts at step 0 and is therefore smaller than initial_learning_rate after the warmup steps.
    This is done to have the schedule equivalent to the one used in the original BERT code.
    See https://github.com/google-research/bert/blob/master/optimization.py for reference. (The step in learning
    rate is not explicitly mentioned there, but a closer inspection of the code reveals it).
    """
    def __init__(self, initial_learning_rate, warm_up_steps, total_steps):
        self.warm_up_steps = warm_up_steps
        self.warm_up_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0,
                                                                              decay_steps=warm_up_steps,
                                                                              end_learning_rate=initial_learning_rate)
        self.decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_learning_rate,
                                                                            decay_steps=total_steps,
                                                                            end_learning_rate=0.0)

    def __call__(self, step):
        return tf.cond(step < self.warm_up_steps,
                       lambda: self.warm_up_schedule(step), lambda: self.decay_schedule(step))


GRAD_CLIP_NORM = 1.0  # default in BERT, see https://github.com/google-research/bert/blob/master/optimization.py


class ClipAdam(tf.keras.optimizers.Adam):
    """
    Small workaround wrapper to clip gradients, as gradient clipping is otherwise ignored in Keras TF2.1 and bug
    fix will only ship with TF2.2.
    See https://github.com/tensorflow/tensorflow/commit/69da929ad4d5ba605507efa1f52b382a55b6a969 for reference.
    """
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads = []
        vars = []
        for grad, var in grads_and_vars:
            grads.append(tf.clip_by_norm(grad, GRAD_CLIP_NORM))
            vars.append(var)
        return super(ClipAdam, self).apply_gradients(zip(grads, vars), name=name, **kwargs)
